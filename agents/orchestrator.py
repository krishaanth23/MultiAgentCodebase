"""Workflow orchestrator for the Intent-Driven Medallion pipeline.

The Supervisor is itself a fully autonomous ReAct agent. For each pipeline phase
it receives the current pipeline state, THINKS about what needs to be done, PLANS
which specialist agents to call and in what order, ACTS by dispatching them as tools
with rich goal descriptions, VERIFIES each output before proceeding, and updates
the pipeline state.

Full observability is captured for the Supervisor itself (via AgentTrace) in addition
to the per-agent traces written by each specialist agent.

## Architecture

Six HITL gates: one STTM-approval gate and one output-approval gate per layer
(Bronze/Silver/Gold), so the user can inspect the materialised data itself —
not just the rules that will produce it — before the pipeline moves on.

Only Phase 1 (profiling + Bronze STTM generation) still runs as a Supervisor
ReAct loop choosing between two tools — there's a real sequencing decision
there. Every other step below is a single specialist agent invoked directly:
once the UI has already gated the step to exactly one action, routing it
through an extra Supervisor "decide what to call" LLM turn adds nothing but
another API call, so those steps call the specialist agent's entry point
directly instead.

Phase 1 — Profile & Bronze STTM (Supervisor: profiler_agent_tool, sttm_agent_tool)
    -> HITL: approve Bronze STTM
Bronze execution (direct: execute_bronze)
    -> HITL: approve Bronze output
Silver STTM generation (direct: generate_silver_sttm)
    -> HITL: approve Silver STTM
Silver execution (direct: execute_silver)
    -> HITL: approve Silver output
Gold STTM generation (direct: generate_gold_sttm)
    -> HITL: approve Gold STTM
Gold execution (direct: execute_gold)
    -> HITL: approve Gold output
Report generation (direct: generate_report)
    -> done

UI contract (streamlit_app.py calls these directly):
    run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState
    run_bronze_execution(state) -> PipelineState
    run_silver_sttm_generation(state) -> PipelineState
    run_silver_execution(state) -> PipelineState
    run_gold_sttm_generation(state) -> PipelineState
    run_gold_execution(state) -> PipelineState
    run_report_generation(state) -> PipelineState

PipelineState keys read by UI:
    run_id, status, error, sttm_bronze_path, sttm_silver_path, sttm_gold_path,
    bronze_output_paths, silver_output_paths, gold_output_paths, report_path
"""

import json
import uuid
import traceback
from typing import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.audit import AuditLogger
from core.llm import make_llm, invoke_agent_with_tool_recovery
from core.memory import store_document
from core.observability import AgentTrace
from agents.profiler import profile_multiple_datasets
from agents.sttm_generator import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from agents.bronze_agent import execute_bronze
from agents.silver_agent import execute_silver
from agents.gold_agent import execute_gold
from agents.reporter import generate_report


# ---------------------------------------------------------------------------
# Pipeline state — keys UNCHANGED, UI reads them directly
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """State flowing through the pipeline. Keys read by Streamlit UI must not change."""
    run_id: str
    status: str
    uploaded_files: list[str]
    business_intent: str
    profile_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_sttm_approved: bool
    silver_sttm_approved: bool
    gold_sttm_approved: bool
    bronze_output_paths: list[str]
    silver_output_paths: list[str]
    gold_output_paths: list[str]
    report_path: str
    error: str


# ---------------------------------------------------------------------------
# Supervisor autonomous agent prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are the Pipeline Supervisor for an Intent-Driven Medallion data pipeline.
You are a fully autonomous ReAct agent. You do NOT follow a rigid script —
you think about the pipeline state, plan what needs to happen, dispatch specialist
agents as tools, verify their outputs, and decide when the phase is complete.

## What this pipeline does
Transforms raw retail CSV data through three quality layers:
  Bronze → raw ingestion with metadata
  Silver → cleansed, typed, deduplicated data
  Gold   → analytics-ready joined and aggregated tables (intent-driven)
Then produces a business-intent-driven executive report.

## Your operating mode — follow this sequence for every phase

1. THINK: Read the phase goal carefully. What is the current state of the pipeline?
   What data is available? What needs to be produced by the end of this phase?

2. PLAN: Before calling any tool, briefly note (1-2 sentences is enough):
   - Which agents will you call, and in what order?
   - What goal will you give each agent?

3. ACT: In THIS SAME RESPONSE, immediately make the actual tool call for the
   first agent in your plan — do not end your turn after only describing the
   plan in prose. Writing "I will call profiler_agent_tool" does not call it;
   emitting a real tool call does. A response with reasoning text but zero
   tool calls is an incomplete turn, not a completed phase, UNLESS every tool
   in your plan has already been called in a previous turn.
   Give each agent a rich, specific goal description — not just "execute".
   Each tool you call launches a fully autonomous agent that will:
     * Inspect its own inputs
     * Form its own execution plan
     * Execute and verify its output
   You do not need to tell the agent HOW to do its job — just WHAT you need.

4. VERIFY: After each tool returns, check its output:
   - Did it return the expected keys (profile_path, sttm_path, output_paths, etc.)?
   - Are the paths non-empty?
   - If a tool returns an error, report it clearly and stop — do not call the next tool.

5. CONFIRM: Once all tools in the phase have completed successfully, summarise
   what was accomplished and confirm phase completion.

## Tool contract
- Every tool captures its own file paths, run IDs, and context via closure.
- You pass a `goal` parameter to each tool describing what you need.
- Tool outputs are JSON objects — read them to verify completion.
- Do NOT attempt to pass file paths from one tool to another — each tool
  resolves its own inputs from the pipeline context automatically.

## Error handling
- If any tool raises an error or returns an error key, stop immediately.
- Report the error clearly including which tool failed and what it returned.
- Do not attempt the next tool after a failure.

## Important
You are coordinating autonomous specialist agents — trust them to handle their
own execution details. Your value is in planning, sequencing, verification,
and understanding the pipeline state."""


# ---------------------------------------------------------------------------
# Phase 1 tool factory: profiler_agent_tool + sttm_agent_tool (Bronze)
# ---------------------------------------------------------------------------

def _make_phase1_tools(uploaded_files: list[str], run_id: str):
    """Build Phase 1 tools: profiler and Bronze STTM generator.

    Both tools are intent-agnostic. Scratchpad allows sttm_agent_tool to
    automatically consume the profile_path produced by profiler_agent_tool
    without the Supervisor reproducing file paths.
    """
    scratchpad: dict = {}

    @tool
    def profiler_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Data Profiler agent.

        The profiler agent will inspect the uploaded CSV files, compute column-level
        statistics, identify semantic meanings, discover potential join keys, and note
        data quality observations. It produces a combined profile JSON used by the
        STTM agent to generate transformation rules.

        Pass a goal describing what profiling is needed and why.
        Returns JSON: {"profile_path": "path/to/profile.json"}.
        Must be called before sttm_agent_tool in Phase 1.
        """
        print(f"[ORCHESTRATOR] Dispatching profiler agent | goal: {goal[:120]}")
        profile_path = profile_multiple_datasets(
            file_paths=uploaded_files,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Files to profile: {uploaded_files}\n"
                "Inspect the files first, then compute full statistics, then return "
                "semantic analysis covering all columns, join keys, and quality notes."
            ),
        )
        scratchpad["profile_path"] = profile_path
        return json.dumps({"profile_path": profile_path})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 1: generates Bronze ingestion rules (column renames, type casts,
        metadata rows) from the data profile. Bronze is intent-agnostic — every
        source column is mapped mechanically. Requires profiler_agent_tool to have run.

        Pass a goal that clearly states: which layer's STTM to generate (Bronze)
        and what the STTM will be used for.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        if "profile_path" not in scratchpad:
            return json.dumps({"error": "profiler_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Bronze) | goal: {goal[:120]}")
        sttm_path = generate_bronze_sttm(
            profile_path=scratchpad["profile_path"],
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Bronze\n"
                f"Profile path: {scratchpad['profile_path']}\n"
                "Bronze is intent-agnostic. Inspect the profile context first, then "
                "generate a complete Bronze STTM covering every column. "
                "Add _load_timestamp and _source_file metadata rows. "
                "Do NOT add a surrogate key — that belongs in Silver."
            ),
        )
        scratchpad["sttm_bronze_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return profiler_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Autonomous Supervisor runner — the orchestrator's own ReAct loop
# ---------------------------------------------------------------------------

def _run_supervisor(
    tools: list,
    phase_goal: str,
    phase_name: str,
    run_id: str,
) -> dict:
    """Instantiate the autonomous Supervisor agent and run it for one phase.

    The Supervisor thinks about the phase goal, plans which tools to call and
    in what order, dispatches them with rich goal descriptions, and verifies
    outputs. Full observability is captured via AgentTrace.

    Args:
        tools: The specialist agent tools available to the Supervisor this phase.
        phase_goal: High-level goal describing what this phase must accomplish.
        phase_name: Short name for logging (e.g. "phase1").
        run_id: Pipeline run identifier.

    Returns:
        dict: The full agent result including message history.
    """
    trace = AgentTrace(f"supervisor_{phase_name}", run_id)
    trace.set_input(
        phase=phase_name,
        goal=phase_goal,
        tools_available=[t.name for t in tools],
    )

    llm = make_llm()
    agent = create_agent(llm, tools, system_prompt=SUPERVISOR_PROMPT)

    print(f"[ORCHESTRATOR] Supervisor starting {phase_name} autonomously")
    print(f"[ORCHESTRATOR] Goal: {phase_goal[:200]}")

    try:
        result = invoke_agent_with_tool_recovery(
            agent, {"messages": [HumanMessage(content=phase_goal)]}, tools
        )
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract final supervisor summary from last AI message
    final_summary = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if type(msg).__name__ == "AIMessage" and isinstance(content, str) and content.strip():
            final_summary = content.strip()[:400]
            break

    trace.set_output(
        phase=phase_name,
        tools_called=[t["tool"] for t in trace.trace["tool_calls"]],
        summary=final_summary,
    ).complete()

    print(f"[ORCHESTRATOR] Supervisor completed {phase_name}")
    return result


# ---------------------------------------------------------------------------
# Pipeline entry points — signatures UNCHANGED, UI calls these directly
# ---------------------------------------------------------------------------

def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Phase 1: Supervisor profiles data and generates Bronze STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py with (saved_paths, business_intent).
    Returns PipelineState with sttm_bronze_path populated.
    """
    run_id = str(uuid.uuid4())
    audit = AuditLogger(run_id)
    audit.log(
        "orchestrator", "pipeline_started",
        intent=business_intent, status="started", phase="upload",
        rationale="User submitted files and intent; Supervisor will profile data then generate Bronze STTM.",
    )
    store_document(
        doc_id=f"intent_{run_id}",
        text=business_intent,
        metadata={"type": "business_intent", "run_id": run_id},
    )

    state: PipelineState = {
        "run_id": run_id,
        "status": "profiling",
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
        "profile_path": "",
        "sttm_bronze_path": "",
        "sttm_silver_path": "",
        "sttm_gold_path": "",
        "bronze_sttm_approved": False,
        "silver_sttm_approved": False,
        "gold_sttm_approved": False,
        "bronze_output_paths": [],
        "silver_output_paths": [],
        "gold_output_paths": [],
        "report_path": "",
        "error": "",
    }

    profiler_t, sttm_t, scratchpad = _make_phase1_tools(uploaded_files, run_id)

    try:
        audit.log(
            "orchestrator", "phase1_supervisor_started",
            status="in_progress", phase="phase1",
            rationale=(
                "Supervisor agent will autonomously decide how to profile the raw data "
                "and generate Bronze STTM ingestion rules. Bronze is intent-agnostic."
            ),
        )
        _run_supervisor(
            tools=[profiler_t, sttm_t],
            phase_goal=(
                f"Phase 1 goal for run_id='{run_id}'.\n\n"
                f"Uploaded files: {uploaded_files}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Profile the uploaded raw CSV files to understand their structure, "
                "column semantics, data quality, and potential join keys across datasets.\n"
                "2. Use that profile to generate a complete Bronze STTM CSV that covers "
                "every column with ingestion rules (renaming, metadata injection -- no type "
                "casting, Bronze is a faithful raw copy).\n\n"
                "Bronze is intent-agnostic — map every column mechanically. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase1",
            run_id=run_id,
        )

        # Safety net: the Supervisor is an LLM and can end its turn after only
        # describing a plan, without actually emitting the tool call(s) needed to
        # execute it (observed with smaller models) -- or a tool call it did make
        # could have hit an unrecoverable corruption (see core.llm's known
        # limitation for multi-call agents). Either way, _run_supervisor can
        # return without raising while the scratchpad stays incomplete. Detect
        # that here and complete whichever step is missing directly, rather than
        # silently handing the UI a Bronze STTM path that doesn't exist.
        profile_path = scratchpad.get("profile_path", "")
        if not profile_path:
            print("[ORCHESTRATOR] Supervisor didn't complete profiling -- completing it directly.")
            profile_path = profile_multiple_datasets(
                file_paths=uploaded_files,
                run_id=run_id,
                task_description=(
                    f"Profile the uploaded raw CSV files for run_id='{run_id}'.\n\n"
                    f"Run ID: {run_id}\n"
                    f"Files to profile: {uploaded_files}\n"
                    "Inspect the files first, then compute full statistics, then return "
                    "semantic analysis covering all columns, join keys, and quality notes."
                ),
            )
        if not profile_path:
            raise RuntimeError("Profiling produced no profile file.")

        sttm_bronze_path = scratchpad.get("sttm_bronze_path", "")
        if not sttm_bronze_path:
            print("[ORCHESTRATOR] Supervisor didn't reach Bronze STTM generation -- completing it directly.")
            sttm_bronze_path = generate_bronze_sttm(
                profile_path=profile_path,
                run_id=run_id,
                task_description=(
                    f"Generate the Bronze STTM from the profile for run_id='{run_id}'.\n\n"
                    f"Run ID: {run_id}\n"
                    f"Layer: Bronze\n"
                    f"Profile path: {profile_path}\n"
                    "Bronze is intent-agnostic. Inspect the profile context first, then "
                    "generate a complete Bronze STTM covering every column. "
                    "Add _load_timestamp and _source_file metadata rows. "
                    "Do NOT add a surrogate key — that belongs in Silver."
                ),
            )
        if not sttm_bronze_path:
            raise RuntimeError(
                "Bronze STTM generation produced no STTM file -- the LLM likely "
                "failed to return valid rows even after retrying."
            )

        state.update({
            "profile_path": profile_path,
            "sttm_bronze_path": sttm_bronze_path,
            "status": "awaiting_bronze_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase1_supervisor_completed",
            status="success", phase="phase1",
            profile_path=profile_path,
            sttm_bronze_path=sttm_bronze_path,
        )
    except Exception as e:
        state.update({
            "error": f"Phase 1 supervisor failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "phase1_supervisor_failed",
            status="failed", phase="phase1", detail=str(e),
        )

    return state


def run_bronze_sttm_regeneration(state: PipelineState, user_feedback: str = "") -> PipelineState:
    """Regenerate the Bronze STTM directly from the already-produced profile.

    UI contract: called by streamlit_app.py when the user wants the AI to redo
    the Bronze STTM (optionally steered by feedback) without leaving the Bronze
    STTM review screen. Status stays "awaiting_bronze_sttm_approval" -- this
    doesn't advance the phase, just refreshes sttm_bronze_path.
    """
    audit = AuditLogger(state["run_id"])
    state["error"] = ""

    feedback_note = (
        f"\nUser feedback on the previous Bronze STTM attempt to account for: {user_feedback.strip()}\n"
        if user_feedback.strip() else ""
    )

    try:
        audit.log(
            "orchestrator", "bronze_sttm_regeneration_started",
            status="in_progress", phase="bronze_sttm",
            rationale="User requested the Bronze STTM be regenerated.",
            user_feedback=user_feedback.strip(),
        )
        sttm_path = generate_bronze_sttm(
            profile_path=state["profile_path"],
            run_id=state["run_id"],
            task_description=(
                f"Regenerate the Bronze STTM from the profile for run_id='{state['run_id']}'.\n\n"
                f"Run ID: {state['run_id']}\n"
                f"Profile path: {state['profile_path']}\n"
                "Bronze is intent-agnostic. Inspect the profile context first, then "
                "generate a complete Bronze STTM covering every column. "
                "Add _load_timestamp and _source_file metadata rows. "
                "Do NOT add a surrogate key — that belongs in Silver."
                f"{feedback_note}"
            ),
        )
        if not sttm_path:
            raise RuntimeError(
                "Bronze STTM regeneration produced no STTM file -- the LLM likely "
                "failed to return valid rows even after retrying."
            )
        state.update({
            "sttm_bronze_path": sttm_path,
            "status": "awaiting_bronze_sttm_approval",
        })
        audit.log(
            "orchestrator", "bronze_sttm_regeneration_completed",
            status="success", phase="bronze_sttm",
            sttm_bronze_path=sttm_path,
        )
    except Exception as e:
        state.update({
            "error": f"Bronze STTM regeneration failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "bronze_sttm_regeneration_failed",
            status="failed", phase="bronze_sttm", detail=str(e),
        )

    return state


def run_bronze_execution(state: PipelineState) -> PipelineState:
    """Execute the approved Bronze STTM directly (single action -- no Supervisor step).

    UI contract: called by streamlit_app.py after Bronze STTM approval.
    Returns PipelineState with bronze_output_paths populated, awaiting output approval.
    """
    audit = AuditLogger(state["run_id"])
    state["bronze_sttm_approved"] = True
    state["error"] = ""

    try:
        audit.log(
            "orchestrator", "bronze_execution_started",
            status="in_progress", phase="bronze_execution",
            rationale="User approved Bronze STTM. Executing Bronze ingestion directly.",
        )
        output_paths = execute_bronze(
            input_files=state["uploaded_files"],
            sttm_path=state["sttm_bronze_path"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Input CSV files: {state['uploaded_files']}\n"
                f"Approved Bronze STTM: {state['sttm_bronze_path']}\n"
                "Inspect the files and STTM rules first. Plan which transformations "
                "apply to each file. Then execute ingestion across all input files."
            ),
        )
        if not output_paths:
            raise RuntimeError(
                "Bronze execution produced no output files -- the agent may not have "
                "actually run the ingestion tool. Try approving again."
            )
        state.update({
            "bronze_output_paths": output_paths,
            "status": "awaiting_bronze_output_approval",
        })
        audit.log(
            "orchestrator", "bronze_execution_completed",
            status="success", phase="bronze_execution",
            bronze_output_paths=output_paths,
        )
    except Exception as e:
        state.update({
            "error": f"Bronze execution failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "bronze_execution_failed",
            status="failed", phase="bronze_execution", detail=str(e),
        )

    return state


def run_silver_sttm_generation(state: PipelineState, user_feedback: str = "") -> PipelineState:
    """Generate the Silver STTM directly from the approved Bronze output.

    UI contract: called by streamlit_app.py after Bronze output approval.
    user_feedback: optional note from the user about what the Bronze output got
    wrong, folded into the generation goal so the Silver STTM can account for it.
    Returns PipelineState with sttm_silver_path populated, awaiting STTM approval.
    """
    audit = AuditLogger(state["run_id"])
    state["error"] = ""

    feedback_note = (
        f"\nUser feedback on the Bronze output to account for: {user_feedback.strip()}\n"
        if user_feedback.strip() else ""
    )

    try:
        audit.log(
            "orchestrator", "silver_sttm_started",
            status="in_progress", phase="silver_sttm",
            rationale="User approved Bronze output. Generating Silver STTM directly.",
            user_feedback=user_feedback.strip(),
        )
        sttm_path = generate_silver_sttm(
            bronze_output_paths=state["bronze_output_paths"],
            bronze_sttm_path=state["sttm_bronze_path"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Layer: Silver\n"
                f"Bronze output files: {state['bronze_output_paths']}\n"
                f"Approved Bronze STTM: {state['sttm_bronze_path']}\n"
                "Silver is intent-agnostic. Inspect the Bronze Parquet metadata first. "
                "Plan null handling, type casting, deduplication, and date standardisation "
                "for every column. Add surrogate key as the first row. Then generate the "
                "complete Silver STTM."
                f"{feedback_note}"
            ),
        )
        if not sttm_path:
            raise RuntimeError(
                "Silver STTM generation produced no STTM file -- the LLM likely "
                "failed to return valid rows even after retrying."
            )
        state.update({
            "sttm_silver_path": sttm_path,
            "status": "awaiting_silver_sttm_approval",
        })
        audit.log(
            "orchestrator", "silver_sttm_completed",
            status="success", phase="silver_sttm",
            sttm_silver_path=sttm_path,
        )
    except Exception as e:
        state.update({
            "error": f"Silver STTM generation failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "silver_sttm_failed",
            status="failed", phase="silver_sttm", detail=str(e),
        )

    return state


def run_silver_execution(state: PipelineState) -> PipelineState:
    """Execute the approved Silver STTM directly.

    UI contract: called by streamlit_app.py after Silver output approval.
    Returns PipelineState with silver_output_paths populated, awaiting output approval.
    """
    audit = AuditLogger(state["run_id"])
    state["silver_sttm_approved"] = True
    state["error"] = ""

    try:
        audit.log(
            "orchestrator", "silver_execution_started",
            status="in_progress", phase="silver_execution",
            rationale="User approved Silver STTM. Executing Silver cleansing directly.",
        )
        output_paths = execute_silver(
            input_files=state["bronze_output_paths"],
            sttm_path=state["sttm_silver_path"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Input Bronze files: {state['bronze_output_paths']}\n"
                f"Approved Silver STTM: {state['sttm_silver_path']}\n"
                "Inspect the Bronze Parquet schemas and STTM rules first. Plan the "
                "cleansing approach for each column and file. Then execute cleansing "
                "across all Bronze inputs, producing Silver Parquet outputs."
            ),
        )
        if not output_paths:
            raise RuntimeError(
                "Silver execution produced no output files -- the agent may not have "
                "actually run the cleansing tool. Try approving again."
            )
        state.update({
            "silver_output_paths": output_paths,
            "status": "awaiting_silver_output_approval",
        })
        audit.log(
            "orchestrator", "silver_execution_completed",
            status="success", phase="silver_execution",
            silver_output_paths=output_paths,
        )
    except Exception as e:
        state.update({
            "error": f"Silver execution failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "silver_execution_failed",
            status="failed", phase="silver_execution", detail=str(e),
        )

    return state


def run_gold_sttm_generation(state: PipelineState, user_feedback: str = "") -> PipelineState:
    """Generate the Gold STTM directly from the approved Silver output.

    UI contract: called by streamlit_app.py after Silver output approval.
    user_feedback: optional note from the user about what the Silver output got
    wrong, folded into the generation goal so the Gold STTM can account for it.
    Returns PipelineState with sttm_gold_path populated, awaiting STTM approval.
    """
    audit = AuditLogger(state["run_id"])
    state["error"] = ""

    feedback_note = (
        f"\nUser feedback on the Silver output to account for: {user_feedback.strip()}\n"
        if user_feedback.strip() else ""
    )

    try:
        audit.log(
            "orchestrator", "gold_sttm_started",
            status="in_progress", phase="gold_sttm",
            rationale="User approved Silver output. Generating Gold STTM directly.",
            user_feedback=user_feedback.strip(),
        )
        sttm_path = generate_gold_sttm(
            silver_output_paths=state["silver_output_paths"],
            silver_sttm_path=state["sttm_silver_path"],
            business_intent=state["business_intent"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Layer: Gold\n"
                f"Business intent: {state['business_intent']}\n"
                f"Silver output files: {state['silver_output_paths']}\n"
                f"Approved Silver STTM: {state['sttm_silver_path']}\n"
                "Inspect the Silver Parquet metadata first. Plan join keys, column "
                "renames, and aggregation rules. Build queryable analytics-ready tables "
                "-- do NOT pre-aggregate for the business question. Add surrogate key "
                "as the first row. Then generate the complete Gold STTM."
                f"{feedback_note}"
            ),
        )
        if not sttm_path:
            raise RuntimeError(
                "Gold STTM generation produced no STTM file -- the LLM likely "
                "failed to return valid rows even after retrying."
            )
        state.update({
            "sttm_gold_path": sttm_path,
            "status": "awaiting_gold_sttm_approval",
        })
        audit.log(
            "orchestrator", "gold_sttm_completed",
            status="success", phase="gold_sttm",
            sttm_gold_path=sttm_path,
        )
    except Exception as e:
        state.update({
            "error": f"Gold STTM generation failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "gold_sttm_failed",
            status="failed", phase="gold_sttm", detail=str(e),
        )

    return state


def run_gold_execution(state: PipelineState) -> PipelineState:
    """Execute the approved Gold STTM directly.

    UI contract: called by streamlit_app.py after Gold STTM approval.
    Returns PipelineState with gold_output_paths populated, awaiting output approval.
    """
    audit = AuditLogger(state["run_id"])
    state["gold_sttm_approved"] = True
    state["error"] = ""

    try:
        audit.log(
            "orchestrator", "gold_execution_started",
            status="in_progress", phase="gold_execution",
            rationale="User approved Gold STTM. Executing Gold materialisation directly.",
        )
        output_paths = execute_gold(
            input_files=state["silver_output_paths"],
            sttm_path=state["sttm_gold_path"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Input Silver files: {state['silver_output_paths']}\n"
                f"Approved Gold STTM: {state['sttm_gold_path']}\n"
                "Inspect the Silver Parquet schemas and Gold STTM rules first, grouped "
                "by target table. Plan joins, renames, and aggregations per Gold table. "
                "Then materialise all Gold target tables from the Silver inputs."
            ),
        )
        if not output_paths:
            raise RuntimeError(
                "Gold execution produced no output files -- the agent may not have "
                "actually run the materialisation tool. Try approving again."
            )
        state.update({
            "gold_output_paths": output_paths,
            "status": "awaiting_gold_output_approval",
        })
        audit.log(
            "orchestrator", "gold_execution_completed",
            status="success", phase="gold_execution",
            gold_output_paths=output_paths,
        )
    except Exception as e:
        state.update({
            "error": f"Gold execution failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "gold_execution_failed",
            status="failed", phase="gold_execution", detail=str(e),
        )

    return state


def run_report_generation(state: PipelineState, user_feedback: str = "") -> PipelineState:
    """Generate the executive report directly from the approved Gold output.

    UI contract: called by streamlit_app.py after Gold output approval.
    user_feedback: optional note from the user about what the Gold output got
    wrong, folded into the generation goal so the report can account for it.
    Returns PipelineState with report_path populated; pipeline complete.
    """
    audit = AuditLogger(state["run_id"])
    state["error"] = ""

    feedback_note = (
        f"\nUser feedback on the Gold output to account for: {user_feedback.strip()}\n"
        if user_feedback.strip() else ""
    )

    try:
        audit.log(
            "orchestrator", "report_generation_started",
            status="in_progress", phase="report_generation",
            rationale="User approved Gold output. Generating executive report directly.",
            user_feedback=user_feedback.strip(),
        )
        report_path = generate_report(
            gold_files=state["gold_output_paths"],
            business_intent=state["business_intent"],
            run_id=state["run_id"],
            task_description=(
                f"Run ID: {state['run_id']}\n"
                f"Business question: {state['business_intent']}\n"
                f"Gold files: {state['gold_output_paths']}\n"
                "Inspect the Gold tables first to understand their structure. Plan your "
                "SQL approach to directly answer the business question. Load the tables, "
                "execute your query, analyse results, and produce a structured HTML report "
                "with charts that provide visual evidence for your answer."
                f"{feedback_note}"
            ),
        )
        if not report_path:
            raise RuntimeError("Report generation produced no report file.")
        state.update({
            "report_path": report_path,
            "status": "completed",
        })
        audit.log(
            "orchestrator", "report_generation_completed",
            status="success", phase="report_generation",
            report_path=report_path,
        )
    except Exception as e:
        state.update({
            "error": f"Report generation failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "report_generation_failed",
            status="failed", phase="report_generation", detail=str(e),
        )

    return state
