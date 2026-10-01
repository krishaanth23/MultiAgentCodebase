"""Gold layer AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, uses inspect_task_tool to
preview Silver Parquet schemas and Gold STTM rules, forms a materialisation
plan, then executes via gold_ingestion_tool.

I/O contract:
    execute_gold(input_files, sttm_path, run_id, task_description) -> list[str]
"""

import os
import re
import json
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import GOLD_DIR, LLM_PROVIDER
from core.llm import make_llm, invoke_agent_with_tool_recovery
from core.audit import AuditLogger
from agents.sttm_generator import GOLD_LOGIC_TAGS, parse_logic_tags
from core.observability import AgentTrace


GOLD_AGENT_PROMPT = """You are an autonomous Analytics Engineer specialising in the Gold layer of a
Medallion data pipeline. You operate independently: you receive a goal from the
orchestrator, inspect the Silver Parquet inputs and Gold STTM rules, form a
concrete materialisation plan, execute it, and verify your output.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the Silver Parquet input files, the STTM path,
   the business intent, and what Gold target tables need to be materialised.

2. INSPECT: Call inspect_task_tool FIRST. This previews each Silver Parquet
   file's schema and lists all Gold STTM materialisation rules grouped by target table.
   State your observations: which source tables need to be joined, which columns
   will be renamed or aggregated, and how many Gold target tables will be produced.

3. PLAN: Based on the inspection output, write your explicit materialisation plan:
   - Which Gold target tables will be created?
   - Which Silver source tables feed each Gold table?
   - What joins, renames, and aggregations will be applied?
   - What surrogate key will be injected?

4. ACT: Call gold_ingestion_tool to execute the full materialisation workflow
   across all Silver inputs using the approved STTM rules.

5. VERIFY: Confirm the list of Gold Parquet output paths returned and report
   what was materialised (table count, rows per table, joins performed).

## Available tools

- **inspect_task_tool**: Previews each Silver Parquet input file (schema, column
  names, dtypes, row count, sample values) and lists all Gold STTM materialisation
  rules grouped by target table. Call this FIRST to form your plan. Returns a JSON summary.

- **gold_ingestion_tool**: Executes the full Gold layer materialisation workflow.
  Loads Silver Parquet files as source tables, groups STTM rules by target table,
  applies joins/merges across sources, executes renames and aggregations, injects
  surrogate keys, and writes Gold Parquet artifacts.
  Returns a JSON list of output file paths.

## Output
After execution, report: (1) your materialisation plan, (2) which joins and
transformations were applied per Gold table, (3) the list of Gold Parquet output paths."""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_task(input_files: list[str], sttm_path: str) -> dict:
    """Preview Silver Parquet schemas and Gold STTM materialisation rules. No LLM."""
    sttm_df = pd.read_csv(sttm_path).fillna("")
    file_summaries = []
    for fp in input_files:
        try:
            df = pd.read_parquet(fp)
            col_info = {}
            for col in df.columns:
                col_info[col] = {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(3).tolist(),
                }
            file_summaries.append({
                "file": os.path.basename(fp),
                "rows": df.shape[0],
                "columns": list(df.columns),
                "column_info": col_info,
            })
        except Exception as e:
            file_summaries.append({"file": os.path.basename(fp), "error": str(e)})

    # Group rules by target table so the agent can see what each Gold table needs
    rules_by_target: dict = {}
    for _, row in sttm_df.iterrows():
        target = str(row.get("target_table", "unknown")).strip()
        if target not in rules_by_target:
            rules_by_target[target] = []
        rules_by_target[target].append({
            "source_table": str(row.get("source_table", "")),
            "source_column": str(row.get("source_column", "")),
            "target_column": str(row.get("target_column", "")),
            "transformation_type": str(row.get("transformation_type", "")),
            "transformation_logic": str(row.get("transformation_logic", "")),
        })

    return {
        "silver_files": file_summaries,
        "gold_target_tables": list(rules_by_target.keys()),
        "rules_by_target_table": rules_by_target,
        "total_files": len(input_files),
        "total_target_tables": len(rules_by_target),
        "total_rules": len(sttm_df),
    }


_CASE_WHEN_RE = re.compile(
    r"when\s+[\w.]+\s*(?P<op>=|==|!=|<>|>=|<=|>|<)\s*'?(?P<val>[^'\s]+?)'?\s+then\s+'?(?P<then>[^'\s]+?)'?"
    r"(?=\s+when\b|\s+else\b|\s+end\b|$)",
    re.IGNORECASE,
)
_CASE_ELSE_RE = re.compile(r"else\s+'?(?P<else>[^'\s]+?)'?\s*end\b", re.IGNORECASE)


def _coerce_case_literal(raw: str):
    """Parse a CASE WHEN literal as int/float when possible, else keep it as a string."""
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


def _eval_case_when(series: pd.Series, logic: str):
    """Evaluate a 'CASE WHEN <col> <op> <val> THEN <val> [WHEN ...] ELSE <val> END' expression.

    Only the comparison operator/value/result are parsed from each WHEN clause — the
    column name itself is assumed to be the rule's own source column (`series`).
    Returns a new Series, or None if `logic` isn't a CASE WHEN expression at all.
    """
    if "case" not in logic.lower() or "when" not in logic.lower():
        return None
    whens = list(_CASE_WHEN_RE.finditer(logic))
    if not whens:
        return None

    result = pd.Series([None] * len(series), index=series.index, dtype=object)
    for m in reversed(whens):  # reversed so the first-listed WHEN wins on overlap, per SQL semantics
        op = m.group("op")
        val = _coerce_case_literal(m.group("val"))
        then_val = _coerce_case_literal(m.group("then"))
        if op in ("=", "=="):
            mask = series.astype(str) == str(val)
        elif op in ("!=", "<>"):
            mask = series.astype(str) != str(val)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = {">": numeric > val, "<": numeric < val, ">=": numeric >= val, "<=": numeric <= val}[op]
        result[mask] = then_val

    else_match = _CASE_ELSE_RE.search(logic)
    if else_match:
        result[result.isna()] = _coerce_case_literal(else_match.group("else"))
    return result


def _apply_gold_rules(
    input_files: list[str], sttm_path: str, run_id: str
) -> list[str]:
    """Load Silver tables, group STTM by target_table, apply joins/renames/aggs, write Gold Parquet."""
    audit = AuditLogger(run_id)
    audit.log("gold_agent", "started", input_files=input_files, sttm_path=sttm_path)

    sttm_df = pd.read_csv(sttm_path).fillna("")

    # Load all Silver files into a dict keyed by source table name
    source_dataframes = {}
    for file_path in input_files:
        source_name = os.path.basename(file_path).replace("_silver.parquet", "")
        source_dataframes[source_name] = pd.read_parquet(file_path)

    print(f"[GOLD] Loaded {len(source_dataframes)} Silver source tables: {list(source_dataframes.keys())}")

    target_tables = sttm_df.groupby("target_table")
    print(f"[GOLD] Creating {len(target_tables)} Gold target tables: {list(target_tables.groups.keys())}")

    output_paths = []

    for target_table_name, table_rules in target_tables:
        target_table_name = str(target_table_name).strip()
        if not target_table_name:
            continue

        print(f"[GOLD] Processing target table: {target_table_name} ({len(table_rules)} rules)")

        source_tables_needed = table_rules["source_table"].unique()
        source_tables_needed = [
            str(s).strip().replace("_silver.parquet", "").replace("_silver", "")
            for s in source_tables_needed if str(s).strip()
        ]
        available_sources = {
            src: source_dataframes[src]
            for src in source_tables_needed
            if src in source_dataframes
        }

        if not available_sources:
            print(f"[GOLD] No matching source data for {target_table_name}, skipping")
            continue

        df = list(available_sources.values())[0].copy()

        if len(available_sources) > 1:
            for source_name, source_df in list(available_sources.items())[1:]:
                common_cols = [col for col in df.columns if col in source_df.columns]
                metadata_cols = {"_load_timestamp", "_source_file", "_row_inserted", "_row_updated"}
                id_cols = [
                    col for col in common_cols
                    if (col.endswith("_id") or col == "id") and col not in metadata_cols
                ]
                if id_cols:
                    df = df.merge(source_df, on=id_cols, how="outer", suffixes=("", "_dup"))
                    dup_cols = [col for col in df.columns if col.endswith("_dup")]
                    if dup_cols:
                        df = df.drop(columns=dup_cols)
                else:
                    df = pd.concat([df, source_df], ignore_index=True, sort=False)

        # Apply transformations. Each STTM row builds its own target_col independently
        # (rather than a single shared rename dict) so one source column can fan out
        # into several target columns — e.g. a passthrough copy plus a CASE-derived one.
        group_by_cols: list[str] = []
        agg_map: dict[str, str] = {}

        for _, rule in table_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            raw_logic = str(rule.get("transformation_logic", ""))
            logic = raw_logic.lower()
            transformation_type = str(rule.get("transformation_type", "")).lower()

            if not (source_col and target_col and source_col in df.columns):
                continue

            case_result = _eval_case_when(df[source_col], raw_logic)
            if case_result is not None:
                df[target_col] = case_result
            elif target_col != source_col:
                df[target_col] = df[source_col]
            # else: passthrough with no rename — target_col already exists as-is.

            if transformation_type == "direct":
                if target_col not in group_by_cols:
                    group_by_cols.append(target_col)

            # A CASE WHEN expression is its own row type, already handled above --
            # don't also tag-match it for aggregation (its branch literals could
            # coincidentally contain a word like "sum" without meaning to aggregate).
            if case_result is None:
                tags = parse_logic_tags(logic)
                if tags and not (tags & GOLD_LOGIC_TAGS):
                    print(f"[GOLD] Unrecognised transformation_logic {logic!r} for column "
                          f"'{target_col}' -- treating as passthrough.")

                if "sum" in tags:
                    agg_map[target_col] = "sum"
                elif "average" in tags:
                    agg_map[target_col] = "mean"
                elif "count" in tags:
                    agg_map[target_col] = "count"
                elif "max" in tags:
                    agg_map[target_col] = "max"
                elif "min" in tags:
                    agg_map[target_col] = "min"

        valid_group_by = [col for col in group_by_cols if col in df.columns]
        valid_agg_map = {col: func for col, func in agg_map.items() if col in df.columns}
        if valid_group_by and valid_agg_map:
            agg_only = {col: func for col, func in valid_agg_map.items() if col not in valid_group_by}
            if agg_only:
                df = df.groupby(valid_group_by, dropna=False, as_index=False).agg(agg_only)

        target_columns = set(table_rules["target_column"].unique())
        columns_to_keep = [
            c for c in df.columns
            if c in target_columns or c.startswith("_") or c.startswith("pk_")
        ]
        if columns_to_keep:
            df = df[columns_to_keep]

        pk_col = "pk_gold_id"
        if pk_col not in df.columns:
            df.insert(0, pk_col, range(1, len(df) + 1))

        output_filename = f"{target_table_name}.parquet"
        output_path = str(GOLD_DIR / output_filename)
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        print(f"[GOLD] Created {target_table_name}: {df.shape[0]} rows x {df.shape[1]} columns")
        audit.log(
            "gold_agent", "table_created",
            target_table=target_table_name,
            output_file=output_path,
            shape=list(df.shape),
        )

    audit.log(
        "gold_agent", "completed",
        output_files=output_paths,
        table_count=len(output_paths),
    )
    return output_paths


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_gold_tools(
    input_files: list[str], sttm_path: str, run_id: str
):
    """Returns inspect + ingestion tools bound to this run's parameters via closure."""

    @tool
    def inspect_task_tool(confirmation: str = "execute") -> str:
        """Preview Silver Parquet input schemas and Gold STTM materialisation rules.

        Returns a JSON summary of each Silver file's column names, dtypes, null counts,
        and sample values, plus all STTM rules grouped by Gold target table showing
        which sources feed each table and what transformations are required.
        Call this FIRST to understand what needs to be materialised and form your plan.
        """
        return json.dumps(_inspect_task(input_files, sttm_path), default=str)

    @tool
    def gold_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute the full Gold layer materialisation using the approved STTM rules.

        Loads Silver Parquet files as source tables, groups STTM rules by target table,
        applies joins/merges across sources, executes renames and aggregations, injects
        surrogate keys (pk_gold_id), and writes Gold Parquet artifacts.
        Returns a JSON list of output file paths.
        """
        output_paths = _apply_gold_rules(input_files, sttm_path, run_id)
        return json.dumps(output_paths)

    return inspect_task_tool, gold_ingestion_tool


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def execute_gold(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Gold AI agent entry point — autonomous ReAct version.

    The agent inspects Silver Parquet schemas and Gold STTM rules, forms an
    explicit materialisation plan, then executes across all input files.
    Business intent is already baked into the approved Gold STTM, so this
    executor is intent-agnostic.

    Args:
        input_files: Silver Parquet file paths to materialise.
        sttm_path: Path to the approved Gold STTM CSV.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        list[str]: Gold Parquet output file paths.
    """
    trace = AgentTrace("gold_agent", run_id)
    trace.set_input(input_files=input_files, sttm_path=sttm_path)

    inspect_tool, ingestion_tool = _make_gold_tools(input_files, sttm_path, run_id)
    llm = make_llm()

    print(f"[GOLD] Running autonomous ReAct agent ({LLM_PROVIDER})")
    agent = create_agent(llm, [inspect_tool, ingestion_tool], system_prompt=GOLD_AGENT_PROMPT)

    try:
        result = invoke_agent_with_tool_recovery(
            agent, {"messages": [HumanMessage(content=task_description)]}, [inspect_tool, ingestion_tool]
        )
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    output_paths: list[str] = []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                    output_paths = parsed
                    break
            except (json.JSONDecodeError, ValueError):
                continue

    trace.set_output(output_paths=output_paths).complete()
    return output_paths
