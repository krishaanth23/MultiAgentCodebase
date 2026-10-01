# 🏅 Agentic Medallion Pipeline

> **Intent-Driven Agentic Data Engineering for Retail Sales Analytics**
>
> Upload messy raw CSV files, ask a business question in plain English, and receive a structured executive report — powered by autonomous AI agents working through a Bronze → Silver → Gold Medallion architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Key Design Patterns](#key-design-patterns)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Observability](#observability)
- [Agent Handoffs](#agent-handoffs)
- [FAQ](#faq)

---

## Overview

Traditional data engineering requires a data engineer to manually inspect raw files, write transformation rules, apply cleansing logic, build aggregation queries, and produce reports. This pipeline automates that entire workflow using autonomous AI agents.

**What you do:**
1. Upload raw CSV files via the Streamlit UI
2. Type a business question (e.g. *"Which product category had the highest sales in Q4?"*)
3. Review and approve AI-generated transformation rules at six human-in-the-loop gates
4. Receive a complete HTML executive report with SQL-backed charts

**What the agents do:**
- Profile your data and understand its structure
- Generate Source-to-Target Mapping (STTM) rules for each pipeline layer
- Execute Bronze ingestion, Silver cleansing, and Gold materialisation
- Write and run SQL to answer your business question
- Produce an interactive HTML report with Plotly charts

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────────────────────────────┐     ┌──────────────┐
│             │     │                  │     │                  AGENTS                   │     │              │
│  Streamlit  │────▶│   Supervisor     │────▶│  Profiler → STTM → Bronze → STTM →       │────▶│   Storage    │
│     UI      │     │  Orchestrator    │     │  Silver → STTM → Gold → Reporter          │     │   & Output   │
│             │◀────│                  │◀────│                                           │◀────│              │
└─────────────┘     └──────────────────┘     └───────────────────────────────────────────┘     └──────────────┘
       │                                                        │
       │         ⏸ Human approves at 6 checkpoints             │
       └────────────────────────────────────────────────────────┘
```

### The Seven Phases

| Phase | Orchestrator Function | Agents | Output | Gate |
|-------|-----------------------|--------|--------|------|
| **Phase 1** | `run_until_bronze_sttm` | Profiler → STTM | `profile.json` + `sttm_bronze.csv` | ⏸ Human approves Bronze STTM |
| **Phase 2** | `run_bronze_execution` | Bronze | `*_bronze.parquet` | ⏸ Human reviews Bronze output |
| **Phase 3** | `run_silver_sttm_generation` | STTM | `sttm_silver.csv` | ⏸ Human approves Silver STTM |
| **Phase 4** | `run_silver_execution` | Silver | `*_silver.parquet` | ⏸ Human reviews Silver output |
| **Phase 5** | `run_gold_sttm_generation` | STTM | `sttm_gold.csv` | ⏸ Human approves Gold STTM |
| **Phase 6** | `run_gold_execution` | Gold | `*_gold.parquet` | ⏸ Human reviews Gold output |
| **Phase 7** | `run_report_generation` | Reporter | `report.html` + `report.json` | ✅ Pipeline complete |

---

## How It Works

### Medallion Layers

```
Raw CSV  ──▶  Bronze  ──▶  Silver  ──▶  Gold  ──▶  Report
             (ingest)    (cleanse)  (aggregate)   (answer)
```

| Layer | What happens |
|-------|-------------|
| **Bronze** | CSV → Parquet. Columns renamed, types cast, `_load_timestamp` and `_source_file` metadata injected. No null handling — faithful raw copy. |
| **Silver** | Bronze Parquet → cleansed Parquet. Nulls handled (drop/fill), deduplication, type standardisation, date formatting (→ YYYY-MM-DD), surrogate key `pk_*_silver_id` injected as first column. |
| **Gold** | Silver Parquet → analytics-ready Parquet. Multi-source joins, aggregations (sum/avg/count/max/min), `pk_gold_id` surrogate key. One file per Gold target table. |
| **Report** | Gold Parquet → HTML. DuckDB runs agent-written SQL, Plotly renders charts, structured executive summary answers the business question. |

### What Makes This Agentic

Every specialist agent follows the same autonomous pattern:

```
THINK → INSPECT → PLAN → ACT → VERIFY
```

1. **THINK** — reads the goal from the Supervisor and understands the context
2. **INSPECT** — calls its inspect tool to preview inputs before touching any data
3. **PLAN** — states an explicit plan based on what it observed
4. **ACT** — calls its execution tool to process data
5. **VERIFY** — confirms outputs and reports back to the Supervisor

The LLM decides **when** to call tools. Python executes the actual data work. This separation gives you AI intelligence with deterministic, reliable data processing.

---

## Project Structure

```
MultiAgentCodebase/
│
├── app/
│   └── streamlit_app.py          # Streamlit UI — entry point for users
│
├── agents/
│   ├── orchestrator.py           # Supervisor agent + phase management
│   ├── profiler.py               # Data Profiler agent
│   ├── sttm_generator.py         # Unified STTM agent (Bronze/Silver/Gold)
│   ├── bronze_agent.py           # Bronze layer ingestion agent
│   ├── silver_agent.py           # Silver layer cleansing agent
│   ├── gold_agent.py             # Gold layer materialisation agent
│   └── reporter.py               # Reporter agent (SQL + HTML report)
│
├── core/
│   ├── config.py                 # Paths, API keys, LLM provider config
│   ├── llm.py                    # LLM factory, rate limiter, tool-call corruption recovery
│   ├── audit.py                  # AuditLogger — append-only JSONL event log
│   ├── observability.py          # AgentTrace — full reasoning trace per agent
│   └── memory.py                 # ChromaDB document store for agent context
│
├── data/
│   ├── landing/                  # Uploaded CSV files
│   ├── profiles/                 # Data profile JSON files
│   ├── sttm/                     # Generated STTM CSV files
│   ├── bronze_layer/             # Bronze Parquet files
│   ├── silver_layer/             # Silver Parquet files
│   ├── gold_layer/               # Gold Parquet files
│   └── traces/                   # Agent observability trace files
│
├── audit_logs/                   # Per-run JSONL audit event logs
├── reports/                      # HTML + JSON report outputs
│
├── tests/
│   ├── test_bronze.py
│   ├── test_silver.py
│   ├── test_gold.py
│   ├── test_sttm.py
│   ├── test_profiler.py
│   ├── test_reporter.py
│   └── test_audit.py
│
├── test_import.py                # Quick import smoke test
├── .env                          # API keys (not committed to git)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

All `data/`, `audit_logs/`, `reports/`, and `.chroma/` directories are created automatically on import of `core/config.py` — no manual setup needed.

---

## Agents

### Supervisor Orchestrator — `agents/orchestrator.py`

The pipeline coordinator. Phase 1 uses an autonomous LangChain ReAct agent to sequence Profiler → STTM. Later phases call specialist agents directly since the UI gates have already determined the single action to take.

**Entry points (called by Streamlit UI):**
```python
run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState   # Phase 1
run_bronze_execution(state) -> PipelineState                               # Phase 2
run_silver_sttm_generation(state) -> PipelineState                         # Phase 3
run_silver_execution(state) -> PipelineState                               # Phase 4
run_gold_sttm_generation(state) -> PipelineState                           # Phase 5
run_gold_execution(state) -> PipelineState                                 # Phase 6
run_report_generation(state) -> PipelineState                              # Phase 7
run_bronze_sttm_regeneration(state, user_feedback) -> PipelineState        # Utility: re-generate Bronze STTM with user feedback
```

**Key patterns used:** `PipelineState` (TypedDict), tool factories with closures, scratchpad dict for intra-phase handoffs.

---

### Profiler Agent — `agents/profiler.py`

Profiles raw CSV files to understand structure, semantics, and quality.

**Tools:**
- `inspect_files_tool` — lightweight preview (shape, column names, dtypes, 3 sample values)
- `profiler_tool` — full statistics (null count/%, unique count, min/max/mean)

**Returns:** Path to `data/profiles/profile_combined_*.json`

**Output contains:** Raw statistics per column, LLM-generated semantic meanings, discovered join key candidates, data quality notes.

**Entry point:**
```python
profile_multiple_datasets(file_paths, run_id, task_description) -> str
```

---

### STTM Agent — `agents/sttm_generator.py`

Unified agent that generates Source-to-Target Mapping rules for all three layers. The orchestrator specifies which layer to generate; the agent inspects context and picks the correct tool.

**Tools:**
- `inspect_context_tool` — previews source data for the requested layer
- `generate_bronze_sttm_tool` — ingestion rules (rename, type cast, metadata injection)
- `generate_silver_sttm_tool` — cleansing rules (nulls, dedup, types, surrogate key)
- `generate_gold_sttm_tool` — materialisation rules (joins, aggregations, surrogate key)

**Entry points:**
```python
generate_bronze_sttm(profile_path, business_intent, run_id, task_description) -> str
generate_silver_sttm(bronze_output_paths, bronze_sttm_path, business_intent, run_id, task_description) -> str
generate_gold_sttm(silver_output_paths, silver_sttm_path, business_intent, run_id, task_description) -> str
```

**STTM CSV columns:** `source_schema`, `source_table`, `source_column`, `target_schema`, `target_table`, `target_column`, `transformation_type`, `transformation_logic`

> Note: `business_intent` is only consumed during Gold STTM generation. Bronze and Silver STTMs are intent-agnostic structural transformations.

---

### Bronze Agent — `agents/bronze_agent.py`

Ingests raw CSV files into the Bronze Parquet layer using approved STTM rules.

**Tools:**
- `inspect_task_tool` — previews CSV file shapes and STTM transformation rules
- `bronze_ingestion_tool` — applies rules and writes Parquet files

**Entry point:**
```python
execute_bronze(input_files, sttm_path, run_id, task_description) -> list[str]
```

**What it does:** Column renaming per STTM rules, type casting (`to_numeric`, `to_datetime`), injects `_load_timestamp` (UTC ISO string) and `_source_file` (source path) into every output file.

---

### Silver Agent — `agents/silver_agent.py`

Cleanses Bronze Parquet files into trusted Silver Parquet files.

**Tools:**
- `inspect_task_tool` — previews Bronze Parquet schemas, null counts, and STTM cleansing rules
- `silver_ingestion_tool` — applies cleansing and writes Silver Parquet files

**Entry point:**
```python
execute_silver(input_files, sttm_path, run_id, task_description) -> list[str]
```

**What it does:** Null handling (`dropna`, `fillna` with mean/median/mode/constant), deduplication (`drop_duplicates`), type casting, date standardisation (→ YYYY-MM-DD), text normalisation (strip/lower/upper), surrogate key injection (`pk_*_silver_id` as first column), column filtering to STTM-approved columns only.

---

### Gold Agent — `agents/gold_agent.py`

Materialises analytics-ready Gold Parquet tables from Silver inputs.

**Tools:**
- `inspect_task_tool` — previews Silver schemas and STTM rules grouped by Gold target table
- `gold_ingestion_tool` — applies joins, aggregations, and writes Gold Parquet files

**Entry point:**
```python
execute_gold(input_files, sttm_path, business_intent, run_id, task_description) -> list[str]
```

**What it does:** Multi-source outer joins on matching `_id` columns, column renaming, `groupby().agg()` for sum/avg/count/max/min aggregations, surrogate key injection (`pk_gold_id` as first column). Produces one Parquet file per Gold target table defined in the STTM.

---

### Reporter Agent — `agents/reporter.py`

Answers the business question by querying Gold tables with agent-written SQL and producing an HTML executive report.

**Tools:**
- `inspect_gold_tables_tool` — previews Gold table schemas and sample rows (no DuckDB)
- `load_gold_data_tool` — registers Gold Parquet files as DuckDB in-memory tables
- `execute_query_tool(sql_query)` — runs agent-written SQL, returns JSON rows

**Entry point:**
```python
generate_report(gold_files, business_intent, run_id, task_description) -> str
```

**What it produces:**
- `reports/report_{run_id[:8]}.html` — self-contained HTML with embedded Plotly charts
- `reports/report_{run_id[:8]}.json` — structured analysis (direct_answer, charts spec, detailed_analysis)

---

## Key Design Patterns

### Tool Factory + Closure

Every agent uses a factory function that captures runtime parameters (file paths, run IDs) via Python closure. This means the LLM never needs to reproduce exact file paths as arguments — eliminating hallucination risk.

```python
def _make_bronze_tools(input_files, sttm_path, run_id):
    # input_files, sttm_path, run_id are "closed over" below

    @tool
    def bronze_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute Bronze ingestion."""
        # input_files and sttm_path come from the closure above
        # The LLM calls this with no file path arguments
        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        return json.dumps(output_paths)

    return bronze_ingestion_tool
```

### Scratchpad Pattern

Within a phase, tools share data via a plain Python dictionary (scratchpad) captured by closure. Tool 1 writes the output path, Tool 2 reads it — no LLM copy-pasting required.

```python
scratchpad = {}   # plain dict — no library needed

@tool
def profiler_agent_tool(goal: str) -> str:
    profile_path = profile_multiple_datasets(...)
    scratchpad["profile_path"] = profile_path   # Tool 1 writes
    return json.dumps({"profile_path": profile_path})

@tool
def sttm_agent_tool(goal: str) -> str:
    sttm_path = generate_bronze_sttm(
        profile_path=scratchpad["profile_path"],  # Tool 2 reads
        ...
    )
```

### Human-in-the-Loop Gates

Six approval gates pause the pipeline at each key decision point. Three gates guard STTM generation (the transformation recipe); three guard the data outputs before the next layer processes them.

```
Phase 1 runs → ⏸ Human approves Bronze STTM
Phase 2 runs → ⏸ Human reviews Bronze output
Phase 3 runs → ⏸ Human approves Silver STTM
Phase 4 runs → ⏸ Human reviews Silver output
Phase 5 runs → ⏸ Human approves Gold STTM
Phase 6 runs → ⏸ Human reviews Gold output → Phase 7 runs
```

Users can also provide free-text feedback at the Bronze STTM gate to steer regeneration via `run_bronze_sttm_regeneration`.

### ReAct Agent Loop

Every agent follows Think → Inspect → Plan → Act → Verify. The inspect tool runs first (lightweight, no side effects), the agent forms a plan, then the execution tool runs. The LLM never executes blindly.

### Separation of Intelligence and Execution

The LLM decides **when** to call tools and **what goal to pursue**. Python tools do the actual data work (pandas, DuckDB, Plotly). This gives AI-level adaptability with deterministic, reliable data processing.

---

## Tech Stack

| Category | Library | Used for |
|----------|---------|----------|
| **Orchestration** | `langchain ≥0.2`, `langgraph ≥0.2` | `create_agent`, `@tool`, ReAct loop, message types |
| **LLM (Groq)** | `langchain-groq`, `groq` | `ChatGroq` — configurable via `GROQ_MODEL` env var |
| **UI** | `streamlit ≥1.38` | File upload, STTM approval, report rendering |
| **Data** | `pandas ≥2.2` | CSV/Parquet read-write, all transformations |
| **Storage** | `pyarrow ≥17` | Parquet file backend |
| **Analytics** | `duckdb ≥0.9` | In-memory SQL engine for Reporter |
| **Charts** | `plotly ≥5.17` | Interactive bar, line, pie, scatter charts |
| **Vector Store** | `chromadb ≥0.5` | Persistent document store for agent memory |
| **Validation** | `pydantic ≥2.0` | Data models and structured outputs |
| **Config** | `python-dotenv` | API key loading from `.env` |
| **Testing** | `pytest ≥8.0` | Unit tests for all agent modules |
| **Utilities** | `uuid`, `json`, `pathlib`, `datetime` | Standard library |

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- A Groq API key

### 1. Clone the repository

```bash
git clone https://github.com/your-org/medallion-pipeline.git
cd medallion-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b       # optional — this is the default
LLM_REQUESTS_PER_MINUTE=20           # optional — rate limit shared across all agents
```

Get your Groq API key at [console.groq.com](https://console.groq.com)

All data directories (`data/`, `audit_logs/`, `reports/`, `.chroma/`) are created automatically when `core/config.py` is first imported — no manual `mkdir` needed.

---

## Running the Pipeline

### Start the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`

### Step-by-step usage

1. **Upload CSV files** — drag and drop your raw data files
2. **Enter your business question** — plain English, e.g. *"Which store had the highest revenue last month?"*
3. **Click Run Phase 1** — agents profile data and generate Bronze STTM
4. **Review Bronze STTM** — check transformation rules; optionally enter feedback and regenerate, then click Approve
5. **Click Run Phase 2** — Bronze agent ingests data to Parquet
6. **Review Bronze output** — verify ingested file shapes and column names, then click Approve
7. **Click Run Phase 3** — Silver STTM generated from Bronze output
8. **Review Silver STTM** — check cleansing rules, click Approve
9. **Click Run Phase 4** — Silver agent cleanses data
10. **Review Silver output** — verify null handling and surrogate keys, then click Approve
11. **Click Run Phase 5** — Gold STTM generated (business intent applied here)
12. **Review Gold STTM** — check join logic and aggregations, click Approve
13. **Click Run Phase 6** — Gold agent materialises analytics tables
14. **Review Gold output** — verify aggregation results, then click Approve
15. **Click Run Phase 7** — Reporter queries Gold tables and generates HTML report
16. **View report** — executive summary with Plotly charts answers your business question

### Sample input CSV format

```csv
sale_date,product_id,product_name,category,store_id,quantity,unit_price
2024-01-15,P001,Laptop Pro,Electronics,S001,2,1299.99
2024-01-15,P002,Office Chair,Furniture,S002,5,249.99
2024-01-16,P001,Laptop Pro,Electronics,S001,1,1299.99
```

### Running tests

```bash
pytest tests/
```

---

## Configuration

All configuration lives in `core/config.py`:

```python
# LLM Provider — Groq is the only supported provider
LLM_PROVIDER = "groq"

# API key + model (model is .env-configurable, with a working default)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Rate limiting — shared across all agents via core/llm.py
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "20"))

# Data directories (all created automatically on import)
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
LANDING_DIR  = DATA_DIR / "landing"
BRONZE_DIR   = DATA_DIR / "bronze_layer"
SILVER_DIR   = DATA_DIR / "silver_layer"
GOLD_DIR     = DATA_DIR / "gold_layer"
STTM_DIR     = DATA_DIR / "sttm"
PROFILES_DIR = DATA_DIR / "profiles"
TRACES_DIR   = DATA_DIR / "traces"
REPORTS_DIR  = BASE_DIR / "reports"
AUDIT_DIR    = BASE_DIR / "audit_logs"
CHROMA_DIR   = BASE_DIR / ".chroma"
```

---

## Observability

### Agent traces

Every agent invocation writes a full trace to `data/traces/`:

```
data/traces/trace_{agent_name}_{run_id[:8]}.json
```

Each trace contains:

```json
{
  "agent": "bronze_agent",
  "run_id": "abc12345-def6-7890-...",
  "started_at": "2025-05-17T10:30:00Z",
  "input": {
    "input_files": ["data/landing/sales.csv"],
    "sttm_path": "data/sttm/sttm_bronze_abc12345.csv"
  },
  "plan": "I will first inspect the CSV files to understand their schema, then apply the STTM rules...",
  "tool_calls": [
    { "tool": "inspect_task_tool", "args": {} },
    { "tool": "bronze_ingestion_tool", "args": {} }
  ],
  "reasoning_steps": [
    { "role": "task_input", "content": "Ingest the raw CSV files..." },
    { "role": "ai_reasoning", "content": "I can see 3 columns..." },
    { "role": "tool_result", "tool": "inspect_task_tool", "content": "..." }
  ],
  "output": {
    "output_paths": ["data/bronze_layer/sales_bronze.parquet"]
  },
  "duration_seconds": 4.2,
  "status": "success"
}
```

### Audit logs

`core/audit.py` writes append-only JSONL event logs to `audit_logs/<run_id>.jsonl`. Every phase start/complete/fail event is recorded with agent name, file paths produced, decisions made, and error details. Useful for debugging failed runs without re-running the pipeline.

### LLM infrastructure (`core/llm.py`)

All agents share a single `make_llm()` factory backed by one `InMemoryRateLimiter` — configurable via `LLM_REQUESTS_PER_MINUTE` — to prevent rate-limit breaches across the full pipeline. The module also contains `invoke_agent_with_tool_recovery()` which recovers from Groq-specific malformed tool call errors (phantom tool names, Harmony-tag corruption, bloated unparseable arguments).

---

## Agent Handoffs

The `run_id` and `business_intent` travel through the entire pipeline. Every file produced (profile, STTMs, Parquets, report, traces) includes `run_id` in its filename — tying a complete run together for debugging.

| Handoff | From | To | Via | What travels |
|---------|------|-----|-----|-------------|
| H0 | User | Orchestrator | Function call | CSV paths + business_intent |
| H1 | Orchestrator | Profiler | Tool call | CSV paths + run_id + goal |
| H2 | Profiler | STTM (Bronze) | Scratchpad | profile.json path |
| H3 | STTM | Bronze Agent | PipelineState | sttm_bronze.csv path *(after human approval)* |
| H4 | Bronze | STTM (Silver) | PipelineState | Bronze Parquet paths *(after human review)* |
| H5 | STTM | Silver Agent | PipelineState | sttm_silver.csv path *(after human approval)* |
| H6 | Silver | STTM (Gold) | PipelineState | Silver Parquet paths *(after human review)* |
| H7 | STTM | Gold Agent | PipelineState | sttm_gold.csv path *(after human approval)* |
| H8 | Gold | Reporter | PipelineState | Gold Parquet paths *(after human review)* |
| H9 | Reporter | Streamlit UI | PipelineState | report.html path |

---

## FAQ

**Q: Why are there 7 phases instead of running everything at once?**

The STTM is the transformation recipe. If it is wrong, all downstream data is wrong. Phases create human review checkpoints before and after each layer executes — preventing errors from propagating silently through the pipeline.

**Q: Why does every agent have an inspect tool AND an execution tool?**

The inspect tool is lightweight — it reads metadata without transforming any data. This lets the agent understand what it is working with and form a plan before committing to execution. It is the difference between a scripted agent and an autonomous one.

**Q: What is the scratchpad? Is it a special library?**

No — it is a plain Python dictionary (`{}`). Both tools in a phase are created inside the same factory function, so they share the same dict via Python closure. Tool 1 writes a path, Tool 2 reads it. No library needed.

**Q: Why use closures instead of passing file paths as tool arguments?**

LLMs are unreliable at reproducing exact file system paths and UUIDs — they paraphrase and truncate. Closures capture the real values at tool creation time. The LLM just calls the tool; it never needs to know or reproduce the paths.

**Q: Can I use a different LLM model?**

Yes, within Groq's hosted models — set `GROQ_MODEL` in your `.env` file (default: `openai/gpt-oss-120b`). The `make_llm()` factory in `core/llm.py` is the single point where the model is constructed, used by every agent.

**Q: What happens if a phase fails partway through?**

All outputs up to that point are saved to disk and in `PipelineState`. You can call the failed phase function again with the saved state — the pipeline resumes from that phase without reprocessing earlier phases.

**Q: Where do I find the generated STTM to review before approving?**

In `data/sttm/`. The Streamlit UI also renders the STTM as a table directly in the browser at the approval gate. You can edit the CSV file before approving, or enter feedback text to trigger Bronze STTM regeneration via `run_bronze_sttm_regeneration`.

**Q: Why does the Gold STTM use the business intent but Bronze and Silver do not?**

Bronze and Silver are structural transformations — they clean and standardise data regardless of what question will be asked. Gold is where aggregations and joins are shaped around the analytics goal, so the business intent first becomes relevant at that stage.

---

## License

MIT License — see `LICENSE` for details.

---

*Built with LangChain · LangGraph · pandas · DuckDB · Plotly · Streamlit · ChromaDB*
