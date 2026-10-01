import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.sttm_generator import (
    _prepare_bronze_context,
    _prepare_silver_context,
    _prepare_gold_context,
    _extract_sttm_rows,
    generate_bronze_sttm,
    generate_silver_sttm,
    generate_gold_sttm,
)


BRONZE_ROW = {
    "source_schema": "CSV", "source_table": "data.csv",
    "source_column": "id", "target_schema": "Bronze",
    "target_table": "bronze_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}
SILVER_ROW = {
    "source_schema": "Bronze", "source_table": "bronze_data",
    "source_column": "id", "target_schema": "Silver",
    "target_table": "silver_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}
GOLD_ROW = {
    "source_schema": "Silver", "source_table": "silver_data",
    "source_column": "id", "target_schema": "Gold",
    "target_table": "gold_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}


def _mock_agent(tool_index: int):
    """Return a fake create_agent whose agent calls the REAL generation tool at
    `tool_index` in `tools` (1=bronze, 2=silver, 3=gold -- _make_sttm_tools always
    returns [inspect, bronze, silver, gold] in that fixed order). This exercises
    the tool's actual scratchpad-populating side effect, exactly like the live
    ReAct agent would -- a fabricated final message alone doesn't save an STTM
    CSV or populate scratchpad["sttm_path"], since that save happens inside the
    tool itself.

    Callers must also patch agents.sttm_generator.make_llm, since the generation
    tool makes its OWN separate inner LLM call (distinct from the outer agent's
    LLM) to actually generate the STTM rows -- see _fake_llm_returning.
    """
    def fake_create_agent(llm, tools, **kwargs):
        mock_agent = MagicMock()
        def invoke(inputs):
            tool_result = tools[tool_index].invoke({})
            return {"messages": [MagicMock(content=tool_result)]}
        mock_agent.invoke = invoke
        return mock_agent
    return fake_create_agent


def _fake_llm_returning(rows: list[dict]):
    """A minimal fake LLM whose .invoke() returns a JSON array response, for
    patching agents.sttm_generator.make_llm -- the generation tools' own inner
    LLM call that actually produces the STTM rows."""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps(rows))
    return fake_llm


# ---------------------------------------------------------------------------
# _prepare_bronze_context
# ---------------------------------------------------------------------------
class TestPrepareBronzeContext:
    def test_returns_profile_dict(self, tmp_path):
        profile = {"files": ["a.csv"], "datasets": {"a": {"columns": {"id": {}}}}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))
        result = _prepare_bronze_context(str(p))
        assert result == profile

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _prepare_bronze_context(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# _prepare_silver_context
# ---------------------------------------------------------------------------
class TestPrepareSilverContext:
    def test_filters_to_approved_columns(self, tmp_path):
        df = pd.DataFrame({"approved_col": [1, 2], "dropped_col": [3, 4], "_meta": ["a", "b"]})
        parquet_path = tmp_path / "bronze.parquet"
        df.to_parquet(str(parquet_path), index=False)

        sttm = pd.DataFrame({"target_column": ["approved_col"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_silver_context([str(parquet_path)], str(sttm_path))
        assert len(result) == 1
        assert "approved_col" in result[0]["columns"]
        assert "_meta" in result[0]["columns"]
        assert "dropped_col" not in result[0]["columns"]

    def test_returns_filename_not_full_path(self, tmp_path):
        df = pd.DataFrame({"col": [1]})
        p = tmp_path / "bronze_data.parquet"
        df.to_parquet(str(p), index=False)
        sttm = pd.DataFrame({"target_column": ["col"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_silver_context([str(p)], str(sttm_path))
        assert result[0]["filename"] == "bronze_data.parquet"


# ---------------------------------------------------------------------------
# _prepare_gold_context
# ---------------------------------------------------------------------------
class TestPrepareGoldContext:
    def test_filters_to_approved_columns(self, tmp_path):
        df = pd.DataFrame({"revenue": [10], "cost": [5], "_sk": [1]})
        p = tmp_path / "silver.parquet"
        df.to_parquet(str(p), index=False)
        sttm = pd.DataFrame({"target_column": ["revenue"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_gold_context([str(p)], str(sttm_path))
        assert "revenue" in result[0]["columns"]
        assert "_sk" in result[0]["columns"]
        assert "cost" not in result[0]["columns"]


# ---------------------------------------------------------------------------
# _extract_sttm_rows
# ---------------------------------------------------------------------------
class TestExtractSttmRows:
    def test_extracts_plain_json_array(self):
        result = {"messages": [MagicMock(content=json.dumps([BRONZE_ROW]))]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]

    def test_extracts_json_with_fences(self):
        content = f"```json\n{json.dumps([BRONZE_ROW])}\n```"
        result = {"messages": [MagicMock(content=content)]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]

    def test_returns_empty_list_when_no_array(self):
        result = {"messages": [MagicMock(content="No rules generated.")]}
        assert _extract_sttm_rows(result) == []

    def test_prefers_last_message(self):
        early = MagicMock(content=json.dumps([{"source_column": "old"}]))
        late = MagicMock(content=json.dumps([BRONZE_ROW]))
        result = {"messages": [early, late]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]


# ---------------------------------------------------------------------------
# generate_bronze_sttm
# ---------------------------------------------------------------------------
class TestGenerateBronzeSttm:
    def test_saves_csv_and_returns_path(self, tmp_path):
        profile = {"files": ["a.csv"], "datasets": {}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))

        with patch("agents.sttm_generator.create_agent", side_effect=_mock_agent(tool_index=1)), \
             patch("agents.sttm_generator.make_llm", return_value=_fake_llm_returning([BRONZE_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_bronze_sttm(
                str(p), "run-b1",
                task_description="Generate Bronze STTM for run-b1.",
            )

        assert Path(path).exists()
        df = pd.read_csv(path)
        assert "source_column" in df.columns
        assert len(df) == 1

    def test_uses_bronze_sttm_agent_prompt(self, tmp_path):
        profile = {"files": [], "datasets": {}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))

        captured = {}

        def fake_create_agent(llm, tools, **kwargs):
            captured["system_prompt"] = kwargs.get("system_prompt", "")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {"messages": [MagicMock(content=json.dumps([BRONZE_ROW]))]}
            return mock_agent

        with patch("agents.sttm_generator.create_agent", side_effect=fake_create_agent), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            generate_bronze_sttm(str(p), "run-b2",
                                  task_description="Generate Bronze STTM.")

        assert "Bronze" in captured["system_prompt"]


# ---------------------------------------------------------------------------
# generate_silver_sttm
# ---------------------------------------------------------------------------
class TestGenerateSilverSttm:
    def test_saves_csv_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"id": [1], "val": [2]})
        parquet = tmp_path / "bronze.parquet"
        df.to_parquet(str(parquet), index=False)
        sttm_csv = tmp_path / "bronze_sttm.csv"
        pd.DataFrame({"target_column": ["id", "val"]}).to_csv(str(sttm_csv), index=False)

        with patch("agents.sttm_generator.create_agent", side_effect=_mock_agent(tool_index=2)), \
             patch("agents.sttm_generator.make_llm", return_value=_fake_llm_returning([SILVER_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_silver_sttm(
                [str(parquet)], str(sttm_csv), "run-s1",
                task_description="Generate Silver STTM for run-s1.",
            )

        assert Path(path).exists()
        df_out = pd.read_csv(path)
        assert len(df_out) == 1


# ---------------------------------------------------------------------------
# generate_gold_sttm
# ---------------------------------------------------------------------------
class TestGenerateGoldSttm:
    def test_saves_csv_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"revenue": [100], "region": ["EU"]})
        parquet = tmp_path / "silver.parquet"
        df.to_parquet(str(parquet), index=False)
        sttm_csv = tmp_path / "silver_sttm.csv"
        pd.DataFrame({"target_column": ["revenue", "region"]}).to_csv(str(sttm_csv), index=False)

        with patch("agents.sttm_generator.create_agent", side_effect=_mock_agent(tool_index=3)), \
             patch("agents.sttm_generator.make_llm", return_value=_fake_llm_returning([GOLD_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_gold_sttm(
                [str(parquet)], str(sttm_csv), "intent", "run-g1",
                task_description="Generate Gold STTM for run-g1.",
            )

        assert Path(path).exists()
        df_out = pd.read_csv(path)
        assert len(df_out) == 1

