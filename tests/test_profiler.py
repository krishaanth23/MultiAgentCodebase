import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _mock_profiler_agent(analysis: dict):
    """Return a fake create_agent that calls the real tools, then returns the analysis JSON."""
    def fake_create_agent(llm, tools, system_prompt):
        mock_agent = MagicMock()

        def invoke(inputs):
            inspect_tool, stats_tool = tools
            messages = [
                MagicMock(content=inspect_tool.invoke({})),
                MagicMock(content=stats_tool.invoke({})),
                MagicMock(content=json.dumps(analysis)),
            ]
            return {"messages": messages}

        mock_agent.invoke = invoke
        return mock_agent

    return fake_create_agent


class TestProfiler:
    def test_profile_dataset_creates_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create a test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.0, 20.0, None]})
        df.to_csv(csv_path, index=False)

        analysis = {
            "semantic_meanings": {"id": "unique identifier", "name": "entity name", "value": "numeric measurement"},
            "join_keys": [],
            "quality_notes": [],
        }

        with patch("agents.profiler.create_agent", side_effect=_mock_profiler_agent(analysis)), \
             patch("agents.profiler.AuditLogger"):
            from agents.profiler import profile_dataset
            result = profile_dataset(str(csv_path), "test-run", "Profile this dataset.")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert profile["datasets"]["test"]["shape"]["rows"] == 3
        assert profile["datasets"]["test"]["shape"]["columns"] == 3
        assert "id" in profile["datasets"]["test"]["columns"]
        assert profile["analysis"] == analysis

    def test_profile_multiple_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create test CSVs
        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        pd.DataFrame({"product_id": [1, 2], "revenue": [100, 200]}).to_csv(csv1, index=False)
        pd.DataFrame({"product_id": [1, 2], "name": ["A", "B"]}).to_csv(csv2, index=False)

        analysis = {"semantic_meanings": {}, "join_keys": ["product_id"], "quality_notes": []}

        with patch("agents.profiler.create_agent", side_effect=_mock_profiler_agent(analysis)), \
             patch("agents.profiler.AuditLogger"):
            from agents.profiler import profile_multiple_datasets
            result = profile_multiple_datasets([str(csv1), str(csv2)], "test-run", "Profile these datasets.")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert len(profile["datasets"]) == 2
