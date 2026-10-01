import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from agents.orchestrator import (
        run_until_bronze_sttm,
        run_bronze_execution,
        run_silver_sttm_generation,
        run_silver_execution,
        run_gold_sttm_generation,
        run_gold_execution,
        run_report_generation,
    )
    print("SUCCESS! All imports worked.")
    print(f"run_until_bronze_sttm: {run_until_bronze_sttm}")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
