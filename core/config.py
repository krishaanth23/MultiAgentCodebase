import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Agent reasoning/plan text is LLM-generated and can contain arbitrary Unicode
# punctuation (em dashes, non-breaking hyphens, smart quotes, ...). On Windows,
# the console's default codepage (cp1252) can't encode many of those characters,
# and a bare print() of that text crashes the whole run. Reconfigure stdout/stderr
# once, here, so unencodable characters are replaced instead of raising.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"
TRACES_DIR = DATA_DIR / "traces"

# LLM Configuration
LLM_PROVIDER = "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"  # Currently supported Groq model
#"openai/gpt-oss-20b"

# Ensure directories exist
for d in [LANDING_DIR, PROFILES_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR, AUDIT_DIR, CHROMA_DIR, TRACES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
