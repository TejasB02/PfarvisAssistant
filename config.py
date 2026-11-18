# =============================================================================
# config.py - shared settings (owned by ALL FOUR; change only by agreement)
# Receives : optional environment-variable overrides.
# Hands off: constants imported by every module, and the offline switches for
#            the ML libraries.
# RULE     : `import config` must be the FIRST import in every entry point
#            (app.py, tests/conftest.py, scripts/prefetch_models.py) so the
#            offline variables are set before any project or ML import.
# =============================================================================
"""Central configuration for Local Notes Assistant. Nothing here leaves the machine."""

from __future__ import annotations

import os
from pathlib import Path

# --- Offline switches -------------------------------------------------------
# setdefault so a developer can still override them in their own shell.
# sentence-transformers / transformers read these at import time, which is why
# this module must be imported before them.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB telemetry off

PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --- Paths --------------------------------------------------------------------
NOTE_DIR: str = _project_path(os.environ.get("NOTE_DIR", "sample_notes"))
DATA_DIR: str = _project_path(os.environ.get("DATA_DIR", "data"))
CHROMA_DIR: str = str(Path(DATA_DIR) / "chroma")
CHROMA_COLLECTION: str = "notes"
# Local folder filled once by scripts/prefetch_models.py. Never load by hub name.
EMBED_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")

# --- Local LLM (Ollama) -------------------------------------------------------
# Fixed to the loopback interface on purpose; deliberately NOT read from env.
OLLAMA_HOST: str = "http://127.0.0.1:11434"
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3")
# The first call loads the model into memory and can be slow.
OLLAMA_TIMEOUT_SECONDS: int = _env_int("OLLAMA_TIMEOUT_SECONDS", 120)

# --- Ingestion ---------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg")
# Optional full path to the local tesseract binary (handy on Windows).
TESSERACT_CMD: str | None = os.environ.get("TESSERACT_CMD") or None

# --- Retrieval & guardrails ---------------------------------------------------
RETRIEVAL_TOP_K: int = _env_int("RETRIEVAL_TOP_K", 5)
# TODO: Tune once Person 2's real embeddings land: set between the best score for
# in-notes questions and for the parking question. (pipeline.ask logs the best
# semantic score of every query at INFO.)
MIN_SIMILARITY: float = _env_float("MIN_SIMILARITY", 0.35)  # cosine similarity, 0..1
GRAPH_HOPS: int = _env_int("GRAPH_HOPS", 1)

# --- Reminders ---------------------------------------------------------------
REMINDER_LEAD_MINUTES: int = _env_int("REMINDER_LEAD_MINUTES", 15)
SCHEDULER_INTERVAL_SECONDS: int = _env_int("SCHEDULER_INTERVAL_SECONDS", 30)

# --- Misc ---------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
