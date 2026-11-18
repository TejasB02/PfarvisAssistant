"""Test entry point: make the project importable and set offline env vars first."""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tests get their own Chroma folder, so they never touch the app's data/ (the
# app may be running and holding it open).
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="notes-test-data-"))

import config  # noqa: E402,F401  - must precede any project or ML import
