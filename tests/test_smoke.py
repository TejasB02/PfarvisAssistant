"""Scaffold smoke tests: modules import, signatures match the README, data flows,
nothing points off the machine, and the Streamlit app runs end to end."""

import ast
import importlib
import inspect
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from contracts import NOTE_FIELDS, make_note_id

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "config", "contracts",
    "module_ingest", "module_ingest.parser", "module_ingest.ocr",
    "module_ingest.watcher", "module_ingest.extract",
    "module_storage", "module_storage.vector_db", "module_storage.knowledge_graph",
    "module_storage.store",
    "module_rag", "module_rag.llm_client", "module_rag.guardrails",
    "module_rag.engine", "module_rag.pipeline",
    "module_ui", "module_ui.ui_app", "module_ui.notifier",
]

# (module, attribute path) -> exact parameter names (the handoff contract)
SIGNATURES = {
    ("module_ingest.parser", "scan_and_parse"): ["dir_path"],
    ("module_ingest.parser", "parse_file"): ["path"],
    ("module_ingest.ocr", "image_to_text"): ["path"],
    ("module_ingest.extract", "guess_client"): ["text"],
    ("module_ingest.extract", "extract_tags"): ["text"],
    ("module_ingest.extract", "extract_event_time"): ["text", "reference"],
    ("module_ingest.watcher", "watch"): ["dir_path", "on_change", "on_delete"],
    ("module_storage.store", "NotesStore.add"): ["self", "notes"],
    ("module_storage.store", "NotesStore.query"): ["self", "text", "k"],
    ("module_storage.store", "NotesStore.hybrid_search"): ["self", "query", "k", "hops"],
    ("module_storage.store", "NotesStore.get_note"): ["self", "note_id"],
    ("module_storage.store", "NotesStore.all_notes"): ["self"],
    ("module_storage.store", "NotesStore.remove"): ["self", "note_id"],
    ("module_rag.llm_client", "is_available"): [],
    ("module_rag.llm_client", "chat"): ["messages", "temperature"],
    ("module_rag.guardrails", "has_enough_context"): ["context"],
    ("module_rag.guardrails", "validate_citations"): ["answer", "context"],
    ("module_rag.engine", "answer_user_query"): ["query", "retrieved_context", "chat_history"],
    ("module_rag.pipeline", "rewrite_query"): ["query", "chat_history"],
    ("module_rag.pipeline", "ask"): ["query", "chat_history", "store", "pinned_note_ids"],
    ("module_ui.ui_app", "main"): ["store", "notes"],
    ("module_ui.notifier", "start_scheduler"): ["get_notes"],
    ("module_ui.notifier", "notify_desktop"): ["title", "message"],
    ("module_ui.notifier", "pop_pending_reminders"): [],
}

DEFAULTS = {
    ("module_storage.store", "NotesStore.query"): {"k": 5},
    ("module_storage.store", "NotesStore.hybrid_search"): {"k": 5, "hops": 1},
    ("module_rag.llm_client", "chat"): {"temperature": 0.0},
    ("module_rag.pipeline", "ask"): {"pinned_note_ids": None},
}

HEAVY = ["chromadb", "sentence_transformers", "ollama", "pypdf", "docx", "pytesseract",
         "PIL", "apscheduler", "plyer", "watchdog", "networkx", "dateparser"]


def _resolve(module_name: str, attr_path: str):
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("key", list(SIGNATURES), ids=lambda k: f"{k[0]}.{k[1]}")
def test_public_signatures(key):
    func = _resolve(*key)
    assert callable(func)
    params = inspect.signature(func).parameters
    assert list(params) == SIGNATURES[key]
    for name, default in DEFAULTS.get(key, {}).items():
        assert params[name].default == default
    assert inspect.getdoc(func), f"{key} needs a docstring"


def test_system_prompt_exists():
    from module_rag import guardrails
    assert isinstance(guardrails.SYSTEM_PROMPT, str) and guardrails.SYSTEM_PROMPT.strip()


def test_core_modules_import_without_heavy_dependencies():
    """Heavy libraries must only be imported inside functions."""
    code = (
        "import sys; sys.path.insert(0, %r); import config\n"
        "import contracts, module_ingest.parser, module_ingest.ocr, module_ingest.extract, "
        "module_ingest.watcher, module_storage.store, module_rag.pipeline, module_ui.notifier\n"
        "print(','.join(m for m in %r if m in sys.modules))"
    ) % (str(ROOT), HEAVY)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"heavy modules imported at module level: {out.stdout}"


def test_scan_and_parse_sample_notes_schema(monkeypatch):
    from module_ingest.parser import scan_and_parse

    monkeypatch.chdir(ROOT)
    notes = scan_and_parse("sample_notes")
    assert notes and isinstance(notes, list)
    for note in notes:
        assert isinstance(note, dict)
        assert list(note) == list(NOTE_FIELDS)
        assert Path(note["filepath"]).is_absolute()
        assert note["id"] == make_note_id(note["filepath"])
        assert isinstance(note["title"], str) and isinstance(note["content"], str)
        assert note["timestamp"] is None or isinstance(note["timestamp"], datetime)
        assert note["event_time"] is None or isinstance(note["event_time"], datetime)
        assert note["client"] is None or isinstance(note["client"], str)
        assert isinstance(note["tags"], list) and all(isinstance(t, str) for t in note["tags"])
    soon = datetime.now() + timedelta(minutes=30)
    assert any(n["event_time"] and n["event_time"] <= soon for n in notes), \
        "need one note due soon so the reminder demo works"


def test_scan_and_parse_missing_folder_returns_empty(tmp_path):
    from module_ingest.parser import scan_and_parse
    assert scan_and_parse(tmp_path / "does-not-exist") == []


def test_pipeline_end_to_end_with_stubs(monkeypatch):
    from module_ingest.parser import scan_and_parse
    from module_rag import pipeline
    from module_storage.store import NotesStore

    monkeypatch.chdir(ROOT)
    store = NotesStore()
    store.add(scan_and_parse("sample_notes"))
    result = pipeline.ask("When is my call with the clients?", [], store)
    assert set(result) == {"answer", "sources", "grounded"}
    assert pipeline.ask("   ", [], store)["grounded"] is False


def test_reminder_fires_once_and_reaches_queue(monkeypatch):
    from module_ui import notifier

    monkeypatch.setattr(notifier, "notify_desktop", lambda title, message: None)
    now = datetime(2026, 11, 18, 15, 50)
    note = {"id": "test-reminder-note", "title": "Client call", "content": "4pm call",
            "filepath": str(ROOT / "sample_notes" / "x.txt"), "event_time": now + timedelta(minutes=10)}
    later = dict(note, id="test-later-note", event_time=now + timedelta(hours=3))

    assert [r["note_id"] for r in notifier.check_due_reminders(lambda: [note, later], now=now)] \
        == ["test-reminder-note"]
    assert notifier.check_due_reminders(lambda: [note], now=now) == []
    popped = [r for r in notifier.pop_pending_reminders() if r["note_id"].startswith("test-")]
    assert len(popped) == 1 and popped[0]["filename"] == "x.txt"


def test_app_imports_config_first():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    first_import = next(n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert isinstance(first_import, ast.Import) and first_import.names[0].name == "config"


def test_streamlit_config_is_local_only():
    text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r"gatherUsageStats\s*=\s*false", text)
    assert re.search(r'address\s*=\s*"127\.0\.0\.1"', text)
    assert re.search(r"headless\s*=\s*true", text)


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "data", "models"}
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
URL_RE = re.compile(r"https?://([^/\s:'\"<>)\]`]+)")


def test_no_remote_urls_in_project_files():
    exempt = ROOT / "scripts" / "prefetch_models.py"
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == exempt:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:  # binary (e.g. the sample PNG)
            continue
        text = raw.decode("utf-8", errors="ignore")
        for host in URL_RE.findall(text):
            if host.lower() not in ALLOWED_HOSTS:
                offenders.append(f"{path.relative_to(ROOT)}: {host}")
    assert not offenders, "remote URLs found:\n" + "\n".join(offenders)


def test_streamlit_app_runs_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from module_ui import notifier

    monkeypatch.setattr(notifier, "notify_desktop", lambda title, message: None)
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    at.chat_input[0].set_value("When is my call with the clients?").run()
    assert not at.exception, at.exception
    roles = [m.name for m in at.chat_message]
    assert roles[-2:] == ["user", "assistant"]
