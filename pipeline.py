"""Ingestion pipeline: build the vector DB from the notes folder and keep it in sync.

``run_ingestion_pipeline`` does the initial full scan. ``start_live_sync`` then
watches ``config.NOTE_DIR`` in a background thread, so any file that is
created, edited, moved or deleted there is updated in ChromaDB + the knowledge
graph immediately, with no manual re-run. The Streamlit app (app.py) starts
both automatically; ``python pipeline.py`` does the same from a terminal.
"""

from __future__ import annotations

import config  # noqa: F401 - must be first: sets the offline switches

import logging
import threading
import time
from pathlib import Path

from module_ingest import parser, watcher
from module_storage import NotesStore

log = logging.getLogger(__name__)

_watched_roots: set[Path] = set()
_watch_lock = threading.Lock()


def run_ingestion_pipeline(
    notes_dir: str | Path = config.NOTE_DIR,
    store: NotesStore | None = None,
    live_sync: bool = False,
) -> NotesStore:
    """Scan and parse every note in ``notes_dir`` and index it in the store.

    Notes deleted since the last run are pruned from ChromaDB by NotesStore.add.
    With ``live_sync=True`` the folder keeps being watched afterwards.
    """
    store = store or NotesStore(config=config)
    log.info("Scanning and parsing notes from: %s", notes_dir)
    store.add(parser.scan_and_parse(notes_dir))
    log.info("Indexed %d notes into the vector database and knowledge graph.", len(store.all_notes()))
    if live_sync:
        start_live_sync(store, notes_dir)
    return store


def start_live_sync(store: NotesStore, notes_dir: str | Path = config.NOTE_DIR) -> bool:
    """Mirror file changes under ``notes_dir`` into ``store`` from a daemon thread.

    created / modified / moved in -> store.add([note])  (upsert by note id)
    deleted / moved out / emptied -> store.remove(note_id)
    Returns False if that folder is already being watched (e.g. on a Streamlit
    rerun) or does not exist.
    """
    root = Path(notes_dir).expanduser().resolve()
    with _watch_lock:
        if root in _watched_roots:
            return False
        if not root.is_dir():
            log.warning("Live sync not started, notes folder missing: %s", root)
            return False
        _watched_roots.add(root)

    def on_change(note: dict) -> None:
        store.add([note])
        log.info("Live sync: indexed %s", note["filepath"])

    def on_delete(note_id: str) -> None:
        if store.get_note(note_id) is None:
            return  # never indexed (or already removed)
        store.remove(note_id)
        log.info("Live sync: removed note %s", note_id)

    watcher.watch(root, on_change=on_change, on_delete=on_delete)
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_ingestion_pipeline(live_sync=True)
    print(f"Watching {config.NOTE_DIR} for changes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
