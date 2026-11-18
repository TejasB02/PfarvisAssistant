# =============================================================================
# module_ingest/watcher.py - owner: Person 1 (Ingestion & OCR) - STRETCH GOAL
# Receives : a folder path plus two callbacks from app.py / Person 4.
# Hands off: on_change(note_dict) when a supported file is created/modified,
#            on_delete(note_id) when one is removed -> typically wired to
#            Person 2's NotesStore.add([note]) / NotesStore.remove(note_id).
# Engine   : watchdog (local filesystem events only).
# STATUS   : STUB - logs and returns without watching.
# =============================================================================
"""Optional live re-ingestion when files in the notes folder change."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def watch(
    dir_path: str | Path,
    on_change: Callable[[dict], None],
    on_delete: Callable[[str], None],
) -> None:
    """Start watching ``dir_path`` recursively in a background thread.

    Inputs:
        dir_path: folder to watch (usually config.NOTE_DIR).
        on_change: called with the new note dict (from parser.parse_file) when a
            supported file is created, modified or moved in.
        on_delete: called with make_note_id(filepath) when a file is deleted or
            moved out.
    Output:
        None. Returns immediately; the watchdog Observer runs as a daemon thread.
    Edge cases:
        - Editors fire several events per save: debounce (~1 s) per path.
        - Ignore unsupported extensions, temp files ("~$*", "*.swp", ".*").
        - parse_file returning None -> do not call on_change.
        - Never let an exception in a callback kill the observer thread.
        - Callbacks run on the watcher thread: they must not touch
          st.session_state.
    """
    # TODO(person 1): implement with watchdog (import inside the function):
    #   from watchdog.observers import Observer
    #   from watchdog.events import FileSystemEventHandler
    log.info("watcher.watch is a stub; live re-ingest is disabled for %s", dir_path)
