# =============================================================================
# module_ingest/watcher.py - owner: Person 1 (Ingestion & OCR) - STRETCH GOAL
# Receives : a folder path plus two callbacks from app.py / Person 4.
# Hands off: on_change(note_dict) when a supported file is created/modified,
#            on_delete(note_id) when one is removed -> typically wired to
#            Person 2's NotesStore.add([note]) / NotesStore.remove(note_id).
# Engine   : watchdog (local filesystem events only).
# =============================================================================
"""Optional live re-ingestion when files in the notes folder change."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import config
from contracts import make_note_id
from module_ingest import parser

log = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 1.0


def _is_ignored(path: Path, root: Path) -> bool:
    """Return True if ``path`` is hidden, an editor temp file, or unsupported."""
    name = path.name
    if (
        name.startswith(".")
        or name.startswith("~$")
        or name.endswith((".swp", ".swx", ".tmp", "~"))
    ):
        return True

    if path.is_relative_to(root):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            return True

    return path.suffix.lower() not in config.SUPPORTED_EXTENSIONS


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
        - parse_file returning None -> do not call on_change; if the file was
          indexed before (e.g. it was emptied), call on_delete so its stale
          vector disappears.
        - A whole folder deleted / moved out -> on_delete for every file the
          watcher knew under it; a folder moved in -> on_change for its files.
        - Never let an exception in a callback kill the observer thread.
        - Callbacks run on the watcher thread: they must not touch
          st.session_state.
    """
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    root = Path(dir_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        log.warning("Cannot watch missing notes folder: %s", root)
        return

    lock = threading.Lock()
    timers: dict[Path, threading.Timer] = {}
    # Every supported file currently under root. Folder-level events carry no
    # per-file paths, so this is how a deleted folder maps back to note ids.
    known: set[Path] = {
        p.resolve() for p in root.rglob("*") if p.is_file() and not _is_ignored(p.resolve(), root)
    }

    def _cancel_timer(target: Path) -> None:
        with lock:
            existing = timers.pop(target, None)
        if existing is not None:
            existing.cancel()

    def _flush_change(target: Path) -> None:
        with lock:
            timers.pop(target, None)
        if not target.is_file():
            return
        with lock:
            known.add(target)
        try:
            note = parser.parse_file(target)
        except Exception as exc:  # noqa: BLE001
            log.warning("Watcher parse error for %s: %s", target, exc)
            return

        if note is None:
            # Emptied or unreadable now: drop any earlier version of it.
            _emit_delete(target)
            return

        try:
            on_change(note)
        except Exception as exc:  # noqa: BLE001
            log.warning("Watcher on_change callback error for %s: %s", target, exc)

    def _schedule_change(raw_path: str | bytes) -> None:
        target = Path(str(raw_path)).expanduser().resolve()
        if _is_ignored(target, root):
            return
        with lock:
            existing = timers.pop(target, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(_DEBOUNCE_SECONDS, _flush_change, args=(target,))
            timer.daemon = True
            timers[target] = timer
            timer.start()

    def _emit_delete(target: Path) -> None:
        try:
            on_delete(make_note_id(target))
        except Exception as exc:  # noqa: BLE001
            log.warning("Watcher on_delete callback error for %s: %s", target, exc)

    def _handle_delete(raw_path: str | bytes) -> None:
        target = Path(str(raw_path)).expanduser().resolve()
        if _is_ignored(target, root):
            return
        _cancel_timer(target)
        with lock:
            known.discard(target)
        _emit_delete(target)

    def _handle_dir_delete(raw_path: str | bytes) -> None:
        folder = Path(str(raw_path)).expanduser().resolve()
        with lock:
            gone = [p for p in known if p.is_relative_to(folder)]
        for target in gone:
            _handle_delete(target)

    def _handle_dir_added(raw_path: str | bytes) -> None:
        folder = Path(str(raw_path)).expanduser().resolve()
        if not folder.is_dir() or not folder.is_relative_to(root):
            return
        for path in folder.rglob("*"):
            if path.is_file():
                _schedule_change(path)

    class _NotesEventHandler(FileSystemEventHandler):
        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                _schedule_change(event.src_path)

        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                _schedule_change(event.src_path)

        def on_deleted(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                _handle_delete(event.src_path)
            # Windows reports a deleted folder as a plain (non-directory) event,
            # so always sweep anything the watcher knew under this path.
            _handle_dir_delete(event.src_path)

        def on_moved(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                _handle_dir_delete(event.src_path)
                _handle_dir_added(event.dest_path)
            else:
                _handle_delete(event.src_path)
                _schedule_change(event.dest_path)

    observer = Observer()
    observer.daemon = True
    observer.schedule(_NotesEventHandler(), str(root), recursive=True)
    observer.start()
    log.info("Started live notes watcher on %s", root)