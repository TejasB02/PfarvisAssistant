# =============================================================================
# module_ingest - Person 1: Ingestion & OCR
# Receives : a folder path (config.NOTE_DIR) full of messy nested notes.
# Hands off: list[dict] (each == asdict(contracts.NoteItem))
#            -> Person 2 (NotesStore.add) and Person 4 (sidebar, reminders).
# Files    : parser.py (scan + dispatch), ocr.py (images), extract.py
#            (client / tags / event_time), watcher.py (live re-ingest, stretch).
# =============================================================================
"""Ingestion package: turn files on disk into note dicts."""
