# =============================================================================
# app.py - Streamlit entry point - owner: Person 4 (Frontend, State & Notifications)
# Run      : streamlit run app.py    (then open http://localhost:8501)
# Receives : nothing; reads config.NOTE_DIR.
# Wires    : Person 1 scan_and_parse -> Person 2 NotesStore.add ->
#            Person 4 notifier.start_scheduler(store.all_notes) ->
#            Person 4 ui_app.main (which calls Person 3 pipeline.ask).
# RULE     : `import config` stays the FIRST import (offline env vars).
# =============================================================================
import config  # noqa: F401  - MUST be first: sets offline env vars before ML imports

import logging

import streamlit as st

from module_ingest.parser import scan_and_parse
from module_storage.store import NotesStore
from module_ui import notifier, ui_app

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Local Notes Assistant", page_icon="🗂️", layout="wide")


@st.cache_resource(show_spinner="Reading and indexing your notes...")
def load_store_and_notes() -> tuple[NotesStore, list[dict]]:
    """Scan the notes folder and index it once per server process."""
    # TODO(person 4): add a "Rescan notes" button that clears this cache.
    notes = scan_and_parse(config.NOTE_DIR)
    store = NotesStore()
    store.add(notes)
    return store, notes


@st.cache_resource(show_spinner=False)
def start_reminders() -> bool:
    """Start the background reminder scheduler once per server process."""
    # TODO(person 4): optionally start module_ingest.watcher.watch here too.
    store, _ = load_store_and_notes()
    notifier.start_scheduler(store.all_notes)
    return True


store, notes = load_store_and_notes()
start_reminders()
ui_app.main(store, notes)
