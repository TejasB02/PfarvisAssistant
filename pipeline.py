from pathlib import Path

import config
from module_ingest import parser
from module_storage import NotesStore


def run_ingestion_pipeline(notes_dir: str | Path = config.NOTE_DIR) -> NotesStore:
    """
    Scans the notes directory, parses all supported files using module_ingest,
    and populates the NotesStore (ChromaDB + NetworkX) via module_storage.
    """
    # 1. Initialize the storage facade
    store = NotesStore(config=config)

    # 2. Run ingestion scan & parse
    print(f"Scanning and parsing notes from: {notes_dir}")
    notes_list = parser.scan_and_parse(notes_dir)

    # 3. Add the parsed note dicts (NoteItem shape) to the storage facade
    store.add(notes_list)

    print(f"Successfully indexed {len(store.all_notes())} notes into the vector database and knowledge graph.")
    return store


if __name__ == "__main__":
    run_ingestion_pipeline()
