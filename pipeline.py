from pathlib import Path
import config
from module_storage import NotesStore
from module_ingest import parser

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

    # 3. Add each parsed note into the storage facade
    for note in notes_list:
        metadata = {
            "title": note["title"],
            "filepath": note["filepath"],
            "timestamp": note["timestamp"].isoformat() if note["timestamp"] else "",
            "client": note["client"] or "",
            "tags": note["tags"],
            "event_time": note["event_time"].isoformat() if note["event_time"] else ""
        }

        store.add(
            note_id=note["id"],
            content=note["content"],
            metadata=metadata
        )

    print(f"Successfully indexed {len(notes_list)} notes into the vector database and knowledge graph.")
    return store

if __name__ == "__main__":
    run_ingestion_pipeline()