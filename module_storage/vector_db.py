# =============================================================================
# module_storage/vector_db.py - owner: Person 2 (Vector DB & Knowledge Graph)
# Receives : note dicts from store.NotesStore.add; query strings from
#            store.NotesStore.hybrid_search.
# Hands off: (note_id, similarity) pairs back to store.NotesStore.
# Engine   : chromadb.PersistentClient(path=config.CHROMA_DIR,
#            settings=Settings(anonymized_telemetry=False)) and
#            sentence-transformers loaded from config.EMBED_MODEL_PATH (local
#            folder). We ALWAYS pass our own embeddings; Chroma's default
#            embedding function must never be used (it downloads a model).
# STATUS   : STUB - in-memory list of ids, mock similarity scores.
# =============================================================================
"""ChromaDB + local embeddings, hidden behind NotesStore."""

from __future__ import annotations

import logging
from typing import Any, Optional

import config
from contracts import note_from_dict, note_to_dict

log = logging.getLogger(__name__)


class VectorDB:
    """Embeds notes locally and stores them in a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str = config.CHROMA_DIR,
        model_path: str = config.EMBED_MODEL_PATH,
    ) -> None:
        """Remember settings; open Chroma and the model lazily on first use.

        Edge cases:
            - model_path missing -> raise a clear error telling the user to run
              scripts/prefetch_models.py (never fall back to a hub download).
        """
        # TODO(person 2): open lazily in _collection() / _embedder():
        #   import chromadb
        #   from chromadb.config import Settings
        #   client = chromadb.PersistentClient(path=self.persist_dir,
        #                                      settings=Settings(anonymized_telemetry=False))
        #   self._coll = client.get_or_create_collection(
        #       name=config.CHROMA_COLLECTION, embedding_function=None,
        #       metadata={"hnsw:space": "cosine"})
        #   from sentence_transformers import SentenceTransformer
        #   self._model = SentenceTransformer(self.model_path, device="cpu")
        # Verify on your installed chromadb version that no model download is
        # triggered (run once with Wi-Fi off).
        self.persist_dir = persist_dir
        self.model_path = model_path
        self._mock_ids: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised embedding per text (384 floats for MiniLM).

        Edge cases: empty list -> []; empty strings still get a vector.
        """
        # TODO(person 2): self._embedder().encode(texts, normalize_embeddings=True).tolist()
        return [[0.0] * 384 for _ in texts]

    def upsert(self, notes: list[dict]) -> None:
        """Embed ``title + content`` of each note and upsert it into Chroma.

        Uses ``ids=[note["id"]]``, ``documents=[content]``,
        ``embeddings=self.embed(...)`` and ``metadatas=[note_to_metadata(note)]``.
        Edge cases: empty list -> no-op; re-adding the same id overwrites it.
        """
        # TODO(person 2): real Chroma upsert with our own embeddings.
        for note in notes:
            if note["id"] not in self._mock_ids:
                self._mock_ids.append(note["id"])

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to ``k`` (note_id, similarity) pairs, best first.

        Use ``query_embeddings=self.embed([query])`` (never ``query_texts``,
        which would invoke an embedding function). With cosine space,
        similarity = 1 - distance, clamped to 0..1.
        Edge cases: empty collection or blank query -> [].
        """
        # TODO(person 2): real Chroma query. Mock: first k ids, decreasing scores.
        if not query.strip():
            return []
        return [(note_id, round(max(0.0, 0.9 - 0.05 * i), 2))
                for i, note_id in enumerate(self._mock_ids[:k])]

    def delete(self, note_id: str) -> None:
        """Remove a note's vector. Unknown id -> no-op."""
        # TODO(person 2): self._collection().delete(ids=[note_id])
        if note_id in self._mock_ids:
            self._mock_ids.remove(note_id)


def note_to_metadata(note: dict) -> dict[str, Any]:
    """Flatten a note dict into Chroma-safe scalar metadata.

    Datetimes -> ISO strings, tags -> comma-joined string, None -> "".
    ``content`` is stored as the Chroma document, not in metadata.
    """
    # TODO(person 2): decide how to handle commas inside tags (extract.py
    # produces [\w-] tags, so none are expected today).
    def iso(value: Any) -> str:
        return value.isoformat() if value is not None else ""

    return {
        "title": note["title"],
        "filepath": note["filepath"],
        "timestamp": iso(note.get("timestamp")),
        "client": note.get("client") or "",
        "tags": ",".join(note.get("tags") or []),
        "event_time": iso(note.get("event_time")),
    }


def metadata_to_note(note_id: str, document: str, metadata: dict[str, Any]) -> dict:
    """Inverse of note_to_metadata: rebuild the contract note dict."""
    # TODO(person 2): keep in sync with note_to_metadata.
    tags_raw: Optional[str] = metadata.get("tags") or ""
    return note_to_dict(note_from_dict({
        "id": note_id,
        "title": metadata.get("title", ""),
        "content": document or "",
        "filepath": metadata.get("filepath", ""),
        "timestamp": metadata.get("timestamp") or None,
        "client": metadata.get("client") or None,
        "tags": [t for t in tags_raw.split(",") if t],
        "event_time": metadata.get("event_time") or None,
    }))
