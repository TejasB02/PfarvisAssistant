# =============================================================================
# module_storage/store.py - owner: Person 2 (Vector DB & Knowledge Graph)
# Receives : list[dict] note dicts from Person 1 (parser.scan_and_parse) via app.py.
# Hands off: NotesStore -> Person 3 (pipeline.ask calls hybrid_search/get_note)
#            and Person 4 (scheduler reads all_notes()).
# Wiring   : vector_db.VectorDBStore (Chroma + local embeddings, semantic) +
#            knowledge_graph.KnowledgeGraph (shared client / tag links).
# Contract : the README's NotesStore API: add(notes), query(text, k),
#            hybrid_search(query, k, hops) -> list[ContextItem], get_note,
#            all_notes, remove. Don't change these signatures.
# =============================================================================
"""NotesStore: the single public storage API used by the rest of the app."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import config as default_config
from contracts import ContextItem

from .knowledge_graph import KnowledgeGraph
from .vector_db import VectorDBStore

log = logging.getLogger(__name__)

# Graph neighbours inherit their parent's score times this factor, so they rank
# below the semantic hit that led to them.
GRAPH_SCORE_DECAY = 0.7


class NotesStore:
    def __init__(self, config=None, custom_embedder=None):
        """Open the local Chroma collection and an empty graph.

        config: the config module (defaults to the project's config.py).
        custom_embedder: object with .encode(texts, normalize_embeddings=True),
            used by tests instead of the real sentence-transformers model.
        """
        self.config = config or default_config
        self.vector_store = VectorDBStore(self.config, custom_model=custom_embedder)
        self.graph = KnowledgeGraph()
        self._notes_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # the watcher thread may add/remove while the UI searches

    def add(self, notes: list[dict]) -> None:
        """Add or update note dicts (NoteItem shape) in the vector DB, graph and cache.

        Notes without an id or with empty content are skipped. After a batch,
        Chroma rows for notes this store does not know (deleted files from an
        earlier run) are removed, so they can't come back as search hits.
        """
        with self._lock:
            for note in notes or []:
                note_id = (note or {}).get("id")
                content = (note or {}).get("content") or ""
                if not note_id or not content.strip():
                    continue
                if note_id in self._notes_cache:
                    self.graph.remove_note(note_id)  # rebuild its edges from the new version
                self._notes_cache[note_id] = dict(note)
                metadata = {k: v for k, v in note.items() if k not in ("id", "content")}
                self.vector_store.upsert(note_id, content, metadata)
                self.graph.add_note(note_id, metadata, self._notes_cache)
            self._prune_stale_vectors()

    def remove(self, note_id: str) -> None:
        """Remove a note from the cache, the vector DB and the graph (no-op if unknown)."""
        with self._lock:
            self._notes_cache.pop(note_id, None)
            self.vector_store.delete(note_id)
            self.graph.remove_note(note_id)

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of one note dict, or None."""
        with self._lock:
            note = self._notes_cache.get(note_id)
            return dict(note) if note is not None else None

    def all_notes(self) -> List[Dict[str, Any]]:
        """Return a snapshot list of every stored note dict (any order)."""
        with self._lock:
            return [dict(n) for n in self._notes_cache.values()]

    def query(self, text: str, k: int = 5) -> list[ContextItem]:
        """Thin alias of hybrid_search with the configured graph hops."""
        return self.hybrid_search(text, k=k, hops=self.config.GRAPH_HOPS)

    def hybrid_search(self, query: str, k: int = 5, hops: int = 1) -> list[ContextItem]:
        """Semantic hits first, then graph neighbours of those hits within ``hops``.

        Output: list[ContextItem]. Semantic hits have via="semantic" and their
        cosine similarity (0..1); graph neighbours have via="graph" and their
        parent's score times GRAPH_SCORE_DECAY. De-duplicated by note id.
        Blank query or empty store -> [].
        """
        if not query or not query.strip():
            return []
        with self._lock:
            if not self._notes_cache:
                return []
            try:
                hits = self.vector_store.query(query, k=k)
            except Exception:  # noqa: BLE001 - a broken index must not crash the chat
                log.exception("Vector search failed")
                return []

            results: list[ContextItem] = []
            scores: Dict[str, float] = {}
            for hit in hits:
                note = self._notes_cache.get(hit["id"])
                if note is None or hit["id"] in scores:
                    continue
                scores[hit["id"]] = hit["score"]
                results.append(_to_context(note, hit["score"], "semantic"))

            if hops > 0 and scores:
                for neighbour in self.graph.get_neighbors(list(scores), hops=hops):
                    note = self._notes_cache.get(neighbour["id"])
                    if note is None or neighbour["id"] in scores:
                        continue
                    score = scores.get(neighbour["parent_id"], 0.0) * GRAPH_SCORE_DECAY
                    scores[neighbour["id"]] = score
                    results.append(_to_context(note, score, "graph"))
            return results

    def _prune_stale_vectors(self) -> None:
        try:
            stored = self.vector_store.collection.get(include=[])["ids"]
        except Exception:  # noqa: BLE001
            return
        for note_id in stored:
            if note_id not in self._notes_cache:
                self.vector_store.delete(note_id)


def _to_context(note: dict, score: float, via: str) -> ContextItem:
    return ContextItem(
        filepath=note["filepath"],
        title=note.get("title") or "",
        content=note.get("content") or "",
        score=max(0.0, min(1.0, float(score))),
        via=via,
    )
