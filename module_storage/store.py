# =============================================================================
# module_storage/store.py - owner: Person 2 (Vector DB & Knowledge Graph)
# Receives : list[dict] note dicts from Person 1 (parser.scan_and_parse) via app.py.
# Hands off: NotesStore -> Person 3 (pipeline.ask calls hybrid_search/get_note)
#            and Person 4 (scheduler reads all_notes()).
# Wiring   : vector_db.VectorDB (semantic) + knowledge_graph.KnowledgeGraph (links).
# STATUS   : STUB - in-memory notes; mock vector scores; graph returns nothing.
# =============================================================================
"""NotesStore: the single public storage API used by the rest of the app."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import config
from contracts import ContextItem
from module_storage.knowledge_graph import KnowledgeGraph
from module_storage.vector_db import VectorDB

log = logging.getLogger(__name__)

# Graph neighbours inherit their parent's score times this factor, so they rank
# below direct semantic hits and never decide has_enough_context on their own.
GRAPH_SCORE_DECAY = 0.8


class NotesStore:
    """Hybrid store: semantic search in Chroma + related notes from the graph.

    Thread-safety: the reminder scheduler calls all_notes() from a background
    thread, so reads and writes of the in-memory note map hold a lock.
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_DIR,
        model_path: str = config.EMBED_MODEL_PATH,
    ) -> None:
        # TODO(person 2): on startup, decide whether to reload notes from Chroma
        # (persisted) or rely on app.py re-adding them every launch.
        self._notes: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._vectors = VectorDB(persist_dir=persist_dir, model_path=model_path)
        self._graph = KnowledgeGraph()

    def add(self, notes: list[dict]) -> None:
        """Embed and upsert notes into Chroma, add graph nodes and edges.

        Inputs:
            notes: note dicts (asdict(NoteItem)) from Person 1.
        Output: None.
        Behaviour:
            - Upsert by ``id`` (re-adding a file replaces the old version).
            - Edges between notes sharing the same client or any tag, with an
              edge attribute "reason".
        Edge cases: empty list -> no-op; a dict missing contract keys -> log and
        skip that note rather than failing the whole batch.
        """
        # TODO(person 2): validate each dict, batch the embeddings.
        if not notes:
            return
        with self._lock:
            for note in notes:
                self._notes[note["id"]] = dict(note)
        self._vectors.upsert(notes)
        self._graph.add_notes(notes)

    def query(self, text: str, k: int = 5) -> list[ContextItem]:
        """Alias for ``hybrid_search(text, k=k, hops=config.GRAPH_HOPS)``."""
        # TODO(person 2): keep as a thin alias.
        return self.hybrid_search(text, k=k, hops=config.GRAPH_HOPS)

    def hybrid_search(self, query: str, k: int = 5, hops: int = 1) -> list[ContextItem]:
        """Semantic matches plus graph neighbours of those matches.

        Inputs:
            query: standalone question (already rewritten by Person 3).
            k: number of semantic hits.
            hops: graph distance for neighbours (0 = semantic only).
        Output:
            list[ContextItem], de-duplicated by note, semantic results first
            (best score first), then graph neighbours (score =
            parent score * GRAPH_SCORE_DECAY, via="graph").
        Edge cases: blank query or empty store -> []. Scores are 0..1.
        """
        # TODO(person 2): this orchestration is fine to keep; make VectorDB.search
        # and KnowledgeGraph.neighbours real underneath it.
        if not query.strip():
            return []
        hits = self._vectors.search(query, k=k)
        results: list[ContextItem] = []
        seen: set[str] = set()
        for note_id, score in hits:
            note = self.get_note(note_id)
            if note is None or note_id in seen:
                continue
            results.append(_to_context(note, score, "semantic"))
            seen.add(note_id)
        if hops > 0:
            for note_id, score in hits:
                for neighbour_id in self._graph.neighbours(note_id, hops=hops):
                    neighbour = self.get_note(neighbour_id)
                    if neighbour is None or neighbour_id in seen:
                        continue
                    results.append(_to_context(neighbour, score * GRAPH_SCORE_DECAY, "graph"))
                    seen.add(neighbour_id)
        return results

    def get_note(self, note_id: str) -> Optional[dict]:
        """Return the note dict for ``note_id``, or None if unknown."""
        # TODO(person 2): fall back to Chroma (metadata_to_note) if not cached.
        with self._lock:
            note = self._notes.get(note_id)
            return dict(note) if note is not None else None

    def all_notes(self) -> list[dict]:
        """Return a snapshot list of every stored note dict (any order)."""
        # TODO(person 2): keep this cheap; the scheduler calls it every 30 s.
        with self._lock:
            return [dict(n) for n in self._notes.values()]

    def remove(self, note_id: str) -> None:
        """Delete a note from Chroma, the graph and the cache. Unknown id -> no-op."""
        # TODO(person 2): make sure Chroma and the graph stay in sync.
        with self._lock:
            self._notes.pop(note_id, None)
        self._vectors.delete(note_id)
        self._graph.remove(note_id)


def _to_context(note: dict, score: float, via: str) -> ContextItem:
    return ContextItem(
        filepath=note["filepath"],
        title=note["title"],
        content=note["content"],
        score=max(0.0, min(1.0, float(score))),
        via=via,  # type: ignore[typeddict-item]
    )
