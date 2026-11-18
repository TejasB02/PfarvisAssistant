# =============================================================================
# module_storage/store.py - owner: Person 2 (Vector DB & Knowledge Graph)
# Receives : list[dict] note dicts from Person 1 (parser.scan_and_parse) via app.py.
# Hands off: NotesStore -> Person 3 (pipeline.ask calls hybrid_search/get_note)
#            and Person 4 (scheduler reads all_notes()).
# Wiring   : vector_db.VectorDB (semantic) + knowledge_graph.KnowledgeGraph (links).
# STATUS   : STUB - in-memory notes; mock vector scores; graph returns nothing.
# =============================================================================
"""NotesStore: the single public storage API used by the rest of the app."""

from typing import List, Dict, Any, Optional
from .vector_db import VectorDBStore
from .knowledge_graph import KnowledgeGraph

class NotesStore:
    def __init__(self, config, custom_embedder=None):
        self.config = config
        self.vector_store = VectorDBStore(config, custom_model=custom_embedder)
        self.graph = KnowledgeGraph()
        self._notes_cache: Dict[str, Dict[str, Any]] = {}

    def add(self, note_id: str, content: str, metadata: Dict[str, Any]):
        """Adds or updates a note across vector DB, graph, and local cache."""
        full_record = {"id": note_id, "content": content, **metadata}
        self._notes_cache[note_id] = full_record
        
        # Upsert into vector database
        self.vector_store.upsert(note_id, content, metadata)
        
        # Update knowledge graph relationships
        self.graph.add_note(note_id, metadata, self._notes_cache)

    def remove(self, note_id: str):
        """Removes a note from the store components."""
        if note_id in self._notes_cache:
            del self._notes_cache[note_id]
        self.vector_store.delete(note_id)
        self.graph.remove_node(note_id)

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single note record by ID."""
        return self._notes_cache.get(note_id)

    def all_notes(self) -> List[Dict[str, Any]]:
        """Returns all stored notes."""
        return list(self._notes_cache.values())

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs pure semantic vector search."""
        return self.vector_store.query(query_text, k=k)

    def hybrid_search(self, query: str, k: int = 5, hops: int = 1) -> List[Dict[str, Any]]:
        """
        Combines vector similarity and graph traversal:
        Returns semantic hits first, followed by graph neighbors within hops,
        scored by parent score times GRAPH_SCORE_DECAY.
        """
        # 1. Get initial semantic hits
        semantic_hits = self.vector_store.query(query, k=k)
        
        results = list(semantic_hits)
        seen_ids = {hit["id"] for hit in semantic_hits}
        seed_ids = list(seen_ids)

        # 2. Get graph neighbors within hops
        graph_neighbors = self.graph.get_neighbors(seed_ids, hops=hops)
        
        decay = getattr(self.config, "GRAPH_SCORE_DECAY", 0.7)
        score_map = {hit["id"]: hit["score"] for hit in semantic_hits}

        for neighbor in graph_neighbors:
            n_id = neighbor["id"]
            if n_id not in seen_ids and n_id in self._notes_cache:
                seen_ids.add(n_id)
                parent_id = neighbor["parent_id"]
                parent_score = score_map.get(parent_id, 0.5)
                decayed_score = parent_score * decay

                note_record = self._notes_cache[n_id]
                results.append({
                    "id": n_id,
                    "content": note_record.get("content", ""),
                    "metadata": {k: v for k, v in note_record.items() if k != "content"},
                    "score": decayed_score,
                    "via": "graph",
                    "reason": neighbor["reason"]
                })

        return results