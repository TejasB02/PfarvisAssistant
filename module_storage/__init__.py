# =============================================================================
# module_storage - Person 2: Vector DB & Knowledge Graph
# Receives : list[dict] note dicts from Person 1 (parser.scan_and_parse).
# Hands off: a NotesStore whose hybrid_search() returns list[ContextItem]
#            -> Person 3 (module_rag.pipeline.ask).
# Files    : store.py (the public NotesStore facade), vector_db.py (ChromaDB +
#            local sentence-transformers), knowledge_graph.py (NetworkX links).
# =============================================================================
"""Storage package: local vector search plus a knowledge graph of related notes."""

from .store import NotesStore

__all__ = ["NotesStore"]