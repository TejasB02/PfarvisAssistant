# =============================================================================
# module_storage/knowledge_graph.py - owner: Person 2 (Vector DB & Knowledge Graph)
# Receives : note dicts from store.NotesStore.add.
# Hands off: neighbouring note ids (related notes) back to
#            store.NotesStore.hybrid_search, which turns them into
#            ContextItems with via="graph".
# Engine   : networkx (in memory; rebuild from all_notes() on startup).
# STATUS   : STUB - remembers node ids, returns no neighbours.
# =============================================================================
"""Links related notes: same client or any shared tag."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


class KnowledgeGraph:
    """Undirected graph: one node per note id, edges between related notes.

    Edge rule: two notes are linked when they share the same (non-None) client
    or ANY tag. Every edge carries a "reason" attribute, e.g.
    "client:Quillmont Systems" or "tag:q3-rollout" (joined with "; " when
    several reasons apply).
    """

    def __init__(self) -> None:
        # TODO(person 2): import networkx inside __init__ and create nx.Graph().
        self._node_ids: set[str] = set()

    def add_notes(self, notes: list[dict]) -> None:
        """Add/refresh nodes for ``notes`` and (re)compute their edges.

        Store client and tags as node attributes. Re-adding a note replaces its
        old edges (its tags may have changed).
        Edge cases: empty list -> no-op; notes with no client and no tags become
        isolated nodes.
        """
        # TODO(person 2): build nodes and "reason"-labelled edges.
        self._node_ids.update(note["id"] for note in notes)

    def neighbours(self, note_id: str, hops: int = 1) -> list[str]:
        """Return ids within ``hops`` edges of ``note_id`` (excluding itself).

        Order: closest first, then by number of shared reasons.
        Edge cases: unknown id or hops <= 0 -> []. Consider capping the result
        (e.g. 5) so one popular tag does not flood the LLM context.
        """
        # TODO(person 2): nx.single_source_shortest_path_length(G, note_id, cutoff=hops)
        return []

    def edge_reason(self, a: str, b: str) -> Optional[str]:
        """Return the "reason" attribute of edge a-b, or None if not linked."""
        # TODO(person 2): G.edges[a, b]["reason"] if G.has_edge(a, b) else None
        return None

    def remove(self, note_id: str) -> None:
        """Remove a node and its edges. Unknown id -> no-op."""
        # TODO(person 2): G.remove_node(note_id) if present
        self._node_ids.discard(note_id)
