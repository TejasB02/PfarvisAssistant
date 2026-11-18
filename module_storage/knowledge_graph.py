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

from typing import Dict, Any, List

class KnowledgeGraph:
    def __init__(self):
        import networkx as nx
        self.graph = nx.Graph()

    def add_note(self, note_id: str, metadata: Dict[str, Any], notes_store: Dict[str, Dict[str, Any]]):
        """Adds a note node and builds edges with other notes sharing clients or tags."""
        self.graph.add_node(note_id, **metadata)
        
        client = metadata.get("client")
        tags_raw = metadata.get("tags", [])
        tags = [t.strip() for t in tags_raw.split(",")] if isinstance(tags_raw, str) else tags_raw

        # Compare with all other loaded notes to establish relational edges
        for other_id, other_data in notes_store.items():
            if other_id == note_id:
                continue
            
            other_client = other_data.get("client")
            other_tags_raw = other_data.get("tags", [])
            other_tags = [t.strip() for t in other_tags_raw.split(",")] if isinstance(other_tags_raw, str) else other_tags_raw

            # Edge condition 1: Shared client
            if client and other_client and client == other_client:
                self.graph.add_edge(note_id, other_id, reason=f"client:{client}")

            # Edge condition 2: Shared tags
            shared_tags = set(tags).intersection(set(other_tags))
            for tag in shared_tags:
                if tag:
                    self.graph.add_edge(note_id, other_id, reason=f"tag:{tag}")

    def remove_note(self, note_id: str):
        """Removes a note node from the graph."""
        if self.graph.has_node(note_id):
            self.graph.remove_node(note_id)

    def get_neighbors(self, seed_ids: List[str], hops: int) -> List[Dict[str, Any]]:
        """Traverses the graph up to `hops` away from seed notes, returning neighbor metadata and reasons."""
        current_layer = set(seed_ids)
        visited = set(seed_ids)
        neighbors_meta = []

        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                if self.graph.has_node(node):
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_layer.add(neighbor)
                            
                            # Retrieve edge reason attribute
                            edge_data = self.graph.get_edge_data(node, neighbor)
                            reason = edge_data.get("reason", "linked") if edge_data else "linked"
                            
                            neighbors_meta.append({
                                "id": neighbor,
                                "parent_id": node,
                                "reason": reason
                            })
            current_layer = next_layer
        return neighbors_meta