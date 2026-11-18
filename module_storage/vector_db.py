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

import os
from typing import List, Dict, Any, Optional

def get_chroma_client(config):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=config.CHROMA_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )

def get_embedding_model(config):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL_PATH)

def note_to_metadata(note: Dict[str, Any]) -> Dict[str, Any]:
    """Converts note fields into Chroma-compatible scalar metadata."""
    metadata = {}
    for key, val in note.items():
        if key == "content":
            continue  # Content is stored in 'documents'
        if isinstance(val, list):
            # Store lists/tags as comma-joined strings for scalar compatibility
            metadata[key] = ",".join(str(v) for v in val) if val else ""
        elif val is not None:
            metadata[key] = str(val)
    return metadata

def metadata_to_note(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Restores scalar metadata back into structured note fields."""
    note = dict(metadata)
    if "tags" in note and isinstance(note["tags"], str):
        note["tags"] = [t.strip() for t in note["tags"].split(",") if t.strip()]
    return note

class VectorDBStore:
    def __init__(self, config, custom_model=None):
        self.config = config
        self.client = get_chroma_client(config)
        self.model = custom_model if custom_model else get_embedding_model(config)
        
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, note_id: str, content: str, note_data: Dict[str, Any]):
        """Encodes note content locally and upserts vectors into ChromaDB."""
        embedding = self.model.encode([content], normalize_embeddings=True)[0].tolist()
        metadata = note_to_metadata(note_data)
        metadata["note_id"] = note_id

        self.collection.upsert(
            ids=[note_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def delete(self, note_id: str):
        """Removes note embeddings from ChromaDB."""
        try:
            self.collection.delete(ids=[note_id])
        except Exception:
            pass

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Queries the vector database using explicit query embeddings with clamped similarity scores."""
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted = []
        if results and results['ids'] and len(results['ids'][0]) > 0:
            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(ids)
            
            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 0.0
                # Similarity = 1 - cosine distance, clamped to 0..1
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                formatted.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": similarity,
                    "via": "vector"
                })
        return formatted