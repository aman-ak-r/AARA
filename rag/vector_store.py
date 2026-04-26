import os
import uuid
import hashlib

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from rag.embeddings import get_embeddings_model

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aara-research")

# Embedding dimension for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION = 384

_pc_client = None
_pc_index = None


def _get_pinecone_index():
    """Lazily initialise the Pinecone client and return the index object.

    Creates the index if it does not already exist.
    """
    global _pc_client, _pc_index

    if _pc_index is not None:
        return _pc_index

    _pc_client = Pinecone(api_key=PINECONE_API_KEY)

    # Check existing indexes
    existing_indexes = [idx.name for idx in _pc_client.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        _pc_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _pc_index = _pc_client.Index(PINECONE_INDEX_NAME)
    return _pc_index


def _chunk_id(text: str) -> str:
    """Generate a deterministic ID for a text chunk to avoid duplicates."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def create_vector_store(text_chunks):
    """Embeds text chunks and upserts them into Pinecone.

    Args:
        text_chunks (list): A list of text chunks extracted from PDFs.

    Returns:
        The Pinecone index object (acts as the 'vector_store' handle).
    """
    embeddings_model = get_embeddings_model()
    index = _get_pinecone_index()

    # Batch upsert – Pinecone recommends batches of ~100
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        vectors = embeddings_model.embed_documents(batch)

        upsert_data = [
            (
                _chunk_id(chunk),           # unique vector id
                vector,                     # embedding list[float]
                {"text": chunk},            # metadata payload
            )
            for chunk, vector in zip(batch, vectors)
        ]
        index.upsert(vectors=upsert_data)

    return index


def delete_all_vectors():
    """Delete all vectors from the Pinecone index (useful for a fresh start)."""
    index = _get_pinecone_index()
    try:
        index.delete(delete_all=True)
    except Exception:
        # Index may be empty / namespace doesn't exist yet — that's fine
        pass
    return index

