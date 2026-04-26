import hashlib
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class HashEmbeddings(Embeddings):
    """Lightweight deterministic fallback embedding model."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str):
        return re.findall(r"\w+", (text or "").lower())

    def _embed(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)

        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimension
            vector[slot] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector.tolist()

    def embed_documents(self, texts: list[str]):
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str):
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embeddings_model():
    """Return preferred embedding model with safe fallback."""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"⚠️ HF embeddings failed, using HashEmbeddings: {e}")
        return HashEmbeddings()