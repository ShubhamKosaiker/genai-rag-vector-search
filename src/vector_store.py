"""
FAISS vector store with cosine similarity.

Design decision: vectors are L2-normalised at ingestion time so that
inner-product search (IndexFlatIP) is equivalent to cosine similarity.
This avoids the Euclidean-distance mistake most RAG implementations make —
cosine similarity is the correct metric for semantic search because it
measures angular distance, not magnitude.

Production migration note: swap IndexFlatIP for IndexIVFFlat or
IndexHNSWFlat when corpus > 100k chunks. For Vertex AI Vector Search
(Matching Engine), this class maps 1-to-1 to a deployed index endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    score: float          # cosine similarity ∈ [-1, 1]; higher = more similar


@dataclass
class FAISSVectorStore:
    embedding_dim: int
    _index: faiss.IndexFlatIP = field(init=False)
    _chunks: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._index = faiss.IndexFlatIP(self.embedding_dim)

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """
        Ingest text chunks and their embeddings.

        embeddings must be shape (n, embedding_dim) float32, already L2-normalised.
        If not normalised, normalisation is applied here as a safety net.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings shape (n, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = (embeddings / norms).astype(np.float32)

        self._index.add(normalised)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        """
        Return top_k most similar chunks for a query embedding.

        query_embedding: shape (embedding_dim,) or (1, embedding_dim), float32.
        """
        if self._index.ntotal == 0:
            return []

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, top_k)

        return [
            SearchResult(
                chunk_id=int(idx),
                text=self._chunks[int(idx)],
                score=float(score),
            )
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    @property
    def size(self) -> int:
        return self._index.ntotal
