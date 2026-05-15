"""
RAGPipeline: orchestrates ingestion and both retrieval strategies.

Strategy A — Raw Vector Search:
    embed(query) → nearest-neighbour search

Strategy B — AI-Enhanced Retrieval:
    expand(query) → centroid of embeddings of [query, rewrite1, rewrite2, rewrite3]
    → nearest-neighbour search on the centroid vector

The pipeline accepts injected dependencies (embedding_model, generative_model)
so it runs identically against local mocks and production Vertex AI clients.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.embeddings import EmbeddingModelProtocol, LocalEmbeddingModel, embeddings_to_matrix
from src.query_expansion import GenerativeModelProtocol, MockGenerativeModel, QueryExpander
from src.vector_store import FAISSVectorStore, SearchResult


@dataclass
class RetrievalResult:
    strategy: str                   # "A" or "B"
    query: str
    expanded_query_used: bool
    results: list[SearchResult]
    latency_ms: float
    rewrites: list[str] = field(default_factory=list)  # Strategy B only


@dataclass
class RAGPipeline:
    """
    Manages corpus ingestion and dual-strategy retrieval.

    Parameters
    ----------
    embedding_model : EmbeddingModelProtocol
        Any model with a get_embeddings(texts) → list[TextEmbedding] interface.
        Defaults to the local sentence-transformers model.
    generative_model : GenerativeModelProtocol
        Any model with generate_content(prompt) → response interface.
        Defaults to the deterministic mock.
    embedding_dim : int
        Must match the output dimension of embedding_model.
    """

    embedding_model: EmbeddingModelProtocol = field(
        default_factory=LocalEmbeddingModel
    )
    generative_model: GenerativeModelProtocol = field(
        default_factory=MockGenerativeModel
    )
    embedding_dim: int = 384

    _store: FAISSVectorStore = field(init=False)
    _expander: QueryExpander = field(init=False)
    _ingested: bool = field(default=False, init=False)

    def __post_init__(self):
        self._store = FAISSVectorStore(embedding_dim=self.embedding_dim)
        self._expander = QueryExpander(
            embedding_model=self.embedding_model,
            generative_model=self.generative_model,
        )

    def ingest(self, chunks: list[str]) -> None:
        """
        Embed and index a list of text chunks.
        Can be called incrementally; chunks are appended to the index.
        """
        if not chunks:
            return

        embeddings_objs = self.embedding_model.get_embeddings(chunks)
        matrix = embeddings_to_matrix(embeddings_objs)
        self._store.add(chunks, matrix)
        self._ingested = True

    def retrieve_strategy_a(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Strategy A: direct embedding similarity search."""
        self._require_ingestion()

        t0 = time.perf_counter()
        query_emb = embeddings_to_matrix(
            self.embedding_model.get_embeddings([query])
        )[0]
        results = self._store.search(query_emb, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            strategy="A",
            query=query,
            expanded_query_used=False,
            results=results,
            latency_ms=latency_ms,
        )

    def retrieve_strategy_b(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Strategy B: query expansion → centroid embedding → similarity search."""
        self._require_ingestion()

        t0 = time.perf_counter()
        centroid = self._expander.expand(query)
        results = self._store.search(centroid, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            strategy="B",
            query=query,
            expanded_query_used=True,
            results=results,
            latency_ms=latency_ms,
            rewrites=list(self._expander.last_rewrites),
        )

    def _require_ingestion(self) -> None:
        if not self._ingested:
            raise RuntimeError("Pipeline has no indexed documents. Call ingest() first.")
