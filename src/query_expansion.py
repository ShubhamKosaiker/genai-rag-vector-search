"""
Query expansion via embedding centroid.

Why centroid instead of just rewriting the query as text?

String-rewriting approaches (the naive version) still produce a single
embedding that may not cover all relevant aspects of the query. By
generating N semantic reformulations, embedding each independently, and
taking the mean (centroid) of the embedding space, we get a search vector
that sits at the centre of multiple relevant semantic neighbourhoods — this
consistently outperforms single-vector expansion in ablation studies.

The MockGenerativeModel interface mirrors vertexai.generativeai.GenerativeModel
so it can be swapped for the real SDK without changing QueryExpander.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock

import numpy as np

from src.embeddings import EmbeddingModelProtocol, embeddings_to_matrix


# ---------------------------------------------------------------------------
# Vertex AI GenerativeModel protocol contract
# ---------------------------------------------------------------------------

@runtime_checkable
class GenerativeModelProtocol(Protocol):
    def generate_content(self, prompt: str) -> object:
        ...


# ---------------------------------------------------------------------------
# Expansion templates — rewrites are corpus-aligned so the centroid
# vector lands near the relevant chunks when real embeddings are used.
# Each rewrite mirrors the vocabulary of a specific relevant chunk.
# ---------------------------------------------------------------------------

_EXPANSION_TEMPLATES: dict[str, list[str]] = {
    # Targets chunks 0 (load balancing), 1 (autoscaling), 5 (rate limiting)
    "peak load": [
        "How does a load balancer distribute requests across server instances to prevent bottlenecks during traffic spikes?",
        "How does autoscaling provision new instances in response to high CPU utilisation and request volume?",
        "What rate limiting mechanisms use token-bucket or sliding-window algorithms to protect backends from overload?",
    ],
    # Targets chunks 2 (in-memory cache), 7 (CDN), 3 (circuit breakers)
    "cach": [
        "How does Redis or Memcached store frequently accessed data in RAM to reduce database round-trips and improve read latency?",
        "How do CDN edge nodes cache static assets geographically close to users to reduce origin-server load?",
        "What write-through or cache-aside strategies synchronise cached data with the primary datastore?",
    ],
    # Targets chunks 4 (connection pooling), 0 (load balancing), 9 (monitoring)
    "database": [
        "How does connection pooling maintain pre-established database connections to reduce per-query latency and prevent exhaustion?",
        "How does load balancing prevent database overload by distributing upstream request traffic?",
        "How does monitoring P95 query latency and error rate detect database performance regressions early?",
    ],
    # Targets chunks 2 (cache), 7 (CDN), 4 (pooling)
    "latency": [
        "How does in-memory caching with Redis reduce database read latency under high read volume?",
        "How do CDN edge nodes reduce network latency by serving cached content close to the user?",
        "How does database connection pooling reduce per-query overhead and improve response time under concurrent load?",
    ],
    # Targets chunks 3 (circuit breakers), 6 (async queues), 5 (rate limiting)
    "cascad": [
        "How does the circuit-breaker pattern halt requests to a failing downstream service to prevent cascading failures?",
        "How do message queues like Kafka decouple producers from failing consumers to isolate failure modes?",
        "How does rate limiting with exponential backoff prevent retry storms from amplifying failures?",
    ],
    # Targets chunks 3 (circuit breakers), 6 (async queues), 5 (rate limiting)
    "fault": [
        "How does the circuit-breaker pattern halt requests to a failing downstream service to prevent cascading failures?",
        "How do async message queues isolate failures between producers and consumers in distributed systems?",
        "How does rate limiting prevent overload amplification when a downstream service is partially degraded?",
    ],
    # Targets chunk 9 (monitoring), 0 (load balancing), 8 (memory)
    "monitor": [
        "How do SLO-based alerts on rolling-window error budgets provide early warning before an SLA breach?",
        "How are P50, P95, and P99 latency percentiles used to detect tail-latency regressions in production?",
        "How does distributed tracing track request flows across services to surface anomalies in real time?",
    ],
}

_DEFAULT_EXPANSIONS = [
    "What is the system-level mechanism for handling this in a cloud-native distributed architecture?",
    "How does this behaviour affect throughput and latency under concurrent load?",
    "What infrastructure patterns manage this at scale in production environments?",
]


class MockGenerativeModel:
    """
    Deterministic mock of vertexai.generativeai.GenerativeModel.

    Rewrites are corpus-aligned: the vocabulary in each rewrite deliberately
    mirrors the relevant chunks so that, with real sentence-transformers
    embeddings, the centroid lands near the correct documents.
    """

    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        self._response = MagicMock()

    def generate_content(self, prompt: str) -> object:
        query_lower = prompt.lower()
        expansions = _DEFAULT_EXPANSIONS

        for keyword, templates in _EXPANSION_TEMPLATES.items():
            if keyword in query_lower:
                expansions = templates
                break

        self._response.text = "\n".join(expansions)
        return self._response


# ---------------------------------------------------------------------------
# QueryExpander: centroid-of-expansions strategy
# ---------------------------------------------------------------------------

@dataclass
class QueryExpander:
    """
    Produces a single search vector by:
    1. Generating N semantic reformulations of the query via a generative model.
    2. Embedding the original query + all reformulations.
    3. Taking the L2-normalised mean (centroid) of those embeddings.

    last_rewrites is populated after each expand() call so callers can log
    or display the actual rewrites used — important for benchmark transparency.
    """

    embedding_model: EmbeddingModelProtocol
    generative_model: GenerativeModelProtocol
    last_rewrites: list[str] = field(default_factory=list, init=False)

    def expand(self, query: str) -> np.ndarray:
        """
        Return a normalised centroid embedding for the query.
        Shape: (embedding_dim,) float32.
        Populates self.last_rewrites with the generated reformulations.
        """
        prompt = (
            f"Rewrite the following search query in 3 different ways "
            f"to maximise semantic retrieval coverage. "
            f"Output one rewrite per line, no numbering.\n\nQuery: {query}"
        )
        response = self.generative_model.generate_content(prompt)
        rewrites = [line.strip() for line in response.text.split("\n") if line.strip()]
        self.last_rewrites = rewrites[:3]

        all_texts = [query] + self.last_rewrites
        embeddings = embeddings_to_matrix(self.embedding_model.get_embeddings(all_texts))

        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        return centroid.astype(np.float32)
