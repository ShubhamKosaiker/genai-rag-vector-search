"""
Tests for the RAGPipeline and query expansion logic.

All tests use the mock embedding and generative models — no sentence-transformers
or GCP SDK calls, so the suite runs offline and in < 2 seconds.
"""

import numpy as np
import pytest

from src.embeddings import MockVertexEmbeddingModel
from src.query_expansion import MockGenerativeModel, QueryExpander
from src.retrieval import RAGPipeline
from src.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_model():
    return MockVertexEmbeddingModel()


@pytest.fixture
def mock_generative_model():
    return MockGenerativeModel()


@pytest.fixture
def pipeline(mock_embedding_model, mock_generative_model):
    return RAGPipeline(
        embedding_model=mock_embedding_model,
        generative_model=mock_generative_model,
        embedding_dim=MockVertexEmbeddingModel.EMBEDDING_DIM,
    )


SAMPLE_CHUNKS = [
    "Load balancing distributes requests to prevent bottlenecks during peak traffic.",
    "Horizontal autoscaling provisions new instances under heavy CPU utilisation.",
    "In-memory caching with Redis reduces database read latency significantly.",
    "Circuit breakers prevent cascading failures by halting calls to failing services.",
    "Database connection pooling avoids costly connection setup per query.",
]


# ---------------------------------------------------------------------------
# FAISSVectorStore tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def test_add_and_size(self, mock_embedding_model):
        store = FAISSVectorStore(embedding_dim=384)
        from src.embeddings import embeddings_to_matrix
        matrix = embeddings_to_matrix(mock_embedding_model.get_embeddings(SAMPLE_CHUNKS))
        store.add(SAMPLE_CHUNKS, matrix)
        assert store.size == len(SAMPLE_CHUNKS)

    def test_search_returns_top_k(self, mock_embedding_model):
        store = FAISSVectorStore(embedding_dim=384)
        from src.embeddings import embeddings_to_matrix
        matrix = embeddings_to_matrix(mock_embedding_model.get_embeddings(SAMPLE_CHUNKS))
        store.add(SAMPLE_CHUNKS, matrix)

        query_emb = embeddings_to_matrix(mock_embedding_model.get_embeddings(["peak load"]))[0]
        results = store.search(query_emb, top_k=3)
        assert len(results) == 3

    def test_scores_are_in_valid_cosine_range(self, mock_embedding_model):
        store = FAISSVectorStore(embedding_dim=384)
        from src.embeddings import embeddings_to_matrix
        matrix = embeddings_to_matrix(mock_embedding_model.get_embeddings(SAMPLE_CHUNKS))
        store.add(SAMPLE_CHUNKS, matrix)

        query_emb = embeddings_to_matrix(mock_embedding_model.get_embeddings(["caching"]))[0]
        results = store.search(query_emb, top_k=3)
        for r in results:
            assert -1.01 <= r.score <= 1.01, f"Cosine score out of range: {r.score}"

    def test_search_on_empty_store_returns_empty(self):
        store = FAISSVectorStore(embedding_dim=384)
        dummy = np.random.rand(384).astype(np.float32)
        assert store.search(dummy, top_k=3) == []

    def test_wrong_embedding_dim_raises(self):
        store = FAISSVectorStore(embedding_dim=384)
        bad_matrix = np.random.rand(3, 128).astype(np.float32)
        with pytest.raises(ValueError, match="Expected embeddings shape"):
            store.add(["a", "b", "c"], bad_matrix)


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def test_ingest_without_error(self, pipeline):
        pipeline.ingest(SAMPLE_CHUNKS)
        assert pipeline._store.size == len(SAMPLE_CHUNKS)

    def test_retrieve_before_ingest_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="Call ingest"):
            pipeline.retrieve_strategy_a("test query")

    def test_strategy_a_returns_correct_top_k(self, pipeline):
        pipeline.ingest(SAMPLE_CHUNKS)
        result = pipeline.retrieve_strategy_a("peak load handling", top_k=3)
        assert result.strategy == "A"
        assert len(result.results) == 3
        assert result.expanded_query_used is False
        assert result.latency_ms >= 0

    def test_strategy_b_returns_correct_top_k(self, pipeline):
        pipeline.ingest(SAMPLE_CHUNKS)
        result = pipeline.retrieve_strategy_b("peak load handling", top_k=3)
        assert result.strategy == "B"
        assert len(result.results) == 3
        assert result.expanded_query_used is True
        assert result.latency_ms >= 0

    def test_strategy_b_uses_different_embedding_than_a(self, pipeline):
        pipeline.ingest(SAMPLE_CHUNKS)
        query = "database performance under load"
        res_a = pipeline.retrieve_strategy_a(query, top_k=3)
        res_b = pipeline.retrieve_strategy_b(query, top_k=3)
        # With a semantic corpus and real expansions the top chunk may differ
        # At minimum the scores should differ due to the centroid shift
        scores_a = [r.score for r in res_a.results]
        scores_b = [r.score for r in res_b.results]
        # They won't be identical because the centroid shifts the search vector
        assert scores_a != scores_b or res_a.results[0].chunk_id != res_b.results[0].chunk_id

    def test_empty_ingest_is_safe(self, pipeline):
        pipeline.ingest([])
        with pytest.raises(RuntimeError):
            pipeline.retrieve_strategy_a("anything")


# ---------------------------------------------------------------------------
# QueryExpander tests
# ---------------------------------------------------------------------------

class TestQueryExpander:
    def test_returns_normalised_vector(self, mock_embedding_model, mock_generative_model):
        expander = QueryExpander(mock_embedding_model, mock_generative_model)
        vec = expander.expand("How does the system handle peak load?")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_output_shape(self, mock_embedding_model, mock_generative_model):
        expander = QueryExpander(mock_embedding_model, mock_generative_model)
        vec = expander.expand("caching strategies")
        assert vec.shape == (384,)

    def test_expansion_is_deterministic(self, mock_embedding_model, mock_generative_model):
        expander = QueryExpander(mock_embedding_model, mock_generative_model)
        q = "How does rate limiting protect the system?"
        v1 = expander.expand(q)
        v2 = expander.expand(q)
        np.testing.assert_array_equal(v1, v2)
