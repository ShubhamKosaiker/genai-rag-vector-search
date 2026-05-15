"""Tests for the Benchmarker evaluation logic."""

import pytest

from src.benchmarker import Benchmarker, BenchmarkReport, _mrr, _hit_at_k
from src.embeddings import MockVertexEmbeddingModel
from src.query_expansion import MockGenerativeModel
from src.retrieval import RAGPipeline
from src.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Unit tests for metric functions
# ---------------------------------------------------------------------------

def _fake_results(chunk_ids: list[int]) -> list[SearchResult]:
    return [SearchResult(chunk_id=cid, text=f"chunk {cid}", score=1.0) for cid in chunk_ids]


class TestMRR:
    def test_first_result_relevant(self):
        assert _mrr({0}, _fake_results([0, 1, 2])) == 1.0

    def test_second_result_relevant(self):
        assert _mrr({1}, _fake_results([0, 1, 2])) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert _mrr({2}, _fake_results([0, 1, 2])) == pytest.approx(1 / 3)

    def test_no_relevant_result(self):
        assert _mrr({9}, _fake_results([0, 1, 2])) == 0.0

    def test_multiple_relevant_ids_first_hit_counts(self):
        assert _mrr({1, 2}, _fake_results([0, 1, 2])) == pytest.approx(0.5)


class TestHitAtK:
    def test_hit_at_1_found(self):
        assert _hit_at_k({0}, _fake_results([0, 1, 2]), 1) is True

    def test_hit_at_1_not_found(self):
        assert _hit_at_k({5}, _fake_results([0, 1, 2]), 1) is False

    def test_hit_at_3_found_at_rank_3(self):
        assert _hit_at_k({2}, _fake_results([0, 1, 2]), 3) is True

    def test_hit_at_3_not_found(self):
        assert _hit_at_k({9}, _fake_results([0, 1, 2]), 3) is False


# ---------------------------------------------------------------------------
# Integration tests for Benchmarker
# ---------------------------------------------------------------------------

MINI_CHUNKS = [
    "Load balancing distributes requests to prevent bottlenecks during peak traffic.",
    "Autoscaling provisions new instances under heavy CPU utilisation.",
    "In-memory caching with Redis reduces database read latency.",
    "Circuit breakers prevent cascading failures in distributed systems.",
    "Connection pooling avoids costly connection setup per query.",
]

MINI_QUERIES = [
    {"query": "How does the system handle peak load?", "relevant_ids": [0, 1]},
    {"query": "What caching strategies improve latency?", "relevant_ids": [2]},
    {"query": "How is database performance maintained?", "relevant_ids": [4]},
]


@pytest.fixture
def benchmarker():
    pipeline = RAGPipeline(
        embedding_model=MockVertexEmbeddingModel(),
        generative_model=MockGenerativeModel(),
        embedding_dim=384,
    )
    pipeline.ingest(MINI_CHUNKS)
    return Benchmarker(pipeline, top_k=3)


class TestBenchmarker:
    def test_report_has_correct_query_count(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        assert len(report.queries) == len(MINI_QUERIES)

    def test_report_mrr_in_valid_range(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        assert 0.0 <= report.mean_mrr_a <= 1.0
        assert 0.0 <= report.mean_mrr_b <= 1.0

    def test_hit_rates_in_valid_range(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        for attr in ("mean_hit_at_1_a", "mean_hit_at_1_b", "mean_hit_at_3_a", "mean_hit_at_3_b"):
            val = getattr(report, attr)
            assert 0.0 <= val <= 1.0, f"{attr} out of range: {val}"

    def test_latency_values_are_positive(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        for qb in report.queries:
            assert qb.latency_p50_a_ms >= 0
            assert qb.latency_p50_b_ms >= 0
            assert qb.latency_p95_a_ms >= 0
            assert qb.latency_p95_b_ms >= 0

    def test_top3_chunks_are_populated(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        for qb in report.queries:
            assert len(qb.top3_chunks_a) > 0
            assert len(qb.top3_chunks_b) > 0

    def test_report_serialises_to_json(self, benchmarker):
        import json
        report = benchmarker.run(MINI_QUERIES)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "mean_mrr_a" in parsed
        assert "mean_mrr_b" in parsed
        assert "queries" in parsed

    def test_improvement_pct_is_numeric(self, benchmarker):
        report = benchmarker.run(MINI_QUERIES)
        assert isinstance(report.strategy_b_mrr_improvement_pct, float)
