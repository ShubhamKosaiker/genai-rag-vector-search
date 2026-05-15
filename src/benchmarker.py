"""
Benchmarker: compares Strategy A vs Strategy B with real evaluation metrics.

Metrics computed:
    MRR@k  — Mean Reciprocal Rank (how high does the first relevant result appear?)
    Hit@1  — Does the top result contain a relevant chunk?
    Hit@k  — Does any result in the top-k contain a relevant chunk?
    Latency — P50 and P95 across repeated runs (5 warm-up + 10 measured)

Why MRR and not just printing top-k chunks?
MRR captures the user experience: a system that surfaces the right answer
at rank 1 is meaningfully better than one that surfaces it at rank 3, even
if both technically "retrieve" the relevant document. This is the metric
used in production search evaluation (Google, Bing, Vertex AI Search).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

from src.retrieval import RAGPipeline, RetrievalResult


@dataclass
class QueryBenchmark:
    query: str
    relevant_chunk_ids: list[int]       # ground-truth chunk indices

    mrr_a: float = 0.0
    mrr_b: float = 0.0
    hit_at_1_a: bool = False
    hit_at_1_b: bool = False
    hit_at_3_a: bool = False
    hit_at_3_b: bool = False
    latency_p50_a_ms: float = 0.0
    latency_p50_b_ms: float = 0.0
    latency_p95_a_ms: float = 0.0
    latency_p95_b_ms: float = 0.0
    top3_chunks_a: list[str] = field(default_factory=list)
    top3_chunks_b: list[str] = field(default_factory=list)
    top3_scores_a: list[float] = field(default_factory=list)
    top3_scores_b: list[float] = field(default_factory=list)
    rewrites_b: list[str] = field(default_factory=list)     # actual rewrites used in Strategy B


@dataclass
class BenchmarkReport:
    queries: list[QueryBenchmark]
    mean_mrr_a: float = 0.0
    mean_mrr_b: float = 0.0
    mean_hit_at_1_a: float = 0.0
    mean_hit_at_1_b: float = 0.0
    mean_hit_at_3_a: float = 0.0
    mean_hit_at_3_b: float = 0.0
    strategy_b_mrr_improvement_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _mrr(relevant_ids: set[int], results: list) -> float:
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _hit_at_k(relevant_ids: set[int], results: list, k: int) -> bool:
    return any(r.chunk_id in relevant_ids for r in results[:k])


def _measure_latency(
    fn: Callable[[], RetrievalResult],
    warmup: int = 3,
    runs: int = 10,
) -> tuple[float, float]:
    """Return (p50_ms, p95_ms) for the given retrieval function."""
    for _ in range(warmup):
        fn()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p95 = latencies[p95_idx]
    return p50, p95


class Benchmarker:
    """
    Runs Strategy A and Strategy B over a set of labelled queries and
    produces a structured BenchmarkReport with retrieval quality metrics
    and latency percentiles.
    """

    def __init__(self, pipeline: RAGPipeline, top_k: int = 3):
        self.pipeline = pipeline
        self.top_k = top_k

    def run(self, queries: list[dict]) -> BenchmarkReport:
        """
        Parameters
        ----------
        queries : list of dicts with keys:
            "query"            : str
            "relevant_ids"     : list[int]  — ground-truth chunk indices
        """
        results = []

        for q in queries:
            text = q["query"]
            relevant = set(q["relevant_ids"])

            # --- latency measurement (warm cache) ---------------------------
            p50_a, p95_a = _measure_latency(
                lambda t=text: self.pipeline.retrieve_strategy_a(t, self.top_k)
            )
            p50_b, p95_b = _measure_latency(
                lambda t=text: self.pipeline.retrieve_strategy_b(t, self.top_k)
            )

            # --- single retrieval for result inspection ---------------------
            res_a = self.pipeline.retrieve_strategy_a(text, self.top_k)
            res_b = self.pipeline.retrieve_strategy_b(text, self.top_k)

            qb = QueryBenchmark(
                query=text,
                relevant_chunk_ids=sorted(relevant),
                mrr_a=_mrr(relevant, res_a.results),
                mrr_b=_mrr(relevant, res_b.results),
                hit_at_1_a=_hit_at_k(relevant, res_a.results, 1),
                hit_at_1_b=_hit_at_k(relevant, res_b.results, 1),
                hit_at_3_a=_hit_at_k(relevant, res_a.results, self.top_k),
                hit_at_3_b=_hit_at_k(relevant, res_b.results, self.top_k),
                latency_p50_a_ms=round(p50_a, 2),
                latency_p50_b_ms=round(p50_b, 2),
                latency_p95_a_ms=round(p95_a, 2),
                latency_p95_b_ms=round(p95_b, 2),
                top3_chunks_a=[r.text[:120] for r in res_a.results],
                top3_chunks_b=[r.text[:120] for r in res_b.results],
                top3_scores_a=[round(r.score, 4) for r in res_a.results],
                top3_scores_b=[round(r.score, 4) for r in res_b.results],
                rewrites_b=res_b.rewrites,
            )
            results.append(qb)

        # --- aggregate metrics ---------------------------------------------
        n = len(results)
        mean_mrr_a = sum(r.mrr_a for r in results) / n
        mean_mrr_b = sum(r.mrr_b for r in results) / n
        improvement = (
            ((mean_mrr_b - mean_mrr_a) / mean_mrr_a * 100)
            if mean_mrr_a > 0
            else 0.0
        )

        report = BenchmarkReport(
            queries=results,
            mean_mrr_a=round(mean_mrr_a, 4),
            mean_mrr_b=round(mean_mrr_b, 4),
            mean_hit_at_1_a=round(sum(r.hit_at_1_a for r in results) / n, 4),
            mean_hit_at_1_b=round(sum(r.hit_at_1_b for r in results) / n, 4),
            mean_hit_at_3_a=round(sum(r.hit_at_3_a for r in results) / n, 4),
            mean_hit_at_3_b=round(sum(r.hit_at_3_b for r in results) / n, 4),
            strategy_b_mrr_improvement_pct=round(improvement, 2),
        )
        return report
