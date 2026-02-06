"""Retrieval quality metrics: Recall@k, MRR, nDCG@5."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Single query evaluation input."""

    query_id: str
    returned_ids: list[str]  # ordered by rank (best first)
    expected_ids: list[str]  # ground-truth relevant IDs
    category: str | None = None


@dataclass
class MetricScores:
    """Aggregate + per-category metric scores."""

    aggregate: float
    by_category: dict[str, float]


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def recall_at_k(results: list[QueryResult], k: int) -> MetricScores:
    """Fraction of queries where at least one expected item is in the top k.

    This is a "hit rate" style recall — did we find *any* relevant item?
    """
    hits: dict[str, list[int]] = {}  # category -> list of 0/1

    for qr in results:
        top_k = set(qr.returned_ids[:k])
        hit = 1 if top_k & set(qr.expected_ids) else 0
        cat = qr.category or "_all"
        hits.setdefault(cat, []).append(hit)

    by_category = {cat: _mean(vals) for cat, vals in hits.items()}
    all_hits = [1 if set(qr.returned_ids[:k]) & set(qr.expected_ids) else 0 for qr in results]
    return MetricScores(aggregate=_mean(all_hits), by_category=by_category)


# ---------------------------------------------------------------------------
# MRR (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------


def mrr(results: list[QueryResult]) -> MetricScores:
    """Mean Reciprocal Rank: 1/rank of the first relevant result."""
    rr_scores: dict[str, list[float]] = {}

    for qr in results:
        expected = set(qr.expected_ids)
        rr = 0.0
        for rank, rid in enumerate(qr.returned_ids, start=1):
            if rid in expected:
                rr = 1.0 / rank
                break
        cat = qr.category or "_all"
        rr_scores.setdefault(cat, []).append(rr)

    by_category = {cat: _mean(vals) for cat, vals in rr_scores.items()}
    all_rr = []
    for qr in results:
        expected = set(qr.expected_ids)
        rr = 0.0
        for rank, rid in enumerate(qr.returned_ids, start=1):
            if rid in expected:
                rr = 1.0 / rank
                break
        all_rr.append(rr)

    return MetricScores(aggregate=_mean(all_rr), by_category=by_category)


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


def ndcg_at_k(results: list[QueryResult], k: int = 5) -> MetricScores:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if the item is in expected_ids, else 0.
    """
    scores: dict[str, list[float]] = {}

    for qr in results:
        expected = set(qr.expected_ids)
        dcg = _dcg(qr.returned_ids[:k], expected)
        # Ideal: all expected items at the top positions
        ideal_count = min(len(expected), k)
        ideal_ids = list(expected)[:ideal_count]
        idcg = _dcg(ideal_ids, expected)
        ndcg_val = dcg / idcg if idcg > 0 else 0.0

        cat = qr.category or "_all"
        scores.setdefault(cat, []).append(ndcg_val)

    by_category = {cat: _mean(vals) for cat, vals in scores.items()}
    all_ndcg = []
    for qr in results:
        expected = set(qr.expected_ids)
        dcg = _dcg(qr.returned_ids[:k], expected)
        ideal_count = min(len(expected), k)
        ideal_ids = list(expected)[:ideal_count]
        idcg = _dcg(ideal_ids, expected)
        all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return MetricScores(aggregate=_mean(all_ndcg), by_category=by_category)


def _dcg(ranked_ids: list[str], relevant: set[str]) -> float:
    """Discounted Cumulative Gain with binary relevance."""
    total = 0.0
    for i, rid in enumerate(ranked_ids):
        rel = 1.0 if rid in relevant else 0.0
        total += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1)
    return total


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------


@dataclass
class RetrievalReport:
    recall_at_1: MetricScores
    recall_at_3: MetricScores
    recall_at_5: MetricScores
    recall_at_10: MetricScores
    mrr: MetricScores
    ndcg_at_5: MetricScores


def compute_retrieval_metrics(results: list[QueryResult]) -> RetrievalReport:
    """Compute all retrieval metrics from a list of query results."""
    return RetrievalReport(
        recall_at_1=recall_at_k(results, k=1),
        recall_at_3=recall_at_k(results, k=3),
        recall_at_5=recall_at_k(results, k=5),
        recall_at_10=recall_at_k(results, k=10),
        mrr=mrr(results),
        ndcg_at_5=ndcg_at_k(results, k=5),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
