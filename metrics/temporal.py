"""Temporal correctness metrics: does the system return the latest version of a fact?"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemporalResult:
    """Result for a single temporal sequence."""

    sequence_id: str
    expected_content: str  # the newest fact
    rank: int | None  # 1-based rank of newest fact in results, None if not found


@dataclass
class TemporalReport:
    """Aggregate temporal metrics."""

    recency_at_1: float  # fraction where newest is rank 1
    recency_at_3: float  # fraction where newest is in top 3
    mrr: float  # mean reciprocal rank
    mean_rank: float  # average rank (lower = better)


def compute_temporal_metrics(results: list[TemporalResult]) -> TemporalReport:
    """Compute temporal metrics from a list of sequence results."""
    if not results:
        return TemporalReport(recency_at_1=0.0, recency_at_3=0.0, mrr=0.0, mean_rank=0.0)

    n = len(results)
    hit_at_1 = 0
    hit_at_3 = 0
    rr_sum = 0.0
    rank_sum = 0.0
    found_count = 0

    for r in results:
        if r.rank is not None:
            if r.rank == 1:
                hit_at_1 += 1
            if r.rank <= 3:
                hit_at_3 += 1
            rr_sum += 1.0 / r.rank
            rank_sum += r.rank
            found_count += 1
        # If not found, rr contribution is 0 (already excluded)

    return TemporalReport(
        recency_at_1=hit_at_1 / n,
        recency_at_3=hit_at_3 / n,
        mrr=rr_sum / n,
        mean_rank=rank_sum / found_count if found_count > 0 else 0.0,
    )
