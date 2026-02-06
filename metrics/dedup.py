"""Deduplication metrics: does the system handle near-duplicate memories?"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DedupResult:
    """Result for a single dedup pair."""

    pair_id: str
    original_content: str
    duplicate_content: str
    original_retrieved: bool  # found in recall@1?


@dataclass
class DedupReport:
    """Aggregate dedup metrics."""

    consolidation_rate: float  # 1 - actual/expected (0% = no dedup)
    recall_after_dedup: float  # fraction of originals still retrievable
    pair_count: int


def compute_dedup_metrics(
    results: list[DedupResult],
    stored_count: int,
    expected_count: int,
) -> DedupReport:
    """Compute dedup metrics from pair results and counts."""
    if not results:
        return DedupReport(consolidation_rate=0.0, recall_after_dedup=0.0, pair_count=0)

    consolidation_rate = 1.0 - (stored_count / expected_count) if expected_count > 0 else 0.0
    # Clamp to [0, 1] — stored_count shouldn't exceed expected, but be safe
    consolidation_rate = max(0.0, min(1.0, consolidation_rate))

    retrieved = sum(1 for r in results if r.original_retrieved)
    recall_after_dedup = retrieved / len(results)

    return DedupReport(
        consolidation_rate=consolidation_rate,
        recall_after_dedup=recall_after_dedup,
        pair_count=len(results),
    )
