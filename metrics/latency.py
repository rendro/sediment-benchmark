"""Latency metrics: per-operation timing statistics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyStats:
    """Timing statistics for a set of operations."""

    count: int
    mean: float  # seconds
    min: float
    max: float
    p50: float
    p95: float
    p99: float


@dataclass
class LatencyReport:
    """Store and recall latency statistics."""

    store: LatencyStats
    recall: LatencyStats


def compute_latency_stats(timings: list[float]) -> LatencyStats:
    """Compute latency statistics from a list of elapsed times (seconds).

    Percentiles use linear interpolation on a sorted list.
    """
    if not timings:
        return LatencyStats(count=0, mean=0.0, min=0.0, max=0.0, p50=0.0, p95=0.0, p99=0.0)

    sorted_t = sorted(timings)
    n = len(sorted_t)

    return LatencyStats(
        count=n,
        mean=sum(sorted_t) / n,
        min=sorted_t[0],
        max=sorted_t[-1],
        p50=_percentile(sorted_t, 0.50),
        p95=_percentile(sorted_t, 0.95),
        p99=_percentile(sorted_t, 0.99),
    )


def compute_latency_report(
    store_timings: list[float],
    recall_timings: list[float],
) -> LatencyReport:
    """Compute a full latency report from store and recall timings."""
    return LatencyReport(
        store=compute_latency_stats(store_timings),
        recall=compute_latency_stats(recall_timings),
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation on a sorted list.

    Matches Python's statistics.quantiles(method='inclusive') / numpy's
    'linear' interpolation: index = p * (n - 1).
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
