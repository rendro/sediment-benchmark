import pytest

from metrics.latency import (
    LatencyReport,
    LatencyStats,
    compute_latency_report,
    compute_latency_stats,
)


class TestComputeLatencyStats:
    def test_empty_list(self):
        stats = compute_latency_stats([])
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0

    def test_single_value(self):
        stats = compute_latency_stats([0.042])
        assert stats.count == 1
        assert stats.mean == pytest.approx(0.042)
        assert stats.min == pytest.approx(0.042)
        assert stats.max == pytest.approx(0.042)
        assert stats.p50 == pytest.approx(0.042)
        assert stats.p95 == pytest.approx(0.042)
        assert stats.p99 == pytest.approx(0.042)

    def test_two_values(self):
        stats = compute_latency_stats([0.010, 0.020])
        assert stats.count == 2
        assert stats.mean == pytest.approx(0.015)
        assert stats.min == pytest.approx(0.010)
        assert stats.max == pytest.approx(0.020)
        assert stats.p50 == pytest.approx(0.015)
        # p95: idx = 0.95 * 1 = 0.95 → 0.010 + 0.95 * 0.010 = 0.0195
        assert stats.p95 == pytest.approx(0.0195)

    def test_known_distribution(self):
        # 100 values: 0.01, 0.02, ..., 1.00
        timings = [i * 0.01 for i in range(1, 101)]
        stats = compute_latency_stats(timings)
        assert stats.count == 100
        assert stats.mean == pytest.approx(0.505, abs=1e-6)
        assert stats.min == pytest.approx(0.01)
        assert stats.max == pytest.approx(1.0)
        # p50: idx = 0.50 * 99 = 49.5 → interpolate between [49]=0.50 and [50]=0.51
        assert stats.p50 == pytest.approx(0.505, abs=1e-4)
        # p95: idx = 0.95 * 99 = 94.05 → interpolate between [94]=0.95 and [95]=0.96
        assert stats.p95 == pytest.approx(0.9505, abs=1e-4)
        # p99: idx = 0.99 * 99 = 98.01 → interpolate between [98]=0.99 and [99]=1.00
        assert stats.p99 == pytest.approx(0.9901, abs=1e-4)

    def test_unsorted_input(self):
        # Ensure sorting is handled internally
        timings = [0.050, 0.010, 0.030, 0.020, 0.040]
        stats = compute_latency_stats(timings)
        assert stats.min == pytest.approx(0.010)
        assert stats.max == pytest.approx(0.050)
        assert stats.mean == pytest.approx(0.030)
        # p50: idx = 0.50 * 4 = 2.0 → sorted[2] = 0.030
        assert stats.p50 == pytest.approx(0.030)

    def test_identical_values(self):
        timings = [0.025] * 20
        stats = compute_latency_stats(timings)
        assert stats.count == 20
        assert stats.mean == pytest.approx(0.025)
        assert stats.min == pytest.approx(0.025)
        assert stats.max == pytest.approx(0.025)
        assert stats.p50 == pytest.approx(0.025)
        assert stats.p95 == pytest.approx(0.025)
        assert stats.p99 == pytest.approx(0.025)

    def test_outlier_heavy(self):
        # 9 fast operations + 1 very slow outlier
        timings = [0.01] * 9 + [10.0]
        stats = compute_latency_stats(timings)
        assert stats.count == 10
        assert stats.mean == pytest.approx(1.009)
        assert stats.min == pytest.approx(0.01)
        assert stats.max == pytest.approx(10.0)
        # p50: idx = 0.50 * 9 = 4.5 → between sorted[4]=0.01 and sorted[5]=0.01
        assert stats.p50 == pytest.approx(0.01)
        # p95: idx = 0.95 * 9 = 8.55 → between sorted[8]=0.01 and sorted[9]=10.0
        assert stats.p95 == pytest.approx(0.01 + 0.55 * (10.0 - 0.01), abs=1e-4)
        # p99: idx = 0.99 * 9 = 8.91 → between sorted[8]=0.01 and sorted[9]=10.0
        assert stats.p99 == pytest.approx(0.01 + 0.91 * (10.0 - 0.01), abs=1e-4)


class TestComputeLatencyReport:
    def test_report_structure(self):
        store_t = [0.01, 0.02, 0.03]
        recall_t = [0.05, 0.10]
        report = compute_latency_report(store_t, recall_t)
        assert isinstance(report, LatencyReport)
        assert isinstance(report.store, LatencyStats)
        assert isinstance(report.recall, LatencyStats)
        assert report.store.count == 3
        assert report.recall.count == 2

    def test_asymmetric_counts(self):
        store_t = [0.01] * 100
        recall_t = [0.05] * 50
        report = compute_latency_report(store_t, recall_t)
        assert report.store.count == 100
        assert report.recall.count == 50
        assert report.store.mean == pytest.approx(0.01)
        assert report.recall.mean == pytest.approx(0.05)

    def test_empty_store_nonempty_recall(self):
        report = compute_latency_report([], [0.05, 0.10])
        assert report.store.count == 0
        assert report.recall.count == 2
