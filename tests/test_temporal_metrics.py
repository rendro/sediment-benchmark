import pytest

from metrics.temporal import TemporalReport, TemporalResult, compute_temporal_metrics


def make_result(rank: int | None, seq_id: str = "t_001") -> TemporalResult:
    return TemporalResult(sequence_id=seq_id, expected_content="latest fact", rank=rank)


class TestComputeTemporalMetrics:
    def test_all_rank_1(self):
        results = [make_result(1) for _ in range(5)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 1.0
        assert report.recency_at_3 == 1.0
        assert report.mrr == 1.0
        assert report.mean_rank == 1.0

    def test_all_rank_2(self):
        results = [make_result(2) for _ in range(4)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 0.0
        assert report.recency_at_3 == 1.0
        assert report.mrr == pytest.approx(0.5)
        assert report.mean_rank == 2.0

    def test_mixed_ranks(self):
        results = [
            make_result(1),  # hit@1, hit@3, rr=1.0
            make_result(3),  # miss@1, hit@3, rr=1/3
            make_result(5),  # miss@1, miss@3, rr=1/5
            make_result(None),  # not found, rr=0
        ]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == pytest.approx(1 / 4)
        assert report.recency_at_3 == pytest.approx(2 / 4)
        assert report.mrr == pytest.approx((1.0 + 1 / 3 + 1 / 5 + 0) / 4)
        # mean_rank only counts found items: (1 + 3 + 5) / 3
        assert report.mean_rank == pytest.approx(9 / 3)

    def test_none_found(self):
        results = [make_result(None) for _ in range(3)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 0.0
        assert report.recency_at_3 == 0.0
        assert report.mrr == 0.0
        assert report.mean_rank == 0.0

    def test_empty_results(self):
        report = compute_temporal_metrics([])
        assert report.recency_at_1 == 0.0
        assert report.recency_at_3 == 0.0
        assert report.mrr == 0.0
        assert report.mean_rank == 0.0

    def test_single_perfect(self):
        results = [make_result(1)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 1.0
        assert report.mrr == 1.0
        assert report.mean_rank == 1.0

    def test_rank_exactly_3(self):
        results = [make_result(3)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 0.0
        assert report.recency_at_3 == 1.0

    def test_rank_4_misses_at_3(self):
        results = [make_result(4)]
        report = compute_temporal_metrics(results)
        assert report.recency_at_1 == 0.0
        assert report.recency_at_3 == 0.0
        assert report.mrr == pytest.approx(0.25)
