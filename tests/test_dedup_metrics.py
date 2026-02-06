import pytest

from metrics.dedup import DedupReport, DedupResult, compute_dedup_metrics


def make_result(retrieved: bool, pair_id: str = "d_001") -> DedupResult:
    return DedupResult(
        pair_id=pair_id,
        original_content="original fact",
        duplicate_content="Note: original fact",
        original_retrieved=retrieved,
    )


class TestComputeDedupMetrics:
    def test_no_consolidation_all_retrieved(self):
        results = [make_result(True) for _ in range(5)]
        report = compute_dedup_metrics(results, stored_count=10, expected_count=10)
        assert report.consolidation_rate == 0.0
        assert report.recall_after_dedup == 1.0
        assert report.pair_count == 5

    def test_full_consolidation(self):
        results = [make_result(True) for _ in range(5)]
        # 5 stored out of 10 expected = 50% consolidation
        report = compute_dedup_metrics(results, stored_count=5, expected_count=10)
        assert report.consolidation_rate == pytest.approx(0.5)
        assert report.recall_after_dedup == 1.0

    def test_partial_recall(self):
        results = [
            make_result(True),
            make_result(True),
            make_result(False),
            make_result(False),
        ]
        report = compute_dedup_metrics(results, stored_count=8, expected_count=8)
        assert report.consolidation_rate == 0.0
        assert report.recall_after_dedup == pytest.approx(0.5)
        assert report.pair_count == 4

    def test_consolidation_with_data_loss(self):
        results = [make_result(False) for _ in range(3)]
        report = compute_dedup_metrics(results, stored_count=3, expected_count=6)
        assert report.consolidation_rate == pytest.approx(0.5)
        assert report.recall_after_dedup == 0.0

    def test_empty_results(self):
        report = compute_dedup_metrics([], stored_count=0, expected_count=0)
        assert report.consolidation_rate == 0.0
        assert report.recall_after_dedup == 0.0
        assert report.pair_count == 0

    def test_consolidation_clamped_to_zero(self):
        # Edge case: stored_count > expected_count shouldn't give negative rate
        results = [make_result(True)]
        report = compute_dedup_metrics(results, stored_count=12, expected_count=10)
        assert report.consolidation_rate == 0.0

    def test_all_consolidated_all_retrieved(self):
        # Aggressive consolidation but still retrievable
        results = [make_result(True) for _ in range(5)]
        report = compute_dedup_metrics(results, stored_count=5, expected_count=10)
        assert report.consolidation_rate == pytest.approx(0.5)
        assert report.recall_after_dedup == 1.0
