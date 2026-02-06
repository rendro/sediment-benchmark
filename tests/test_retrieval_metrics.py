import math

import pytest

from metrics.retrieval import (
    QueryResult,
    compute_retrieval_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_qr(returned: list[str], expected: list[str], category: str = "test") -> QueryResult:
    return QueryResult(query_id="q1", returned_ids=returned, expected_ids=expected, category=category)


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_hit_at_1(self):
        results = [make_qr(["a", "b", "c"], ["a"])]
        assert recall_at_k(results, k=1).aggregate == 1.0

    def test_miss_at_1_hit_at_3(self):
        results = [make_qr(["b", "c", "a"], ["a"])]
        assert recall_at_k(results, k=1).aggregate == 0.0
        assert recall_at_k(results, k=3).aggregate == 1.0

    def test_complete_miss(self):
        results = [make_qr(["b", "c", "d"], ["a"])]
        assert recall_at_k(results, k=3).aggregate == 0.0

    def test_multiple_queries(self):
        results = [
            make_qr(["a", "b"], ["a"]),  # hit
            make_qr(["c", "d"], ["a"]),  # miss
        ]
        assert recall_at_k(results, k=2).aggregate == 0.5

    def test_per_category(self):
        results = [
            make_qr(["a"], ["a"], category="arch"),
            make_qr(["b"], ["a"], category="arch"),
            make_qr(["a"], ["a"], category="code"),
        ]
        scores = recall_at_k(results, k=1)
        assert scores.by_category["arch"] == 0.5
        assert scores.by_category["code"] == 1.0

    def test_empty_results(self):
        assert recall_at_k([], k=5).aggregate == 0.0

    def test_multiple_expected(self):
        # Any one of the expected items in top-k counts as a hit
        results = [make_qr(["x", "b", "c"], ["a", "b"])]
        assert recall_at_k(results, k=2).aggregate == 1.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_position(self):
        results = [make_qr(["a", "b", "c"], ["a"])]
        assert mrr(results).aggregate == 1.0

    def test_second_position(self):
        results = [make_qr(["b", "a", "c"], ["a"])]
        assert mrr(results).aggregate == 0.5

    def test_third_position(self):
        results = [make_qr(["b", "c", "a"], ["a"])]
        assert mrr(results).aggregate == pytest.approx(1 / 3)

    def test_not_found(self):
        results = [make_qr(["b", "c", "d"], ["a"])]
        assert mrr(results).aggregate == 0.0

    def test_average_across_queries(self):
        results = [
            make_qr(["a"], ["a"]),     # RR = 1.0
            make_qr(["b", "a"], ["a"]),  # RR = 0.5
        ]
        assert mrr(results).aggregate == pytest.approx(0.75)

    def test_multiple_expected_takes_first_hit(self):
        # RR uses the *first* relevant item found
        results = [make_qr(["x", "b", "a"], ["a", "b"])]
        assert mrr(results).aggregate == 0.5  # b found at rank 2


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self):
        # Single expected item at rank 1 -> DCG = iDCG -> nDCG = 1.0
        results = [make_qr(["a", "b", "c"], ["a"])]
        assert ndcg_at_k(results, k=3).aggregate == 1.0

    def test_imperfect_ranking(self):
        # Expected at rank 2 instead of rank 1
        results = [make_qr(["b", "a", "c"], ["a"])]
        score = ndcg_at_k(results, k=3).aggregate
        # DCG = 1/log2(3) ≈ 0.631, iDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert score == pytest.approx(expected)

    def test_no_relevant(self):
        results = [make_qr(["b", "c", "d"], ["a"])]
        assert ndcg_at_k(results, k=3).aggregate == 0.0

    def test_two_relevant_items(self):
        # Both expected items in top 3
        results = [make_qr(["a", "x", "b"], ["a", "b"])]
        score = ndcg_at_k(results, k=3).aggregate
        # DCG = 1/log2(2) + 1/log2(4), iDCG = 1/log2(2) + 1/log2(3)
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert score == pytest.approx(dcg / idcg)


# ---------------------------------------------------------------------------
# compute_retrieval_metrics (integration)
# ---------------------------------------------------------------------------


def test_compute_all_metrics():
    results = [
        make_qr(["a", "b", "c", "d", "e"], ["a"], category="arch"),
        make_qr(["x", "y", "a", "z", "w"], ["a"], category="code"),
    ]
    report = compute_retrieval_metrics(results)

    assert report.recall_at_1.aggregate == 0.5
    assert report.recall_at_3.aggregate == 1.0
    assert report.mrr.aggregate == pytest.approx((1.0 + 1 / 3) / 2)
    assert report.recall_at_1.by_category["arch"] == 1.0
    assert report.recall_at_1.by_category["code"] == 0.0
