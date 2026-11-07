import math
from types import SimpleNamespace

import pytest

import evaluation.test as questions
from evaluation import eval as ev


def docs(*texts):
    return [SimpleNamespace(page_content=t) for t in texts]


def test_mrr_is_reciprocal_rank_of_first_match():
    assert ev.calculate_mrr("beta", docs("alpha", "Beta is here", "beta")) == 0.5


def test_mrr_is_zero_without_match():
    assert ev.calculate_mrr("zeta", docs("alpha", "beta")) == 0.0


def test_dcg_discounts_by_position():
    assert ev.calculate_dcg([1, 0, 1], 3) == pytest.approx(1 + 1 / math.log2(4))


def test_ndcg_is_one_when_keyword_ranked_first():
    assert ev.calculate_ndcg("alpha", docs("alpha", "x", "y")) == pytest.approx(1.0)


def test_ndcg_penalises_late_match():
    assert ev.calculate_ndcg("alpha", docs("x", "y", "alpha")) == pytest.approx(0.5)


def test_ndcg_is_zero_without_match():
    assert ev.calculate_ndcg("zeta", docs("x", "y")) == 0.0


def test_evaluate_retrieval_aggregates_over_keywords(monkeypatch):
    monkeypatch.setattr(ev, "fetch_context", lambda q: docs("alpha text", "x beta"))
    test = questions.TestQuestion(
        question="q",
        keywords=["alpha", "beta", "gamma"],
        reference_answer="r",
        category="direct_fact",
    )
    result = ev.evaluate_retrieval(test)
    assert result.keywords_found == 2 and result.total_keywords == 3
    assert result.keyword_coverage == pytest.approx(200 / 3)
    assert result.mrr == pytest.approx((1 + 0.5 + 0) / 3)