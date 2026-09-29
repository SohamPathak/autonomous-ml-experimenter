from pathlib import Path

import pandas as pd
import pytest

from autonomous_ml_experimenter.demo.recommendation.evaluation import (
    evaluate_recommender,
    ranking_metrics,
)
from autonomous_ml_experimenter.demo.recommendation.models import PopularityModel


def test_ranking_metrics_match_hand_calculation() -> None:
    predictions = {1: [10, 20, 30], 2: [20, 10, 30]}
    relevant = {1: {20}, 2: {30}}
    metrics = ranking_metrics(predictions, relevant, catalog_size=4, k=3)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert metrics["ndcg_at_3"] == pytest.approx((1 / 1.5849625 + 0.5) / 2)
    assert metrics["mrr_at_3"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert metrics["catalog_coverage"] == pytest.approx(0.75)
    assert metrics["recommendation_concentration"] == pytest.approx(1 / 3)


def test_popularity_is_weighted_deterministic_and_excludes_seen(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2, 3],
            "itemid": [10, 20, 10, 30, 30],
            "weight": [1.0, 1.0, 1.0, 5.0, 1.0],
        }
    )
    model = PopularityModel().fit(train)
    assert model.predict([1], k=3)[1] == [30]
    assert model.predict([99], k=3)[99] == [30, 10, 20]
    assert model.fallback_rate == pytest.approx(1.0)

    path = tmp_path / "popularity.json"
    model.save(path)
    restored = PopularityModel.load(path)
    assert restored.predict([1], k=3) == {1: [30]}


def test_evaluator_reports_cohorts_unknowns_and_latency() -> None:
    train = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2],
            "itemid": [10, 20, 10, 30],
            "weight": [1.0, 1.0, 1.0, 3.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "visitorid": [1, 2, 99],
            "itemid": [30, 20, 999],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    model = PopularityModel().fit(train)
    metrics = evaluate_recommender(model, train, evaluation, k=2)
    assert set(
        [
            "ndcg_at_2",
            "recall_at_2",
            "mrr_at_2",
            "catalog_coverage",
            "recommendation_concentration",
            "warm_recall_at_2",
            "cold_recall_at_2",
            "unknown_user_rate",
            "unknown_item_rate",
            "fallback_rate",
            "mean_latency_ms",
            "p95_latency_ms",
        ]
    ).issubset(metrics)
    assert metrics["unknown_user_rate"] == pytest.approx(1 / 3)
    assert metrics["unknown_item_rate"] == pytest.approx(1 / 3)
    assert metrics["fallback_rate"] == pytest.approx(1 / 3)
    assert metrics["p95_latency_ms"] >= metrics["mean_latency_ms"] >= 0
