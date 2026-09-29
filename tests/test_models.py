from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from autonomous_ml_experimenter.demo.recommendation.models import (
    ALSModel,
    CoVisitationModel,
)


def _training_data() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [
                start,
                start + timedelta(minutes=2),
                start + timedelta(hours=1),
                start + timedelta(hours=1, minutes=2),
                start + timedelta(hours=2),
                start + timedelta(hours=3),
                start + timedelta(hours=3, minutes=2),
            ],
            "visitorid": [1, 1, 2, 2, 3, 4, 4],
            "itemid": [10, 20, 10, 30, 10, 20, 30],
            "event": ["view", "transaction", "view", "addtocart", "view", "view", "view"],
            "weight": [1.0, 5.0, 1.0, 3.0, 1.0, 1.0, 1.0],
        }
    )


def test_covisitation_is_personalized_excludes_seen_and_round_trips(tmp_path: Path) -> None:
    train = _training_data()
    model = CoVisitationModel(session_minutes=30, recency_decay=0.9, neighbor_limit=20).fit(train)
    recommendation = model.predict([3], k=2)[3]
    assert recommendation
    assert 10 not in recommendation
    assert recommendation[0] == 20
    assert model.predict([999], k=2)[999] == [20, 30]

    path = tmp_path / "covisitation.json"
    model.save(path)
    restored = CoVisitationModel.load(path)
    assert restored.predict([3], k=2) == model.predict([3], k=2)


def test_als_is_deterministic_excludes_seen_falls_back_and_round_trips(tmp_path: Path) -> None:
    train = _training_data()
    model = ALSModel(factors=3, regularization=0.1, iterations=8, alpha=10.0, seed=7).fit(train)
    first = model.predict([1], k=2)[1]
    assert set(first).isdisjoint({10, 20})
    assert model.predict([999], k=2)[999] == [20, 30]

    path = tmp_path / "als"
    model.save(path)
    restored = ALSModel.load(path)
    assert restored.predict([1], k=2) == {1: first}
    assert restored.predict([999], k=2) == {999: [20, 30]}


def test_adapters_expose_shared_model_contract() -> None:
    train = _training_data()
    for model in [
        CoVisitationModel().fit(train),
        ALSModel(factors=2, iterations=3, seed=1).fit(train),
    ]:
        assert model.name
        assert model.known_users
        assert model.known_items
        assert isinstance(model.predict([1], k=1), dict)
