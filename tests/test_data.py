from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from autonomous_ml_experimenter.config import DataConfig
from autonomous_ml_experimenter.demo.recommendation.data import (
    EVENT_WEIGHTS,
    InteractionSplits,
    prepare_interactions,
    validate_events,
)


def _events() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for day in range(10):
        for user in range(4):
            rows.append(
                {
                    "timestamp": start + timedelta(days=day, minutes=user),
                    "visitorid": user,
                    "event": "transaction" if day % 5 == 0 else "view",
                    "itemid": user * 10 + day % 3,
                }
            )
    rows.extend(
        [
            {
                "timestamp": start + timedelta(days=8, hours=1),
                "visitorid": 99,
                "event": "view",
                "itemid": 999,
            },
            {
                "timestamp": start + timedelta(days=9, hours=1),
                "visitorid": 99,
                "event": "addtocart",
                "itemid": 999,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_schema_validation_rejects_missing_and_unknown_events() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_events(pd.DataFrame({"visitorid": [1]}))

    invalid = _events()
    invalid.loc[0, "event"] = "refund"
    with pytest.raises(ValueError, match="unsupported event"):
        validate_events(invalid)


def test_preparation_is_deterministic_and_fingerprinted() -> None:
    config = DataConfig(
        mode="synthetic",
        sample_users=20,
        min_user_events=2,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    first = prepare_interactions(_events(), config, seed=9)
    second = prepare_interactions(_events(), config, seed=9)
    assert isinstance(first, InteractionSplits)
    assert first.manifest.fingerprint == second.manifest.fingerprint
    pd.testing.assert_frame_equal(first.train, second.train)
    assert first.manifest.row_count == len(_events())


def test_temporal_boundaries_do_not_overlap() -> None:
    config = DataConfig(sample_users=20, min_user_events=2)
    splits = prepare_interactions(_events(), config, seed=4)
    assert splits.train["timestamp"].max() < splits.validation["timestamp"].min()
    assert splits.validation["timestamp"].max() < splits.test["timestamp"].min()
    assert not splits.train.empty
    assert not splits.validation.empty
    assert not splits.test.empty


def test_mappings_are_training_only_and_cold_entities_are_preserved() -> None:
    config = DataConfig(sample_users=20, min_user_events=2)
    splits = prepare_interactions(_events(), config, seed=4)
    assert 99 not in splits.known_users
    assert 999 not in splits.known_items
    assert 99 in set(splits.validation["visitorid"]) | set(splits.test["visitorid"])
    assert splits.cohort_report["cold_users"] >= 1
    assert splits.cohort_report["cold_items"] >= 1


def test_event_weights_are_fixed_and_applied() -> None:
    splits = prepare_interactions(_events(), DataConfig(sample_users=20, min_user_events=2))
    for event, weight in EVENT_WEIGHTS.items():
        matching = splits.train.loc[splits.train["event"] == event, "weight"]
        if not matching.empty:
            assert matching.eq(weight).all()
