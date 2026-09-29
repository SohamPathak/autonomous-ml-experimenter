from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from autonomous_ml_experimenter.config import AppConfig, DataConfig, load_config
from autonomous_ml_experimenter.core.security import redact
from autonomous_ml_experimenter.demo.recommendation.fixtures import generate_interactions


def test_demo_config_loads() -> None:
    config = load_config(Path("configs/demo.yaml"))
    assert config.schema_version == "1.0"
    assert config.data.mode == "synthetic"


def test_public_data_requires_path() -> None:
    with pytest.raises(ValidationError):
        DataConfig(mode="full")


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unexpected": True})


def test_fixture_is_deterministic_and_temporal() -> None:
    first = generate_interactions(users=24, days=8, seed=7)
    second = generate_interactions(users=24, days=8, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["timestamp"].is_monotonic_increasing
    assert set(first.columns) == {"timestamp", "visitorid", "event", "itemid"}


def test_redaction_removes_secret_values_and_absolute_paths() -> None:
    payload = {
        "access_token": "do-not-keep",
        "nested": {"credential_path": "/Users/example/secret.json"},
        "artifact": "/Users/example/output.json",
    }
    safe = redact(payload)
    assert safe["access_token"] == "[REDACTED]"
    assert safe["nested"]["credential_path"] == "[REDACTED]"
    assert safe["artifact"] == "output.json"
    assert "do-not-keep" not in str(safe)
