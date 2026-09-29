from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from autonomous_ml_experimenter.core.contracts import (
    DataManifest,
    DecisionRecord,
    DecisionStatus,
    HealthStatus,
    Hypothesis,
    MonitoringWindow,
    Narrative,
    PresentationBundle,
    TrialOutcome,
    TrialResult,
)
from autonomous_ml_experimenter.demo.recommendation.models import PopularityModel
from autonomous_ml_experimenter.monitoring.replay import MonitoringReplay
from autonomous_ml_experimenter.presentation.bundle import load_bundle, write_bundle


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    train_rows: list[dict[str, object]] = []
    later_rows: list[dict[str, object]] = []
    for day in range(12):
        target = train_rows if day < 6 else later_rows
        for user in range(8):
            target.append(
                {
                    "timestamp": start + timedelta(days=day, minutes=user),
                    "visitorid": user,
                    "itemid": (user + day) % 10,
                    "event": "view" if day % 3 else "addtocart",
                    "weight": 1.0 if day % 3 else 3.0,
                }
            )
    return pd.DataFrame(train_rows), pd.DataFrame(later_rows)


def _trial() -> TrialResult:
    return TrialResult(
        trial_id="trial-1",
        phase="V1",
        model_name="popularity",
        parameters={},
        metrics={"ndcg_at_10": 0.1, "p95_latency_ms": 1.0},
        hypothesis=Hypothesis(
            observation="Baseline measured.",
            hypothesis="Monitor over time.",
            expected_effect="Detect drift.",
            parameter_changes={},
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=1,
    )


def _bundle(
    monitoring: list[MonitoringWindow], conclusion: str = "Evidence is offline."
) -> PresentationBundle:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return PresentationBundle(
        data_manifest=DataManifest(
            fingerprint="abc",
            row_count=10,
            user_count=2,
            item_count=3,
            start_time=start,
            end_time=start + timedelta(days=1),
            source="generated fixture",
        ),
        trials=[_trial()],
        decision=DecisionRecord(
            status=DecisionStatus.ITERATE,
            champion_model="popularity",
            challenger_model="popularity",
            reasons=["Offline evidence only."],
            guardrails={"latency": True},
        ),
        monitoring=monitoring,
        narrative=Narrative(
            headline="Iterate",
            conclusion=conclusion,
            tradeoffs=[],
            limitations=["Not a production monitor."],
            next_experiment="Run another temporal replay.",
            provider="deterministic",
        ),
    )


def test_monitoring_replay_is_temporal_and_labels_simulated_incident() -> None:
    train, later = _frames()
    model = PopularityModel().fit(train)
    windows = MonitoringReplay(k=3, latency_limit_ms=50).run(
        model,
        train,
        later,
        reference_metrics={"ndcg_at_3": 0.1},
        window_count=3,
        inject_incident=True,
    )
    assert len(windows) == 4
    assert all(windows[index].end_time <= windows[index + 1].start_time for index in range(2))
    assert {"event_volume", "event_type_js", "item_popularity_js"}.issubset(windows[0].metrics)
    assert windows[-1].simulated_incident
    assert windows[-1].status == HealthStatus.CRITICAL
    assert any("SIMULATED" in alert for alert in windows[-1].alerts)


def test_presentation_bundle_round_trips_and_rejects_sensitive_paths(tmp_path: Path) -> None:
    train, later = _frames()
    model = PopularityModel().fit(train)
    monitoring = MonitoringReplay(k=3).run(model, train, later, {"ndcg_at_3": 0.1}, window_count=2)
    path = tmp_path / "bundle.json"
    write_bundle(_bundle(monitoring), path)
    restored = load_bundle(path)
    assert restored.schema_version == "1.0"
    assert restored.monitoring
    assert path.stat().st_size < 1_000_000

    unsafe = _bundle(monitoring, conclusion="Loaded /Users/example/private.json")
    with pytest.raises(ValueError, match="unsafe presentation artifact"):
        write_bundle(unsafe, tmp_path / "unsafe.json")
