from pathlib import Path

from mlflow import MlflowClient

from autonomous_ml_experimenter.core.contracts import Hypothesis, TrialOutcome, TrialResult
from autonomous_ml_experimenter.tracking.mlflow_tracker import MLflowTracker


def _result(run_id: str | None = None) -> TrialResult:
    return TrialResult(
        trial_id="trial-001",
        phase="V1 · Co-visitation",
        model_name="co_visitation",
        parameters={"session_minutes": 30},
        metrics={"ndcg_at_10": 0.21, "catalog_coverage": 0.42},
        hypothesis=Hypothesis(
            observation="Short sessions are noisy.",
            hypothesis="A 30-minute window will improve intent coherence.",
            expected_effect="Improve NDCG@10.",
            parameter_changes={"session_minutes": 30},
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=1.2,
        run_id=run_id,
    )


def test_nested_runs_capture_lineage_metrics_and_safe_metadata(tmp_path: Path) -> None:
    tracker = MLflowTracker(
        database_path=tmp_path / "tracking.db",
        artifact_root=tmp_path / "mlruns",
        experiment_name="tracking-test",
        repository_root=Path.cwd(),
    )
    with (
        tracker.study("V1", {"data_fingerprint": "abc", "credential_path": "/secret"}) as parent,
        tracker.trial(
            "trial-001",
            {"session_minutes": 30, "credential_path": "/Users/example/key.json"},
            _result().hypothesis,
        ) as child,
    ):
        tracker.log_result(_result(child))

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    parent_run = client.get_run(parent)
    child_run = client.get_run(child)
    assert child_run.data.tags["mlflow.parentRunId"] == parent
    assert child_run.data.metrics["ndcg_at_10"] == 0.21
    assert child_run.data.params["session_minutes"] == "30"
    assert child_run.data.params["credential_path"] == "[REDACTED]"
    assert parent_run.data.tags["credential_path"] == "[REDACTED]"
    assert child_run.data.tags["code.git_state"] in {"clean", "dirty", "unborn"}


def test_registry_version_has_candidate_alias_and_pending_review(tmp_path: Path) -> None:
    tracker = MLflowTracker(
        database_path=tmp_path / "tracking.db",
        artifact_root=tmp_path / "mlruns",
        experiment_name="registry-test",
        repository_root=Path.cwd(),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text('{"model":"demo"}', encoding="utf-8")

    with (
        tracker.study("V2"),
        tracker.trial("candidate", {}, _result().hypothesis) as run_id,
    ):
        tracker.log_model_directory(model_dir)
        version = tracker.register_candidate(
            registered_name="retailrocket-recommender",
            run_id=run_id,
            tags={"validation_status": "pending"},
        )

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    candidate = client.get_model_version_by_alias("retailrocket-recommender", "candidate")
    assert str(candidate.version) == version
    stored = client.get_model_version("retailrocket-recommender", version)
    assert stored.tags["validation_status"] == "pending"
    assert stored.run_id == run_id
