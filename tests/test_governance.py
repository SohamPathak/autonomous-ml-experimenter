from __future__ import annotations

from pathlib import Path

from mlflow import MlflowClient

from autonomous_ml_experimenter.config import GovernanceConfig
from autonomous_ml_experimenter.core.contracts import (
    DecisionStatus,
    Hypothesis,
    TrialOutcome,
    TrialResult,
)
from autonomous_ml_experimenter.governance.policy import GuardrailedDecisionPolicy
from autonomous_ml_experimenter.tracking.mlflow_tracker import MLflowTracker


def _trial(
    name: str,
    ndcg: float,
    coverage: float = 0.25,
    latency: float = 10.0,
    cold_recall: float = 0.10,
) -> TrialResult:
    return TrialResult(
        trial_id=name,
        phase="test",
        model_name=name,
        parameters={},
        metrics={
            "ndcg_at_10": ndcg,
            "catalog_coverage": coverage,
            "p95_latency_ms": latency,
            "cold_recall_at_10": cold_recall,
        },
        hypothesis=Hypothesis(
            observation="Measured candidate.",
            hypothesis="Candidate may improve ranking.",
            expected_effect="Improve quality within guardrails.",
            parameter_changes={},
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=1,
    )


def _policy() -> GuardrailedDecisionPolicy:
    return GuardrailedDecisionPolicy(GovernanceConfig())


def test_policy_recommends_promote_only_when_all_guardrails_pass() -> None:
    decision = _policy().decide(_trial("champion", 0.10), _trial("challenger", 0.12))
    assert decision.status == DecisionStatus.PROMOTE
    assert all(decision.guardrails.values())
    assert decision.requires_human_approval
    assert not decision.approved


def test_policy_returns_iterate_for_promising_guardrail_failure() -> None:
    decision = _policy().decide(
        _trial("champion", 0.10),
        _trial("challenger", 0.12, coverage=0.01, latency=70),
    )
    assert decision.status == DecisionStatus.ITERATE
    assert not decision.guardrails["catalog_coverage"]
    assert not decision.guardrails["p95_latency"]


def test_policy_rejects_regression_and_marks_missing_evidence_inconclusive() -> None:
    rejected = _policy().decide(_trial("champion", 0.10), _trial("challenger", 0.09))
    assert rejected.status == DecisionStatus.REJECT

    incomplete = _trial("challenger", 0.12)
    incomplete.metrics.pop("p95_latency_ms")
    inconclusive = _policy().decide(_trial("champion", 0.10), incomplete)
    assert inconclusive.status == DecisionStatus.INCONCLUSIVE
    assert "missing" in " ".join(inconclusive.reasons).lower()


def test_registry_champion_changes_only_after_explicit_approval(tmp_path: Path) -> None:
    tracker = MLflowTracker(
        database_path=tmp_path / "tracking.db",
        artifact_root=tmp_path / "mlruns",
        experiment_name="approval-test",
        repository_root=Path.cwd(),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.json").write_text("{}", encoding="utf-8")
    with (
        tracker.study("V1"),
        tracker.trial("candidate", {}, _trial("candidate", 0.12).hypothesis) as run_id,
    ):
        tracker.log_model_directory(model_dir)
        version = tracker.register_candidate("approval-model", run_id)

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    assert client.get_model_version_by_alias("approval-model", "candidate")
    try:
        client.get_model_version_by_alias("approval-model", "champion")
        champion_exists = True
    except Exception:
        champion_exists = False
    assert not champion_exists

    tracker.approve_candidate("approval-model", version, approved_by="human-reviewer")
    champion = client.get_model_version_by_alias("approval-model", "champion")
    assert str(champion.version) == version
    assert champion.tags["validation_status"] == "approved"
    assert champion.tags["approved_by"] == "human-reviewer"
