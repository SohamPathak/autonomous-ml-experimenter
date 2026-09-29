"""End-to-end orchestration for the recommendation demonstration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_ml_experimenter.ai.vertex import (
    DeterministicNarrator,
    VertexGateway,
    VertexNarrator,
    VertexResearchPlanner,
)
from autonomous_ml_experimenter.config import AppConfig
from autonomous_ml_experimenter.core.contracts import (
    Hypothesis,
    PresentationBundle,
    TrialOutcome,
    TrialResult,
)
from autonomous_ml_experimenter.demo.recommendation.evaluation import evaluate_recommender
from autonomous_ml_experimenter.demo.recommendation.task import RecommendationTask
from autonomous_ml_experimenter.governance.policy import GuardrailedDecisionPolicy
from autonomous_ml_experimenter.monitoring.replay import MonitoringReplay
from autonomous_ml_experimenter.presentation.bundle import write_bundle
from autonomous_ml_experimenter.research.loop import BoundedResearchLoop
from autonomous_ml_experimenter.tracking.mlflow_tracker import MLflowTracker


def run_pipeline(
    config: AppConfig,
    output: Path,
    repository_root: Path,
    use_vertex: bool | None = None,
) -> PresentationBundle:
    """Run research, final evaluation, governance, monitoring, and cloud export."""
    task = RecommendationTask(config)
    train, validation, test = task.create_splits(task.load_data())
    metric_name = config.governance.primary_metric
    vertex_enabled = config.vertex.enabled if use_vertex is None else use_vertex
    gateway = VertexGateway(config.vertex.model, config.vertex.location) if vertex_enabled else None
    tracker = _build_tracker(config, task.name, repository_root)

    baseline_validation = _evaluate_trial(
        task,
        "popularity",
        {},
        train,
        validation,
        "v0-000",
        "V0 · Baseline",
        "Establish a weighted popularity fallback and quality floor.",
    )
    if tracker is not None:
        baseline_validation = _track_standalone_trial(
            tracker, baseline_validation, task.splits.manifest.fingerprint if task.splits else ""
        )

    planner = VertexResearchPlanner(gateway) if gateway is not None else None
    loop = BoundedResearchLoop(
        task,
        config.research,
        metric_name,
        tracker=tracker,
        planner=planner,
    )
    co_visitation = loop.run_phase(
        "co_visitation",
        train,
        validation,
        "V1 · Co-visitation",
        [baseline_validation],
    )
    als = loop.run_phase(
        "implicit_als",
        train,
        validation,
        "V2 · Implicit ALS",
        [baseline_validation, *co_visitation],
    )
    research_history = [baseline_validation, *co_visitation, *als]
    candidate_validation = max(
        [trial for trial in research_history if trial.model_name != "popularity"],
        key=lambda trial: trial.metrics[metric_name],
    )

    final_test, monitoring_later = _split_final_and_monitoring(test)
    development = pd.concat([train, validation], ignore_index=True).sort_values("timestamp")
    final_baseline = _evaluate_trial(
        task,
        "popularity",
        {},
        development,
        final_test,
        "final-popularity",
        "Final · Untouched test",
        "Lock the baseline and evaluate it once on the untouched final test slice.",
        evaluation_split="test",
    )
    final_candidate = _evaluate_trial(
        task,
        candidate_validation.model_name,
        candidate_validation.parameters,
        development,
        final_test,
        f"final-{candidate_validation.model_name}",
        "Final · Untouched test",
        "Lock the selected configuration and evaluate it once on the untouched final test slice.",
        evaluation_split="test",
        parent_trial_id=candidate_validation.trial_id,
    )
    selected_model = task.build_model(
        candidate_validation.model_name, candidate_validation.parameters
    ).fit(development)
    if tracker is not None:
        final_candidate = _register_final_candidate(
            tracker,
            final_candidate,
            selected_model,
            repository_root,
            task.splits.manifest.fingerprint if task.splits else "",
        )

    decision = GuardrailedDecisionPolicy(config.governance).decide(final_baseline, final_candidate)
    all_trials = [*research_history, final_baseline, final_candidate]
    if gateway is not None:
        narrative = VertexNarrator(gateway).narrate(decision, all_trials)
    else:
        narrative = DeterministicNarrator().narrate(decision, all_trials)
    monitoring = MonitoringReplay(config.top_k, config.governance.maximum_p95_latency_ms).run(
        selected_model,
        development,
        monitoring_later,
        {metric_name: final_candidate.metrics[metric_name]},
        window_count=4,
        inject_incident=True,
    )
    if task.splits is None:
        raise RuntimeError("recommendation splits were not prepared")
    bundle = PresentationBundle(
        data_manifest=task.splits.manifest,
        trials=all_trials,
        decision=decision,
        monitoring=monitoring,
        narrative=narrative,
        source_run_ids=[trial.run_id for trial in all_trials if trial.run_id],
    )
    write_bundle(bundle, output)
    return bundle


def _evaluate_trial(
    task: RecommendationTask,
    model_name: str,
    parameters: dict[str, Any],
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    trial_id: str,
    phase: str,
    hypothesis_text: str,
    evaluation_split: str = "validation",
    parent_trial_id: str | None = None,
) -> TrialResult:
    started = time.perf_counter()
    model = task.build_model(model_name, parameters).fit(training)
    metrics = evaluate_recommender(model, training, evaluation, task.config.top_k)
    return TrialResult(
        trial_id=trial_id,
        phase=phase,
        model_name=model_name,
        parameters=parameters,
        metrics=metrics,
        hypothesis=Hypothesis(
            observation="The configuration and data boundary are locked before evaluation.",
            hypothesis=hypothesis_text,
            expected_effect=(
                "Produce auditable evidence without feeding this result back into search."
            ),
            parameter_changes=parameters,
            parent_trial_id=parent_trial_id,
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=time.perf_counter() - started,
        evaluation_split=evaluation_split,
    )


def _split_final_and_monitoring(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = test["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) < 4:
        raise ValueError("test data needs four distinct timestamps for final and monitoring slices")
    cutoff = times.iloc[len(times) // 2]
    final = test[test["timestamp"] < cutoff].copy()
    monitoring = test[test["timestamp"] >= cutoff].copy()
    if final.empty or monitoring.empty:
        raise ValueError("final/monitoring temporal split produced an empty partition")
    return final, monitoring


def _build_tracker(
    config: AppConfig, experiment_name: str, repository_root: Path
) -> MLflowTracker | None:
    if not config.tracking.enabled:
        return None
    return MLflowTracker(
        config.tracking.database_path,
        config.tracking.artifact_root,
        experiment_name,
        repository_root,
    )


def _track_standalone_trial(
    tracker: MLflowTracker, result: TrialResult, data_fingerprint: str
) -> TrialResult:
    with (
        tracker.study(result.phase, {"data_fingerprint": data_fingerprint}) as parent_id,
        tracker.trial(result.trial_id, result.parameters, result.hypothesis) as run_id,
    ):
        tracked = result.model_copy(update={"parent_run_id": parent_id, "run_id": run_id})
        tracker.log_result(tracked)
    return tracked


def _register_final_candidate(
    tracker: MLflowTracker,
    result: TrialResult,
    model: Any,
    repository_root: Path,
    data_fingerprint: str,
) -> TrialResult:
    artifact_dir = repository_root / "artifacts" / "models" / result.trial_id
    model.save(artifact_dir)
    with (
        tracker.study(result.phase, {"data_fingerprint": data_fingerprint}) as parent_id,
        tracker.trial(result.trial_id, result.parameters, result.hypothesis) as run_id,
    ):
        tracked = result.model_copy(update={"parent_run_id": parent_id, "run_id": run_id})
        tracker.log_result(tracked)
        tracker.log_model_directory(artifact_dir)
        version = tracker.register_candidate(
            "retailrocket-recommender",
            run_id,
            {"validation_status": "pending", "data_fingerprint": data_fingerprint},
        )
    return tracked.model_copy(update={"model_version": version})
