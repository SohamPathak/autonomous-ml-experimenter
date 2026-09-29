"""Command-line entry point."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from autonomous_ml_experimenter.ai.vertex import (
    DeterministicNarrator,
    VertexGateway,
    VertexNarrator,
)
from autonomous_ml_experimenter.config import load_config
from autonomous_ml_experimenter.core.contracts import (
    DecisionRecord,
    Hypothesis,
    Narrative,
    PresentationBundle,
    TrialOutcome,
    TrialResult,
)
from autonomous_ml_experimenter.core.security import redact
from autonomous_ml_experimenter.demo.recommendation.data import (
    load_interactions,
    prepare_interactions,
)
from autonomous_ml_experimenter.demo.recommendation.evaluation import evaluate_recommender
from autonomous_ml_experimenter.demo.recommendation.fixtures import generate_interactions
from autonomous_ml_experimenter.demo.recommendation.models import (
    ALSModel,
    CoVisitationModel,
    PopularityModel,
)
from autonomous_ml_experimenter.demo.recommendation.task import RecommendationTask
from autonomous_ml_experimenter.governance.policy import GuardrailedDecisionPolicy
from autonomous_ml_experimenter.monitoring.replay import MonitoringReplay
from autonomous_ml_experimenter.pipeline import run_pipeline
from autonomous_ml_experimenter.presentation.bundle import write_bundle
from autonomous_ml_experimenter.research.loop import BoundedResearchLoop
from autonomous_ml_experimenter.tracking.mlflow_tracker import MLflowTracker

app = typer.Typer(
    help="Bounded, model-agnostic autonomous ML experimentation.", no_args_is_help=True
)


@app.callback()
def main() -> None:
    """Run governed machine-learning research loops."""


@app.command()
def validate(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
) -> None:
    """Validate configuration and the deterministic demo task."""
    settings = load_config(config)
    interactions = generate_interactions(settings.data.sample_users or 120, seed=settings.seed)
    safe = redact(settings.model_dump(mode="json"))
    typer.echo(f"Configuration valid (schema {safe['schema_version']}).")
    typer.echo(
        f"Synthetic task valid: {len(interactions):,} events, "
        f"{interactions['visitorid'].nunique()} users, {interactions['itemid'].nunique()} items."
    )
    typer.echo("Credential material is loaded only through runtime identity; none was inspected.")


@app.command("data-report")
def data_report(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/data_report.json"),
) -> None:
    """Prepare temporal splits and write an auditable data-quality report."""
    settings = load_config(config)
    frame = load_interactions(settings.data, settings.seed)
    splits = prepare_interactions(frame, settings.data, settings.seed)
    report = {
        "manifest": splits.manifest.model_dump(mode="json"),
        "quality": splits.quality_report,
        "cohorts": splits.cohort_report,
        "boundaries": {
            "train_end": splits.train["timestamp"].max().isoformat(),
            "validation_start": splits.validation["timestamp"].min().isoformat(),
            "validation_end": splits.validation["timestamp"].max().isoformat(),
            "test_start": splits.test["timestamp"].min().isoformat(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Data report written to {output} ({splits.manifest.fingerprint[:12]}…).")


@app.command()
def baseline(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/baseline_result.json"),
) -> None:
    """Train and evaluate the auditable popularity baseline."""
    settings = load_config(config)
    frame = load_interactions(settings.data, settings.seed)
    splits = prepare_interactions(frame, settings.data, settings.seed)
    started = time.perf_counter()
    model = PopularityModel().fit(splits.train)
    metrics = evaluate_recommender(model, splits.train, splits.validation, settings.top_k)
    result = TrialResult(
        trial_id=f"baseline-{uuid4().hex[:8]}",
        phase="V0 · Baseline",
        model_name=model.name,
        parameters={"event_weighting": "fixed"},
        metrics=metrics,
        hypothesis=Hypothesis(
            observation="No evaluated model exists.",
            hypothesis="Weighted popularity establishes a production fallback and quality floor.",
            expected_effect="Create a deterministic baseline for personalized challengers.",
            parameter_changes={"model": "popularity"},
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=time.perf_counter() - started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    metric = metrics[f"ndcg_at_{settings.top_k}"]
    typer.echo(f"Baseline evaluated: NDCG@{settings.top_k}={metric:.4f}; result: {output}")


@app.command()
def compare(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/model_comparison.json"),
) -> None:
    """Train and compare all demonstration model adapters."""
    settings = load_config(config)
    splits = prepare_interactions(
        load_interactions(settings.data, settings.seed), settings.data, settings.seed
    )
    models: list[Any] = [
        PopularityModel(),
        CoVisitationModel(session_minutes=30, recency_decay=0.9, neighbor_limit=50),
        ALSModel(factors=16, regularization=0.05, iterations=10, alpha=20, seed=settings.seed),
    ]
    comparison: list[dict[str, Any]] = []
    for model in models:
        started = time.perf_counter()
        model.fit(splits.train)
        metrics = evaluate_recommender(model, splits.train, splits.validation, settings.top_k)
        comparison.append(
            {
                "model": model.name,
                "metrics": metrics,
                "duration_seconds": time.perf_counter() - started,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    metric_name = f"ndcg_at_{settings.top_k}"
    winner = max(comparison, key=lambda row: float(row["metrics"][metric_name]))
    typer.echo(
        f"Compared {len(comparison)} models; best validation {metric_name}="
        f"{float(winner['metrics'][metric_name]):.4f} ({winner['model']})."
    )


@app.command("track-baseline")
def track_baseline(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
) -> None:
    """Track and register a baseline with full local lineage."""
    settings = load_config(config)
    splits = prepare_interactions(
        load_interactions(settings.data, settings.seed), settings.data, settings.seed
    )
    model = PopularityModel().fit(splits.train)
    metrics = evaluate_recommender(model, splits.train, splits.validation, settings.top_k)
    hypothesis = Hypothesis(
        observation="No registered model exists.",
        hypothesis="Weighted popularity provides an auditable fallback baseline.",
        expected_effect="Establish a registry candidate and comparison floor.",
        parameter_changes={"model": "popularity"},
    )
    tracker = MLflowTracker(
        settings.tracking.database_path,
        settings.tracking.artifact_root,
        "retailrocket-recommendation",
        Path.cwd(),
    )
    model_dir = Path("artifacts/models/popularity-v0")
    model.save(model_dir / "model.json")
    with (
        tracker.study("V0 · Baseline", {"data_fingerprint": splits.manifest.fingerprint}),
        tracker.trial("baseline", {"event_weighting": "fixed"}, hypothesis) as run_id,
    ):
        result = TrialResult(
            trial_id="baseline",
            phase="V0 · Baseline",
            model_name=model.name,
            parameters={"event_weighting": "fixed"},
            metrics=metrics,
            hypothesis=hypothesis,
            outcome=TrialOutcome.IMPROVEMENT,
            duration_seconds=0.0,
            run_id=run_id,
        )
        tracker.log_result(result)
        tracker.log_model_directory(model_dir)
        version = tracker.register_candidate(
            "retailrocket-recommender",
            run_id,
            {"validation_status": "pending", "data_fingerprint": splits.manifest.fingerprint},
        )
    typer.echo(f"Registered candidate version {version}; run {run_id[:8]}…")


@app.command()
def research(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/research_history.json"),
) -> None:
    """Run bounded co-visitation and ALS research phases."""
    settings = load_config(config)
    task = RecommendationTask(settings)
    train, validation, _ = task.create_splits(task.load_data())
    metric_name = f"ndcg_at_{settings.top_k}"
    baseline_model = PopularityModel().fit(train)
    baseline_metrics = task.evaluate(baseline_model, validation)
    baseline_result = TrialResult(
        trial_id="v0-000",
        phase="V0 · Baseline",
        model_name="popularity",
        parameters={},
        metrics=baseline_metrics,
        hypothesis=Hypothesis(
            observation="No evaluated model exists.",
            hypothesis="Establish a weighted popularity quality floor.",
            expected_effect="Create a deterministic fallback and comparison point.",
            parameter_changes={"model": "popularity"},
        ),
        outcome=TrialOutcome.IMPROVEMENT,
        duration_seconds=0.0,
    )
    tracker = (
        MLflowTracker(
            settings.tracking.database_path,
            settings.tracking.artifact_root,
            task.name,
            Path.cwd(),
        )
        if settings.tracking.enabled
        else None
    )
    loop = BoundedResearchLoop(task, settings.research, metric_name, tracker=tracker)
    co_visitation = loop.run_phase(
        "co_visitation", train, validation, "V1 · Co-visitation", [baseline_result]
    )
    als = loop.run_phase(
        "implicit_als",
        train,
        validation,
        "V2 · Implicit ALS",
        [baseline_result, *co_visitation],
    )
    history = [baseline_result, *co_visitation, *als]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([result.model_dump(mode="json") for result in history], indent=2),
        encoding="utf-8",
    )
    best = max(history, key=lambda result: result.metrics[metric_name])
    typer.echo(
        f"Research completed: {len(history)} trials; best {metric_name}="
        f"{best.metrics[metric_name]:.4f} ({best.model_name})."
    )


@app.command()
def decide(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    history_path: Annotated[Path, typer.Option(exists=True)] = Path(
        "artifacts/research_history.json"
    ),
    output: Annotated[Path, typer.Option()] = Path("artifacts/decision_report.json"),
    use_vertex: Annotated[bool, typer.Option("--vertex/--deterministic")] = False,
) -> None:
    """Apply deterministic governance and generate an evidence-grounded narrative."""
    settings = load_config(config)
    raw_history: list[dict[str, Any]] = json.loads(history_path.read_text(encoding="utf-8"))
    history = [TrialResult.model_validate(item) for item in raw_history]
    champion = next(result for result in history if result.model_name == "popularity")
    candidates = [result for result in history if result.model_name != "popularity"]
    challenger = max(
        candidates, key=lambda result: result.metrics[settings.governance.primary_metric]
    )
    decision = GuardrailedDecisionPolicy(settings.governance).decide(champion, challenger)
    narrator: Any
    if use_vertex:
        narrator = VertexNarrator(VertexGateway(settings.vertex.model, settings.vertex.location))
    else:
        narrator = DeterministicNarrator()
    narrative = narrator.narrate(decision, history)
    payload = {
        "decision": decision.model_dump(mode="json"),
        "narrative": narrative.model_dump(mode="json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(
        f"Decision: {decision.status.value.upper()} {challenger.model_name}; "
        f"narrative={narrative.provider}; human approval required."
    )


@app.command()
def bundle(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    history_path: Annotated[Path, typer.Option(exists=True)] = Path(
        "artifacts/research_history.json"
    ),
    decision_path: Annotated[Path, typer.Option(exists=True)] = Path(
        "artifacts/decision_report.json"
    ),
    output: Annotated[Path, typer.Option()] = Path("app/data/demo_bundle.json"),
) -> None:
    """Replay monitoring and export the secret-scanned cloud presentation bundle."""
    settings = load_config(config)
    task = RecommendationTask(settings)
    train, _, test = task.create_splits(task.load_data())
    raw_history: list[dict[str, Any]] = json.loads(history_path.read_text(encoding="utf-8"))
    history = [TrialResult.model_validate(item) for item in raw_history]
    report: dict[str, Any] = json.loads(decision_path.read_text(encoding="utf-8"))
    decision = DecisionRecord.model_validate(report["decision"])
    narrative = Narrative.model_validate(report["narrative"])
    candidates = [result for result in history if result.model_name == decision.challenger_model]
    selected = max(
        candidates, key=lambda result: result.metrics[settings.governance.primary_metric]
    )
    model = task.build_model(selected.model_name, selected.parameters).fit(train)
    monitoring = MonitoringReplay(settings.top_k, settings.governance.maximum_p95_latency_ms).run(
        model,
        train,
        test,
        {settings.governance.primary_metric: selected.metrics[settings.governance.primary_metric]},
        window_count=4,
        inject_incident=True,
    )
    if task.splits is None:
        raise RuntimeError("recommendation splits were not prepared")
    presentation = PresentationBundle(
        data_manifest=task.splits.manifest,
        trials=history,
        decision=decision,
        monitoring=monitoring,
        narrative=narrative,
        source_run_ids=[result.run_id for result in history if result.run_id],
    )
    write_bundle(presentation, output)
    typer.echo(
        f"Presentation bundle written to {output}: {len(history)} trials, "
        f"{len(monitoring)} monitoring windows."
    )


@app.command("run")
def run_end_to_end(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/demo.yaml"),
    output: Annotated[Path, typer.Option()] = Path("app/data/demo_bundle.json"),
    offline: Annotated[bool, typer.Option("--offline")] = False,
) -> None:
    """Run the complete governed research workflow and export the dashboard bundle."""
    settings = load_config(config)
    result = run_pipeline(settings, output, Path.cwd(), use_vertex=False if offline else None)
    typer.echo(
        f"End-to-end run complete: {len(result.trials)} trials; "
        f"decision={result.decision.status.value}; narrative={result.narrative.provider}."
    )


if __name__ == "__main__":
    app()
