"""Pure dashboard transformations over a validated presentation bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_ml_experimenter.core.contracts import PresentationBundle
from autonomous_ml_experimenter.presentation.bundle import load_bundle


def load_dashboard_bundle(path: Path) -> PresentationBundle:
    return load_bundle(path)


def primary_metric(bundle: PresentationBundle) -> str:
    metrics = {name for trial in bundle.trials for name in trial.metrics}
    candidates = sorted(name for name in metrics if name.startswith("ndcg_at_"))
    if not candidates:
        raise ValueError("presentation bundle has no NDCG metric")
    return candidates[0]


def trial_frame(bundle: PresentationBundle) -> pd.DataFrame:
    metric = primary_metric(bundle)
    rows: list[dict[str, Any]] = []
    for order, trial in enumerate(bundle.trials):
        rows.append(
            {
                "order": order,
                "trial_id": trial.trial_id,
                "phase": trial.phase,
                "model": trial.model_name,
                "outcome": trial.outcome.value,
                metric: trial.metrics.get(metric, 0.0),
                "catalog_coverage": trial.metrics.get("catalog_coverage", 0.0),
                "p95_latency_ms": trial.metrics.get("p95_latency_ms", 0.0),
                "hypothesis": trial.hypothesis.hypothesis,
                "expected_effect": trial.hypothesis.expected_effect,
                "parameters": trial.parameters,
                "parent_trial_id": trial.hypothesis.parent_trial_id or "—",
                "run_id": trial.run_id or "local-export",
                "model_version": trial.model_version or "candidate",
                "evaluation_split": trial.evaluation_split,
            }
        )
    return pd.DataFrame(rows)


def model_summary(bundle: PresentationBundle) -> pd.DataFrame:
    frame = trial_frame(bundle)
    frame = frame[frame["evaluation_split"] == "validation"]
    metric = primary_metric(bundle)
    best_indices = frame.groupby("model")[metric].idxmax()
    result = frame.loc[
        best_indices,
        ["model", metric, "catalog_coverage", "p95_latency_ms", "trial_id", "outcome"],
    ]
    return result.sort_values(metric, ascending=False, ignore_index=True)


def monitoring_frame(bundle: PresentationBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in bundle.monitoring:
        rows.append(
            {
                "window": window.window_id,
                "start": window.start_time,
                "end": window.end_time,
                "status": window.status.value,
                "alerts": " ".join(window.alerts) or "No alerts",
                "simulated_incident": window.simulated_incident,
                **window.metrics,
            }
        )
    return pd.DataFrame(rows)


def selected_trials(bundle: PresentationBundle) -> tuple[pd.Series[Any], pd.Series[Any]]:
    frame = trial_frame(bundle)
    final = frame[frame["evaluation_split"] == "test"].set_index("model")
    source = final if not final.empty else model_summary(bundle).set_index("model")
    champion = source[source.index == bundle.decision.champion_model].iloc[0]
    challenger = source[source.index == bundle.decision.challenger_model].iloc[0]
    return champion, challenger
