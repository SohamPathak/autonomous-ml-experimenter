"""Pure dashboard transformations over the exported presentation bundle.

The producer validates this artifact with Pydantic before export
(`autonomous_ml_experimenter.presentation.bundle`). The dashboard deliberately reads
the versioned JSON contract using only libraries that ship with Streamlit, so the
hosted app starts even when no extra dependencies are installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
REQUIRED_KEYS = ("schema_version", "data_manifest", "trials", "decision", "monitoring", "narrative")


def load_dashboard_bundle(path: Path) -> dict[str, Any]:
    """Load and shape-check an exported presentation bundle."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"presentation bundle is missing keys: {missing}")
    version = str(payload["schema_version"])
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported presentation bundle schema: {version}")
    if not payload["trials"] or not payload["monitoring"]:
        raise ValueError("presentation bundle has no trials or monitoring windows")
    return payload


def primary_metric(bundle: dict[str, Any]) -> str:
    """Resolve the ranking metric present across the recorded trials."""
    metrics = {name for trial in bundle["trials"] for name in trial["metrics"]}
    candidates = sorted(str(name) for name in metrics if str(name).startswith("ndcg_at_"))
    if not candidates:
        raise ValueError("presentation bundle has no NDCG metric")
    return candidates[0]


def trial_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    """Flatten every trial into an ordered, chart-ready frame."""
    metric = primary_metric(bundle)
    rows: list[dict[str, Any]] = []
    for order, trial in enumerate(bundle["trials"]):
        metrics = trial["metrics"]
        hypothesis = trial["hypothesis"]
        rows.append(
            {
                "order": order,
                "trial_id": trial["trial_id"],
                "phase": trial["phase"],
                "model": trial["model_name"],
                "outcome": trial["outcome"],
                metric: float(metrics.get(metric, 0.0)),
                "catalog_coverage": float(metrics.get("catalog_coverage", 0.0)),
                "p95_latency_ms": float(metrics.get("p95_latency_ms", 0.0)),
                "hypothesis": hypothesis["hypothesis"],
                "expected_effect": hypothesis["expected_effect"],
                "parent_trial_id": hypothesis.get("parent_trial_id") or "—",
                "run_id": trial.get("run_id") or "local-export",
                "model_version": trial.get("model_version") or "candidate",
                "evaluation_split": trial.get("evaluation_split", "validation"),
            }
        )
    return pd.DataFrame(rows)


def model_summary(bundle: dict[str, Any]) -> pd.DataFrame:
    """Return the best validation trial per model family."""
    frame = trial_frame(bundle)
    frame = frame[frame["evaluation_split"] == "validation"]
    metric = primary_metric(bundle)
    best_indices = frame.groupby("model")[metric].idxmax()
    result = frame.loc[
        best_indices,
        ["model", metric, "catalog_coverage", "p95_latency_ms", "trial_id", "outcome"],
    ]
    return result.sort_values(metric, ascending=False, ignore_index=True)


def monitoring_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    """Flatten monitoring windows, keeping simulated incidents explicit."""
    rows: list[dict[str, Any]] = []
    for window in bundle["monitoring"]:
        rows.append(
            {
                "window": window["window_id"],
                "start": window["start_time"],
                "end": window["end_time"],
                "status": window["status"],
                "alerts": " ".join(window.get("alerts", [])) or "No alerts",
                "simulated_incident": bool(window.get("simulated_incident", False)),
                **{key: float(value) for key, value in window["metrics"].items()},
            }
        )
    return pd.DataFrame(rows)


def selected_trials(bundle: dict[str, Any]) -> tuple[pd.Series[Any], pd.Series[Any]]:
    """Return the champion and challenger rows behind the recorded decision."""
    decision = bundle["decision"]
    frame = trial_frame(bundle)
    final = frame[frame["evaluation_split"] == "test"].set_index("model")
    source = final if not final.empty else model_summary(bundle).set_index("model")
    champion = source[source.index == decision["champion_model"]].iloc[0]
    challenger = source[source.index == decision["challenger_model"]].iloc[0]
    return champion, challenger
