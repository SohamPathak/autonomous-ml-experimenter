"""Historical delayed-label monitoring with an explicitly simulated incident."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from autonomous_ml_experimenter.core.contracts import HealthStatus, MonitoringWindow
from autonomous_ml_experimenter.demo.recommendation.evaluation import evaluate_recommender


class MonitoringReplay:
    """Replay a fixed model on untouched later windows; never imply live telemetry."""

    def __init__(self, k: int = 10, latency_limit_ms: float = 50.0) -> None:
        self.k = k
        self.latency_limit_ms = latency_limit_ms

    def run(
        self,
        model: Any,
        train: pd.DataFrame,
        later: pd.DataFrame,
        reference_metrics: dict[str, float],
        window_count: int = 4,
        inject_incident: bool = False,
    ) -> list[MonitoringWindow]:
        if later.empty:
            raise ValueError("monitoring replay requires later observations")
        ordered = later.sort_values("timestamp", kind="stable")
        unique_times = ordered["timestamp"].drop_duplicates().to_numpy()
        partitions = [part for part in np.array_split(unique_times, window_count) if len(part)]
        windows: list[MonitoringWindow] = []
        for index, partition in enumerate(partitions, start=1):
            start_time = pd.Timestamp(partition[0])
            end_time = pd.Timestamp(partition[-1])
            current = ordered[
                (ordered["timestamp"] >= start_time) & (ordered["timestamp"] <= end_time)
            ].copy()
            metrics = evaluate_recommender(model, train, current, self.k)
            metrics.update(self._distribution_metrics(train, current))
            metrics["event_volume"] = float(len(current))
            metrics["missing_required_rate"] = float(
                current[["timestamp", "visitorid", "event", "itemid"]].isna().mean().mean()
            )
            metrics["window_duration_hours"] = max(
                0.0, (end_time - start_time).total_seconds() / 3600
            )
            status, alerts = self._health(metrics, reference_metrics)
            windows.append(
                MonitoringWindow(
                    window_id=f"observed-{index}",
                    start_time=start_time.to_pydatetime(),
                    end_time=end_time.to_pydatetime(),
                    metrics=metrics,
                    status=status,
                    alerts=alerts,
                )
            )
        if inject_incident:
            windows.append(self._simulated_latency_incident(windows[-1]))
        return windows

    def _distribution_metrics(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> dict[str, float]:
        reference_event = reference["event"].value_counts(normalize=True)
        current_event = current["event"].value_counts(normalize=True)
        reference_items = reference.groupby("itemid")["weight"].sum()
        current_items = current.groupby("itemid")["weight"].sum()
        reference_activity = float(reference.groupby("visitorid").size().mean())
        current_activity = float(current.groupby("visitorid").size().mean())
        activity_shift = abs(current_activity - reference_activity) / max(reference_activity, 1.0)
        return {
            "event_type_js": _jensen_shannon(reference_event, current_event),
            "item_popularity_js": _jensen_shannon(reference_items, current_items),
            "user_activity_relative_shift": activity_shift,
        }

    def _health(
        self, metrics: dict[str, float], reference_metrics: dict[str, float]
    ) -> tuple[HealthStatus, list[str]]:
        warnings: list[str] = []
        critical: list[str] = []
        if metrics["event_type_js"] > 0.25:
            critical.append("Event-type distribution changed materially.")
        elif metrics["event_type_js"] > 0.10:
            warnings.append("Event-type distribution is drifting.")
        if metrics["item_popularity_js"] > 0.35:
            critical.append("Item-popularity distribution changed materially.")
        elif metrics["item_popularity_js"] > 0.15:
            warnings.append("Item-popularity distribution is drifting.")
        if metrics.get("unknown_item_rate", 0.0) > 0.35:
            critical.append("Unknown-item rate breached the critical threshold.")
        elif metrics.get("unknown_item_rate", 0.0) > 0.15:
            warnings.append("Unknown-item rate breached the warning threshold.")
        if metrics.get("p95_latency_ms", 0.0) > self.latency_limit_ms:
            critical.append("p95 inference latency exceeded the serving guardrail.")
        for metric_name, reference in reference_metrics.items():
            if metric_name not in metrics or reference <= 0:
                continue
            ratio = metrics[metric_name] / reference
            if ratio < 0.5:
                critical.append(f"{metric_name} fell materially below the reference window.")
            elif ratio < 0.8:
                warnings.append(f"{metric_name} is below the reference window.")
        if critical:
            return HealthStatus.CRITICAL, critical + warnings
        if warnings:
            return HealthStatus.WARNING, warnings
        return HealthStatus.HEALTHY, []

    def _simulated_latency_incident(self, source: MonitoringWindow) -> MonitoringWindow:
        metrics = source.metrics.copy()
        metrics["p95_latency_ms"] = self.latency_limit_ms * 2.5
        start = source.end_time + timedelta(seconds=1)
        return MonitoringWindow(
            window_id="simulated-latency-incident",
            start_time=start,
            end_time=start + timedelta(hours=1),
            metrics=metrics,
            status=HealthStatus.CRITICAL,
            alerts=["SIMULATED INCIDENT — p95 latency exceeded the serving guardrail."],
            simulated_incident=True,
        )


def _jensen_shannon(left: pd.Series[Any], right: pd.Series[Any]) -> float:
    labels = left.index.union(right.index)
    left_values = left.reindex(labels, fill_value=0).to_numpy(dtype=float)
    right_values = right.reindex(labels, fill_value=0).to_numpy(dtype=float)
    left_values = left_values / left_values.sum() if left_values.sum() else left_values
    right_values = right_values / right_values.sum() if right_values.sum() else right_values
    midpoint = (left_values + right_values) / 2

    def divergence(values: np.ndarray[Any, Any]) -> float:
        mask = (values > 0) & (midpoint > 0)
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * divergence(left_values) + 0.5 * divergence(right_values)
