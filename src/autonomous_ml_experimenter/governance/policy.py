"""Guardrailed champion/challenger decision policy."""

from __future__ import annotations

import math

from autonomous_ml_experimenter.config import GovernanceConfig
from autonomous_ml_experimenter.core.contracts import (
    DecisionRecord,
    DecisionStatus,
    TrialResult,
)


class GuardrailedDecisionPolicy:
    """Recommend lifecycle actions from evidence; never mutate the registry."""

    def __init__(self, config: GovernanceConfig) -> None:
        self.config = config

    def decide(self, champion: TrialResult, challenger: TrialResult) -> DecisionRecord:
        required = {
            self.config.primary_metric,
            "catalog_coverage",
            "p95_latency_ms",
        }
        cold_key = _cold_recall_key(champion, challenger)
        if cold_key:
            required.add(cold_key)
        missing = sorted(
            metric
            for metric in required
            if metric not in champion.metrics or metric not in challenger.metrics
        )
        invalid = sorted(
            metric
            for metric in required.difference(missing)
            if not math.isfinite(champion.metrics[metric])
            or not math.isfinite(challenger.metrics[metric])
        )
        if missing or invalid or cold_key is None:
            evidence_issues = []
            if missing:
                evidence_issues.append(f"Missing required evidence: {', '.join(missing)}.")
            if invalid:
                evidence_issues.append(f"Invalid required evidence: {', '.join(invalid)}.")
            if cold_key is None:
                evidence_issues.append("Missing compatible cold-start recall evidence.")
            return DecisionRecord(
                status=DecisionStatus.INCONCLUSIVE,
                champion_model=champion.model_name,
                challenger_model=challenger.model_name,
                reasons=evidence_issues,
                guardrails={},
            )

        champion_score = champion.metrics[self.config.primary_metric]
        challenger_score = challenger.metrics[self.config.primary_metric]
        relative_lift = _relative_lift(champion_score, challenger_score)
        guardrails = {
            "primary_metric": relative_lift >= self.config.minimum_relative_improvement,
            "catalog_coverage": (
                challenger.metrics["catalog_coverage"] >= self.config.minimum_catalog_coverage
            ),
            "p95_latency": (
                challenger.metrics["p95_latency_ms"] <= self.config.maximum_p95_latency_ms
            ),
            "cold_start": (
                challenger.metrics[cold_key]
                >= champion.metrics[cold_key] - self.config.maximum_cold_recall_degradation
            ),
        }
        reasons = [
            f"{self.config.primary_metric} was measured for champion and challenger.",
            "Primary metric cleared the configured lift threshold."
            if guardrails["primary_metric"]
            else "Primary metric did not clear the configured lift threshold.",
        ]
        failed = [name for name, passed in guardrails.items() if not passed]
        if failed:
            reasons.append(f"Failed guardrails: {', '.join(failed)}.")
        else:
            reasons.append("All quality and operational guardrails passed.")

        if guardrails["primary_metric"] and all(guardrails.values()):
            status = DecisionStatus.PROMOTE
        elif challenger_score > champion_score:
            status = DecisionStatus.ITERATE
        else:
            status = DecisionStatus.REJECT
        return DecisionRecord(
            status=status,
            champion_model=champion.model_name,
            challenger_model=challenger.model_name,
            reasons=reasons,
            guardrails=guardrails,
        )


def _relative_lift(champion: float, challenger: float) -> float:
    if champion == 0:
        return float("inf") if challenger > 0 else 0.0
    return (challenger - champion) / abs(champion)


def _cold_recall_key(champion: TrialResult, challenger: TrialResult) -> str | None:
    champion_keys = {name for name in champion.metrics if name.startswith("cold_recall_at_")}
    challenger_keys = {name for name in challenger.metrics if name.startswith("cold_recall_at_")}
    shared = sorted(champion_keys.intersection(challenger_keys))
    return shared[0] if shared else None
