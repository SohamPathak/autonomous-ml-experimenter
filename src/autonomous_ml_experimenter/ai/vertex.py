"""Vertex AI adapters with strict schemas, grounding, and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

import google.auth
from google import genai
from google.genai import types
from pydantic import BaseModel

from autonomous_ml_experimenter.core.contracts import (
    DecisionRecord,
    Hypothesis,
    Narrative,
    TrialResult,
)
from autonomous_ml_experimenter.core.security import redact
from autonomous_ml_experimenter.research.planner import DeterministicResearchPlanner

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")


class VertexGateway:
    """Small Vertex Gemini gateway using Application Default Credentials only."""

    def __init__(
        self,
        model: str | None = None,
        location: str | None = None,
        project: str | None = None,
    ) -> None:
        self.model: str = model or os.getenv("VERTEX_MODEL") or "gemini-2.5-flash"
        self.location: str = location or os.getenv("VERTEX_LOCATION") or "global"
        self.project = project or os.getenv("VERTEX_PROJECT")

    def generate(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Generate one schema-valid response without logging credentials or paths."""
        credentials, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = self.project or detected_project
        if not project:
            raise RuntimeError("Vertex project was not resolved through ADC or VERTEX_PROJECT")
        client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=self.location,
            http_options=types.HttpOptions(timeout=30_000),
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                seed=42,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        if not response.text:
            raise RuntimeError("Vertex returned an empty structured response")
        return schema.model_validate_json(response.text)


class DeterministicNarrator:
    """Always-available evidence renderer with no external calls."""

    def narrate(self, decision: DecisionRecord, trials: list[TrialResult]) -> Narrative:
        champion = _select_model_trial(trials, decision.champion_model)
        challenger = _select_model_trial(trials, decision.challenger_model)
        metric = _primary_metric(champion, challenger)
        champion_value = champion.metrics.get(metric, 0.0)
        challenger_value = challenger.metrics.get(metric, 0.0)
        failed = [name for name, passed in decision.guardrails.items() if not passed]
        tradeoffs = decision.reasons.copy()
        if failed:
            tradeoffs.append(f"Failed guardrails: {', '.join(failed)}.")
        return Narrative(
            headline=f"Recommendation: {decision.status.value.upper()} {decision.challenger_model}",
            conclusion=(
                f"{decision.challenger_model} produced {metric}={challenger_value:.4f} versus "
                f"{decision.champion_model} at {champion_value:.4f}. The deterministic policy "
                f"returned {decision.status.value}."
            ),
            tradeoffs=tradeoffs,
            limitations=[
                "Evidence comes from leakage-safe offline replay, not a live randomized experiment."
            ],
            next_experiment=(
                "Replay the candidate on the next untouched temporal window before online exposure."
            ),
            provider="deterministic",
        )


class VertexNarrator:
    """Grounded Vertex narration with silent deterministic fallback."""

    def __init__(self, gateway: Any, fallback: DeterministicNarrator | None = None) -> None:
        self.gateway = gateway
        self.fallback = fallback or DeterministicNarrator()

    def narrate(self, decision: DecisionRecord, trials: list[TrialResult]) -> Narrative:
        evidence = {
            "decision": decision.model_dump(mode="json"),
            "trials": [trial.model_dump(mode="json") for trial in trials],
            "rules": [
                "Use only supplied evidence.",
                "Do not use significant/significantly or causal language.",
                "Describe PROMOTE only as a promotion recommendation pending human approval.",
                "Every number must occur in the evidence.",
            ],
        }
        prompt = (
            "Produce a concise executive experiment narrative from this redacted evidence:\n"
            + json.dumps(redact(evidence), sort_keys=True)
        )
        try:
            candidate = self.gateway.generate(prompt, Narrative)
            if not isinstance(candidate, Narrative):
                raise TypeError("Vertex response did not match Narrative")
            validate_claim_language(candidate)
            validate_numerical_grounding(candidate, trials)
            return candidate.model_copy(update={"provider": "vertex-ai"})
        except Exception:
            fallback = self.fallback.narrate(decision, trials)
            return fallback.model_copy(update={"provider": "deterministic-fallback"})


class VertexResearchPlanner:
    """Ask Vertex to explain a fixed optimizer proposal, never to execute code."""

    def __init__(
        self,
        gateway: Any,
        fallback: DeterministicResearchPlanner | None = None,
    ) -> None:
        self.gateway = gateway
        self.fallback = fallback or DeterministicResearchPlanner()

    def propose(
        self,
        history: list[TrialResult],
        phase: str,
        model_name: str,
        parameters: dict[str, Any],
    ) -> Hypothesis:
        fallback = self.fallback.propose(history, phase, model_name, parameters)
        evidence = {
            "phase": phase,
            "model_name": model_name,
            "fixed_optimizer_proposal": parameters,
            "recent_trials": [
                trial.model_dump(mode="json", exclude={"error"}) for trial in history[-8:]
            ],
            "rules": [
                "Explain only the fixed proposal; do not add or alter parameters.",
                "Do not propose code, shell commands, paths, secrets, or test-set access.",
            ],
        }
        try:
            candidate = self.gateway.generate(
                "Explain this bounded trial as a research hypothesis:\n"
                + json.dumps(redact(evidence), sort_keys=True),
                Hypothesis,
            )
            if not isinstance(candidate, Hypothesis):
                raise TypeError("Vertex response did not match Hypothesis")
            if candidate.parameter_changes != parameters:
                raise ValueError("Vertex attempted to alter the optimizer proposal")
            return candidate.model_copy(
                update={
                    "parameter_changes": parameters,
                    "parent_trial_id": history[-1].trial_id if history else None,
                }
            )
        except Exception:
            return fallback


def validate_claim_language(narrative: Narrative) -> None:
    """Reject causal, significance, or completed-promotion claims from offline evidence."""
    text = " ".join(
        [
            narrative.headline,
            narrative.conclusion,
            *narrative.tradeoffs,
            *narrative.limitations,
            narrative.next_experiment,
        ]
    ).lower()
    forbidden = (
        "significant",
        "significantly",
        "caused",
        "causal effect",
        "confidence",
        "leading to its promotion",
        "was promoted",
        "has been promoted",
    )
    found = [phrase for phrase in forbidden if phrase in text]
    if found:
        raise ValueError(f"unsupported offline claim language: {', '.join(found)}")


def validate_numerical_grounding(narrative: Narrative, trials: list[TrialResult]) -> None:
    """Reject numerical claims that cannot be matched to metric/parameter evidence."""
    allowed_values: list[float] = []
    allowed_labels: set[float] = set()
    for trial in trials:
        allowed_values.extend(float(value) for value in trial.metrics.values())
        for value in trial.parameters.values():
            if isinstance(value, int | float) and not isinstance(value, bool):
                allowed_values.append(float(value))
        for metric_name in trial.metrics:
            allowed_labels.update(float(value) for value in re.findall(r"\d+", metric_name))
    text = " ".join(
        [
            narrative.headline,
            narrative.conclusion,
            *narrative.tradeoffs,
            *narrative.limitations,
            narrative.next_experiment,
        ]
    )
    for token in _NUMBER.findall(text):
        is_percent = token.endswith("%")
        numeric = float(token.rstrip("%"))
        normalized = numeric / 100 if is_percent else numeric
        value_match = any(
            abs(normalized - evidence) <= max(0.00005, abs(evidence) * 0.0005)
            for evidence in allowed_values
        )
        if not value_match and numeric not in allowed_labels:
            raise ValueError(f"ungrounded numerical claim: {token}")


def _select_model_trial(trials: list[TrialResult], model_name: str) -> TrialResult:
    matches = [trial for trial in trials if trial.model_name == model_name]
    if not matches:
        raise ValueError(f"no trial evidence for model {model_name}")
    return max(matches, key=lambda trial: max(trial.metrics.values(), default=float("-inf")))


def _primary_metric(champion: TrialResult, challenger: TrialResult) -> str:
    shared = set(champion.metrics).intersection(challenger.metrics)
    ranked = sorted(shared, key=lambda name: ("ndcg" not in name, name))
    if not ranked:
        raise ValueError("champion and challenger share no metrics")
    return ranked[0]
