from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from autonomous_ml_experimenter.ai.vertex import (
    DeterministicNarrator,
    VertexNarrator,
    VertexResearchPlanner,
    validate_numerical_grounding,
)
from autonomous_ml_experimenter.core.contracts import (
    DecisionRecord,
    DecisionStatus,
    Hypothesis,
    Narrative,
    TrialOutcome,
    TrialResult,
)


def _trials() -> list[TrialResult]:
    base = Hypothesis(
        observation="No baseline.",
        hypothesis="Establish baseline.",
        expected_effect="Measure quality.",
        parameter_changes={},
    )
    return [
        TrialResult(
            trial_id="base",
            phase="V0",
            model_name="popularity",
            parameters={},
            metrics={"ndcg_at_10": 0.10, "catalog_coverage": 0.20},
            hypothesis=base,
            outcome=TrialOutcome.IMPROVEMENT,
            duration_seconds=1,
        ),
        TrialResult(
            trial_id="candidate",
            phase="V1",
            model_name="co_visitation",
            parameters={"session_minutes": 30},
            metrics={"ndcg_at_10": 0.21, "catalog_coverage": 0.42},
            hypothesis=base,
            outcome=TrialOutcome.IMPROVEMENT,
            duration_seconds=1,
        ),
    ]


def _decision() -> DecisionRecord:
    return DecisionRecord(
        status=DecisionStatus.PROMOTE,
        champion_model="popularity",
        challenger_model="co_visitation",
        reasons=["Primary metric improved."],
        guardrails={"coverage": True, "latency": True},
    )


class _Gateway:
    def __init__(self, response: BaseModel | Exception) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_deterministic_narrator_is_factual() -> None:
    narrative = DeterministicNarrator().narrate(_decision(), _trials())
    assert narrative.provider == "deterministic"
    assert "0.2100" in narrative.conclusion
    validate_numerical_grounding(narrative, _trials())


def test_vertex_narrator_falls_back_on_failure_or_ungrounded_number() -> None:
    failed = VertexNarrator(_Gateway(RuntimeError("/Users/example/private.json")))
    assert failed.narrate(_decision(), _trials()).provider == "deterministic-fallback"

    hallucinated = Narrative(
        headline="Promote with 99.9% confidence",
        conclusion="NDCG@10 significantly reached 0.2100, leading to its promotion.",
        tradeoffs=[],
        limitations=[],
        next_experiment="Run a replay.",
        provider="vertex-ai",
    )
    guarded = VertexNarrator(_Gateway(hallucinated))
    assert guarded.narrate(_decision(), _trials()).provider == "deterministic-fallback"


def test_vertex_planner_cannot_change_optimizer_parameters() -> None:
    unsafe = Hypothesis(
        observation="Prior plateau.",
        hypothesis="Change code and hidden parameter.",
        expected_effect="Improve score.",
        parameter_changes={"unbounded": 999},
        parent_trial_id=None,
    )
    planner = VertexResearchPlanner(_Gateway(unsafe))
    parameters: dict[str, Any] = {"session_minutes": 30}
    result = planner.propose(_trials(), "V1", "co_visitation", parameters)
    assert result.parameter_changes == parameters
    assert result.parent_trial_id == "candidate"


def test_grounding_rejects_numbers_absent_from_evidence() -> None:
    narrative = Narrative(
        headline="Score improved by 77%",
        conclusion="Candidate reached 0.2100 NDCG@10.",
        tradeoffs=[],
        limitations=[],
        next_experiment="Run one replay.",
        provider="vertex-ai",
    )
    with pytest.raises(ValueError, match="ungrounded numerical claim"):
        validate_numerical_grounding(narrative, _trials())


def test_claim_language_guard_rejects_significance_and_completed_promotion() -> None:
    from autonomous_ml_experimenter.ai.vertex import validate_claim_language

    unsafe = Narrative(
        headline="Model was promoted",
        conclusion="The candidate significantly outperformed the baseline.",
        tradeoffs=[],
        limitations=[],
        next_experiment="Replay the next window.",
        provider="vertex-ai",
    )
    with pytest.raises(ValueError, match="unsupported offline claim language"):
        validate_claim_language(unsafe)

    validate_claim_language(DeterministicNarrator().narrate(_decision(), _trials()))
