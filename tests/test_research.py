from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from autonomous_ml_experimenter.config import ResearchConfig
from autonomous_ml_experimenter.core.contracts import TrialOutcome
from autonomous_ml_experimenter.research.loop import BoundedResearchLoop


@dataclass
class _Model:
    value: int
    name: str = "quadratic"

    def fit(self, data: pd.DataFrame) -> _Model:
        return self


class _Task:
    def __init__(self, constant: bool = False) -> None:
        self.constant = constant
        self.evaluated_ids: list[int] = []

    def build_model(self, model_name: str, parameters: dict[str, Any]) -> _Model:
        return _Model(int(parameters["value"]), model_name)

    def evaluate(self, model: _Model, data: pd.DataFrame) -> dict[str, float]:
        self.evaluated_ids.append(id(data))
        score = 0.5 if self.constant else 1.0 - abs(model.value - 3) / 10
        return {"score": score, "catalog_coverage": 0.5, "p95_latency_ms": 1.0}

    def search_space(self, model_name: str, trial: Any) -> dict[str, Any]:
        return {"value": trial.suggest_int("value", 0, 5)}


def _config(max_trials: int = 6, patience: int = 3) -> ResearchConfig:
    return ResearchConfig(
        max_trials_per_phase=max_trials,
        timeout_seconds=60,
        plateau_patience=patience,
        minimum_improvement=0.001,
        sampler_seed=11,
    )


def test_loop_is_bounded_reproducible_and_uses_validation_only() -> None:
    train = pd.DataFrame({"x": [1]})
    validation = pd.DataFrame({"x": [2]})
    test = pd.DataFrame({"x": [3]})
    first_task = _Task()
    second_task = _Task()
    first = BoundedResearchLoop(first_task, _config(), "score").run_phase(
        "quadratic", train, validation, "V1"
    )
    second = BoundedResearchLoop(second_task, _config(), "score").run_phase(
        "quadratic", train, validation, "V1"
    )
    assert 1 <= len(first) <= 6
    assert [(run.parameters, run.metrics["score"]) for run in first] == [
        (run.parameters, run.metrics["score"]) for run in second
    ]
    assert all(0 <= int(run.parameters["value"]) <= 5 for run in first)
    assert set(first_task.evaluated_ids) == {id(validation)}
    assert id(test) not in first_task.evaluated_ids


def test_plateau_stops_study_before_budget() -> None:
    train = pd.DataFrame({"x": [1]})
    validation = pd.DataFrame({"x": [2]})
    runs = BoundedResearchLoop(_Task(constant=True), _config(10, 2), "score").run_phase(
        "constant", train, validation, "V1"
    )
    assert len(runs) == 3
    assert runs[0].outcome == TrialOutcome.IMPROVEMENT
    assert runs[-1].outcome == TrialOutcome.PLATEAU


def test_new_model_family_is_recorded_as_pivot() -> None:
    train = pd.DataFrame({"x": [1]})
    validation = pd.DataFrame({"x": [2]})
    loop = BoundedResearchLoop(_Task(), _config(2, 2), "score")
    first_phase = loop.run_phase("family_a", train, validation, "V1")
    second_phase = loop.run_phase("family_b", train, validation, "V2", prior_history=first_phase)
    assert second_phase[0].outcome == TrialOutcome.PIVOT
    assert second_phase[0].hypothesis.parent_trial_id == first_phase[-1].trial_id
