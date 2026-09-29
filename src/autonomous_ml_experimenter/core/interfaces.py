"""Model-agnostic extension points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypeVar

from autonomous_ml_experimenter.core.contracts import (
    DecisionRecord,
    Hypothesis,
    Narrative,
    TrialResult,
)

DataT = TypeVar("DataT")
ModelT = TypeVar("ModelT")


class ModelAdapter(Protocol[DataT]):
    name: str

    def fit(self, data: DataT) -> None: ...

    def predict(self, entity_ids: list[int], k: int) -> dict[int, list[int]]: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> ModelAdapter[DataT]: ...


class ExperimentTask(Protocol[DataT]):
    name: str

    def load_data(self) -> DataT: ...

    def create_splits(self, data: DataT) -> tuple[DataT, DataT, DataT]: ...

    def build_model(self, model_name: str, parameters: dict[str, Any]) -> ModelAdapter[DataT]: ...

    def evaluate(self, model: ModelAdapter[DataT], data: DataT) -> dict[str, float]: ...

    def search_space(self, model_name: str, trial: Any) -> dict[str, Any]: ...


class ResearchPlanner(Protocol):
    def propose(
        self,
        history: list[TrialResult],
        phase: str,
        model_name: str,
        parameters: dict[str, Any],
    ) -> Hypothesis: ...


class DecisionPolicy(Protocol):
    def decide(self, champion: TrialResult, challenger: TrialResult) -> DecisionRecord: ...


class Narrator(Protocol):
    def narrate(self, decision: DecisionRecord, trials: list[TrialResult]) -> Narrative: ...
