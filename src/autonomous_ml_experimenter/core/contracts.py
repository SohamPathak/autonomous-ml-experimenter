"""Serializable contracts shared across orchestration boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base class that rejects accidental, unversioned fields."""

    model_config = ConfigDict(extra="forbid")


class TrialOutcome(StrEnum):
    IMPROVEMENT = "improvement"
    FAILURE = "failure"
    PLATEAU = "plateau"
    PIVOT = "pivot"
    PRUNED = "pruned"


class DecisionStatus(StrEnum):
    PROMOTE = "promote"
    ITERATE = "iterate"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class DataManifest(StrictModel):
    fingerprint: str
    row_count: int = Field(ge=0)
    user_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    start_time: datetime
    end_time: datetime
    source: str
    sampling: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(StrictModel):
    observation: str
    hypothesis: str
    expected_effect: str
    parameter_changes: dict[str, Any]
    parent_trial_id: str | None = None


class TrialResult(StrictModel):
    trial_id: str
    phase: str
    model_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    hypothesis: Hypothesis
    outcome: TrialOutcome
    duration_seconds: float = Field(ge=0)
    parent_run_id: str | None = None
    run_id: str | None = None
    model_version: str | None = None
    error: str | None = None
    evaluation_split: str = "validation"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DecisionRecord(StrictModel):
    status: DecisionStatus
    champion_model: str
    challenger_model: str
    reasons: list[str]
    guardrails: dict[str, bool]
    requires_human_approval: bool = True
    approved: bool = False


class MonitoringWindow(StrictModel):
    window_id: str
    start_time: datetime
    end_time: datetime
    metrics: dict[str, float]
    status: HealthStatus
    alerts: list[str] = Field(default_factory=list)
    simulated_incident: bool = False


class Narrative(StrictModel):
    headline: str
    conclusion: str
    tradeoffs: list[str]
    limitations: list[str]
    next_experiment: str
    provider: str


class PresentationBundle(StrictModel):
    schema_version: str = "1.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    project: str = "Autonomous ML Experimenter"
    tagline: str = (
        "Model-agnostic research loops for experimentation, comparison, governance, and monitoring."
    )
    data_manifest: DataManifest
    trials: list[TrialResult]
    decision: DecisionRecord
    monitoring: list[MonitoringWindow]
    narrative: Narrative
    source_run_ids: list[str] = Field(default_factory=list)
