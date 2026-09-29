"""Validated application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(ConfigModel):
    mode: Literal["synthetic", "demo", "full"] = "synthetic"
    events_path: Path | None = None
    sample_users: int | None = Field(default=120, ge=20)
    min_user_events: int = Field(default=3, ge=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.2, gt=0, lt=0.5)

    @model_validator(mode="after")
    def require_events_for_public_data(self) -> DataConfig:
        if self.mode in {"demo", "full"} and self.events_path is None:
            raise ValueError("events_path is required in demo/full mode")
        if self.mode == "synthetic" and self.sample_users is None:
            raise ValueError("sample_users is required in synthetic mode")
        return self


class ResearchConfig(ConfigModel):
    max_trials_per_phase: int = Field(default=6, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=10)
    plateau_patience: int = Field(default=3, ge=1)
    minimum_improvement: float = Field(default=0.005, ge=0)
    sampler_seed: int = 42


class GovernanceConfig(ConfigModel):
    primary_metric: str = "ndcg_at_10"
    minimum_relative_improvement: float = Field(default=0.03, ge=0)
    minimum_catalog_coverage: float = Field(default=0.08, ge=0, le=1)
    maximum_p95_latency_ms: float = Field(default=50.0, gt=0)
    maximum_cold_recall_degradation: float = Field(default=0.02, ge=0)


class VertexConfig(ConfigModel):
    enabled: bool = False
    model: str = "gemini-2.5-flash"
    location: str = "global"


class TrackingConfig(ConfigModel):
    enabled: bool = False
    database_path: Path = Path("artifacts/mlflow.db")
    artifact_root: Path = Path("artifacts/mlruns")


class AppConfig(ConfigModel):
    schema_version: str = "1.0"
    seed: int = 42
    top_k: int = Field(default=10, ge=1, le=100)
    data: DataConfig = Field(default_factory=DataConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def load_config(path: Path) -> AppConfig:
    """Load a YAML configuration without resolving or exposing secrets."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)
