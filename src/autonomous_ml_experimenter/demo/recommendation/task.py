"""Recommendation plugin for the model-agnostic experiment platform."""

from __future__ import annotations

from typing import Any

import pandas as pd

from autonomous_ml_experimenter.config import AppConfig
from autonomous_ml_experimenter.demo.recommendation.data import (
    InteractionSplits,
    load_interactions,
    prepare_interactions,
)
from autonomous_ml_experimenter.demo.recommendation.evaluation import evaluate_recommender
from autonomous_ml_experimenter.demo.recommendation.models import (
    ALSModel,
    CoVisitationModel,
    PopularityModel,
)


class RecommendationTask:
    """Retailrocket demonstration task behind general experiment contracts."""

    name = "retailrocket_recommendation"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.splits: InteractionSplits | None = None

    def load_data(self) -> pd.DataFrame:
        return load_interactions(self.config.data, self.config.seed)

    def create_splits(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.splits = prepare_interactions(data, self.config.data, self.config.seed)
        return self.splits.train, self.splits.validation, self.splits.test

    def build_model(self, model_name: str, parameters: dict[str, Any]) -> Any:
        if model_name == "popularity":
            return PopularityModel()
        if model_name == "co_visitation":
            return CoVisitationModel(**parameters)
        if model_name == "implicit_als":
            return ALSModel(seed=self.config.seed, **parameters)
        raise ValueError(f"unknown recommendation model: {model_name}")

    def evaluate(self, model: Any, data: pd.DataFrame) -> dict[str, float]:
        if self.splits is None:
            raise RuntimeError("create_splits must run before evaluate")
        return evaluate_recommender(model, self.splits.train, data, self.config.top_k)

    def search_space(self, model_name: str, trial: Any) -> dict[str, Any]:
        if model_name == "co_visitation":
            return {
                "session_minutes": trial.suggest_categorical("session_minutes", [15, 30, 60]),
                "recency_decay": trial.suggest_float("recency_decay", 0.65, 0.98),
                "neighbor_limit": trial.suggest_categorical("neighbor_limit", [20, 50, 100]),
            }
        if model_name == "implicit_als":
            return {
                "factors": trial.suggest_categorical("factors", [8, 16, 32]),
                "regularization": trial.suggest_float("regularization", 0.01, 0.2, log=True),
                "iterations": trial.suggest_int("iterations", 5, 15),
                "alpha": trial.suggest_float("alpha", 5.0, 40.0, log=True),
            }
        return {}
