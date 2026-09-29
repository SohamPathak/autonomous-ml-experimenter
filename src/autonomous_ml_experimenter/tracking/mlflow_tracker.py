"""Safe MLflow tracking with nested studies and registry lineage."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from autonomous_ml_experimenter.core.contracts import Hypothesis, TrialResult
from autonomous_ml_experimenter.core.security import redact


class MLflowTracker:
    """SQLite-backed local tracker that never serializes secret-like metadata."""

    def __init__(
        self,
        database_path: Path,
        artifact_root: Path,
        experiment_name: str,
        repository_root: Path,
    ) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{database_path.resolve()}"
        self.artifact_root = artifact_root.resolve()
        self.experiment_name = experiment_name
        self.repository_root = repository_root.resolve()
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self._ensure_experiment()

    def _ensure_experiment(self) -> None:
        existing = self.client.get_experiment_by_name(self.experiment_name)
        if existing is None:
            self.client.create_experiment(
                self.experiment_name, artifact_location=self.artifact_root.as_uri()
            )

    @contextmanager
    def study(self, phase: str, tags: Mapping[str, Any] | None = None) -> Iterator[str]:
        """Open one parent run for a bounded research phase."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        safe_tags = self._safe_mapping(tags or {})
        safe_tags.update(self._git_metadata())
        with mlflow.start_run(run_name=phase, tags=safe_tags) as run:
            yield run.info.run_id

    @contextmanager
    def trial(
        self,
        trial_id: str,
        parameters: Mapping[str, Any],
        hypothesis: Hypothesis,
    ) -> Iterator[str]:
        """Open a child run and log its bounded proposal."""
        with mlflow.start_run(run_name=trial_id, nested=True) as run:
            safe_parameters = self._safe_mapping(parameters)
            if safe_parameters:
                mlflow.log_params(safe_parameters)
            mlflow.log_dict(redact(hypothesis.model_dump(mode="json")), "hypothesis.json")
            for key, value in self._git_metadata().items():
                mlflow.set_tag(key, value)
            yield run.info.run_id

    def log_result(self, result: TrialResult) -> None:
        """Record normalized metrics and outcome on the active trial."""
        valid_metrics = {
            key: value for key, value in result.metrics.items() if math.isfinite(value)
        }
        if valid_metrics:
            mlflow.log_metrics(valid_metrics)
        mlflow.set_tags(
            {
                "trial.outcome": result.outcome.value,
                "trial.phase": result.phase,
                "trial.model": result.model_name,
            }
        )
        mlflow.log_dict(redact(result.model_dump(mode="json")), "trial_result.json")

    def log_model_directory(self, directory: Path) -> None:
        """Log an already-serialized model directory under a stable artifact path."""
        if not directory.is_dir():
            raise FileNotFoundError(f"model artifact directory not found: {directory.name}")
        mlflow.log_artifacts(str(directory), artifact_path="model")

    def register_candidate(
        self,
        registered_name: str,
        run_id: str,
        tags: Mapping[str, Any] | None = None,
    ) -> str:
        """Create an immutable version and point the candidate alias to it."""
        try:
            self.client.get_registered_model(registered_name)
        except MlflowException:
            self.client.create_registered_model(registered_name)
        version = self.client.create_model_version(
            name=registered_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            tags=self._safe_mapping(tags or {}),
        )
        self.client.set_registered_model_alias(registered_name, "candidate", version.version)
        return str(version.version)

    def approve_candidate(self, registered_name: str, version: str, approved_by: str) -> None:
        """Apply an explicit human approval and update the champion alias."""
        if not approved_by.strip():
            raise ValueError("approved_by is required for champion promotion")
        self.client.set_model_version_tag(registered_name, version, "validation_status", "approved")
        self.client.set_model_version_tag(
            registered_name, version, "approved_by", approved_by.strip()
        )
        self.client.set_registered_model_alias(registered_name, "champion", version)

    @staticmethod
    def _safe_mapping(values: Mapping[str, Any]) -> dict[str, str]:
        safe = redact(dict(values))
        return {
            str(key): value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            for key, value in safe.items()
        }

    def _git_metadata(self) -> dict[str, str]:
        sha = self._git(["rev-parse", "HEAD"])
        status = self._git(["status", "--porcelain"])
        return {
            "code.git_sha": sha if sha else "unborn",
            "code.git_state": "dirty" if status else ("unborn" if not sha else "clean"),
        }

    def _git(self, arguments: list[str]) -> str:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
