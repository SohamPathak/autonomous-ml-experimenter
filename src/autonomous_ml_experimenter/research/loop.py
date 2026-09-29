"""Model-agnostic bounded adaptive experiment loop."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import optuna

from autonomous_ml_experimenter.config import ResearchConfig
from autonomous_ml_experimenter.core.contracts import TrialOutcome, TrialResult
from autonomous_ml_experimenter.core.interfaces import ResearchPlanner
from autonomous_ml_experimenter.core.security import redact
from autonomous_ml_experimenter.research.planner import DeterministicResearchPlanner

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BoundedResearchLoop:
    """Run adaptive trials under explicit budgets, schemas, and stopping rules."""

    def __init__(
        self,
        task: Any,
        config: ResearchConfig,
        primary_metric: str,
        tracker: Any | None = None,
        planner: ResearchPlanner | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.primary_metric = primary_metric
        self.tracker = tracker
        self.planner = planner or DeterministicResearchPlanner()

    def run_phase(
        self,
        model_name: str,
        train: Any,
        validation: Any,
        phase: str,
        prior_history: list[TrialResult] | None = None,
    ) -> list[TrialResult]:
        """Optimize one family against validation data only."""
        history = list(prior_history or [])
        phase_results: list[TrialResult] = []
        phase_best = float("-inf")
        stale_trials = 0
        sampler = optuna.samplers.TPESampler(seed=self.config.sampler_seed)
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study_context = (
            self.tracker.study(phase, {"model_family": model_name})
            if self.tracker is not None
            else nullcontext(None)
        )

        with study_context as parent_run_id:

            def objective(trial: optuna.Trial) -> float:
                nonlocal phase_best, stale_trials
                parameters = self.task.search_space(model_name, trial)
                combined_history = history + phase_results
                hypothesis = self.planner.propose(combined_history, phase, model_name, parameters)
                trial_id = f"{_slug(phase)}-{trial.number:03d}"
                trial_context = (
                    self.tracker.trial(trial_id, parameters, hypothesis)
                    if self.tracker is not None
                    else nullcontext(None)
                )
                started = time.perf_counter()
                with trial_context as run_id:
                    try:
                        model = self.task.build_model(model_name, parameters)
                        fitted = model.fit(train)
                        metrics = self.task.evaluate(fitted, validation)
                        score = float(metrics[self.primary_metric])
                        is_pivot = bool(
                            combined_history
                            and not phase_results
                            and combined_history[-1].model_name != model_name
                        )
                        if is_pivot:
                            outcome = TrialOutcome.PIVOT
                        elif score > phase_best + self.config.minimum_improvement:
                            outcome = TrialOutcome.IMPROVEMENT
                        else:
                            outcome = TrialOutcome.PLATEAU
                        if score > phase_best + self.config.minimum_improvement:
                            phase_best = score
                            stale_trials = 0
                        else:
                            stale_trials += 1
                        result = TrialResult(
                            trial_id=trial_id,
                            phase=phase,
                            model_name=model_name,
                            parameters=parameters,
                            metrics=metrics,
                            hypothesis=hypothesis,
                            outcome=outcome,
                            duration_seconds=time.perf_counter() - started,
                            parent_run_id=parent_run_id,
                            run_id=run_id,
                        )
                    except Exception as error:  # isolated trial failure is data, not a crash
                        stale_trials += 1
                        safe_error = redact(str(error))
                        result = TrialResult(
                            trial_id=trial_id,
                            phase=phase,
                            model_name=model_name,
                            parameters=parameters,
                            metrics={self.primary_metric: float("-1e30")},
                            hypothesis=hypothesis,
                            outcome=TrialOutcome.FAILURE,
                            duration_seconds=time.perf_counter() - started,
                            parent_run_id=parent_run_id,
                            run_id=run_id,
                            error=str(safe_error),
                        )
                        score = float("-1e30")
                    phase_results.append(result)
                    if self.tracker is not None:
                        self.tracker.log_result(result)
                if stale_trials >= self.config.plateau_patience:
                    study.stop()
                return score

            study.optimize(
                objective,
                n_trials=self.config.max_trials_per_phase,
                timeout=self.config.timeout_seconds,
                show_progress_bar=False,
            )
        return phase_results


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip(
        "-"
    )
