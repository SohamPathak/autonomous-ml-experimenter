"""Schema-bounded deterministic research hypotheses."""

from __future__ import annotations

from typing import Any

from autonomous_ml_experimenter.core.contracts import Hypothesis, TrialResult


class DeterministicResearchPlanner:
    """Explain optimizer proposals without inventing executable changes."""

    def propose(
        self,
        history: list[TrialResult],
        phase: str,
        model_name: str,
        parameters: dict[str, Any],
    ) -> Hypothesis:
        if history:
            parent = history[-1]
            observation = (
                f"Latest {parent.model_name} trial was {parent.outcome.value}; "
                f"the research history contains {len(history)} completed trials."
            )
            expected = (
                f"Test whether this bounded {model_name} configuration improves the declared "
                "validation objective without weakening guardrails."
            )
            parent_id = parent.trial_id
        else:
            observation = f"No completed trials exist for {phase}."
            expected = "Establish the first measured point for this model family."
            parent_id = None
        return Hypothesis(
            observation=observation,
            hypothesis=f"Adaptive search proposes a new {model_name} configuration.",
            expected_effect=expected,
            parameter_changes=parameters,
            parent_trial_id=parent_id,
        )
