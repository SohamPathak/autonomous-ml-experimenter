from pathlib import Path

from autonomous_ml_experimenter.config import load_config
from autonomous_ml_experimenter.pipeline import run_pipeline
from autonomous_ml_experimenter.presentation.bundle import load_bundle


def test_end_to_end_pipeline_exports_final_test_and_monitoring(tmp_path: Path) -> None:
    config = load_config(Path("configs/demo.yaml"))
    config = config.model_copy(
        update={
            "research": config.research.model_copy(
                update={"max_trials_per_phase": 2, "plateau_patience": 2}
            ),
            "tracking": config.tracking.model_copy(update={"enabled": False}),
            "vertex": config.vertex.model_copy(update={"enabled": False}),
        }
    )
    output = tmp_path / "presentation.json"
    result = run_pipeline(config, output, repository_root=Path.cwd())
    restored = load_bundle(output)
    assert result == restored
    assert len([trial for trial in restored.trials if trial.evaluation_split == "test"]) == 2
    assert restored.decision.requires_human_approval
    assert not restored.decision.approved
    assert restored.monitoring[-1].simulated_incident
    assert restored.narrative.provider == "deterministic"
