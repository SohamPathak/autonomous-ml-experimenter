import re
import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.view_models import load_dashboard_bundle, model_summary, monitoring_frame, trial_frame

BUNDLE_PATH = Path("app/data/demo_bundle.json")


def test_dashboard_view_models_use_actual_bundle() -> None:
    bundle = load_dashboard_bundle(BUNDLE_PATH)
    trials = trial_frame(bundle)
    summary = model_summary(bundle)
    monitoring = monitoring_frame(bundle)
    assert len(trials) == len(bundle.trials)
    assert {"model", "ndcg_at_10", "catalog_coverage", "outcome"}.issubset(trials.columns)
    assert set(summary["model"]) == {trial.model_name for trial in bundle.trials}
    assert monitoring["simulated_incident"].sum() == 1
    assert monitoring.iloc[-1]["status"] == "critical"


def test_streamlit_overview_and_research_journey_render() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Autonomous ML Experimenter"
    assert "PROMOTE" in " ".join(item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Research Journey")
    app.run(timeout=30)
    assert not app.exception
    assert any("Research Journey" in item.value for item in app.title)
    assert app.get("plotly_chart")


def test_all_dashboard_views_render_without_live_services() -> None:
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path).run(timeout=30)
    pages = [
        "Overview",
        "Research Journey",
        "Experiment Comparison",
        "Registry & Lineage",
        "Monitoring",
        "Summary",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=30)
        assert not app.exception, page


def test_dashboard_runtime_dependencies_are_declared_in_both_manifests() -> None:
    """Hosted platforms may resolve either manifest, so both must start the app."""
    root = Path(__file__).parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_bytes().decode("utf-8"))
    declared = {
        re.split(r"[=<>!~\[]", entry)[0].strip().lower()
        for entry in pyproject["project"]["dependencies"]
    }
    pinned = {
        re.split(r"[=<>!~\[]", line)[0].strip().lower()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {"streamlit", "plotly", "pandas", "pydantic"}
    assert required <= declared, f"pyproject is missing {required - declared}"
    assert required <= pinned, f"requirements.txt is missing {required - pinned}"
