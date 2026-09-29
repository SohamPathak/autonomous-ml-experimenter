"""Read-only Streamlit control plane for Autonomous ML Experimenter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.view_models import (  # noqa: E402
    load_dashboard_bundle,
    model_summary,
    monitoring_frame,
    primary_metric,
    selected_trials,
    trial_frame,
)

BUNDLE_PATH = Path(os.getenv("PRESENTATION_BUNDLE", "app/data/demo_bundle.json"))

st.set_page_config(
    page_title="Autonomous ML Experimenter",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#102a2a; --muted:#60706e; --green:#00a878; --mint:#dff8ef; --amber:#f4b942; }
    .stApp { background: linear-gradient(145deg,#f8fbfa 0%,#eef7f4 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#102a2a; }
    [data-testid="stSidebar"] * { color:#f4fffb !important; }
    .block-container { padding-top:2rem; max-width:1350px; }
    .eyebrow { color:#008d68; font-size:.78rem; font-weight:800;
               letter-spacing:.13em; text-transform:uppercase; }
    .subtitle { color:var(--muted); font-size:1.05rem; margin-top:-.5rem; }
    .decision { border-left:5px solid var(--green); background:white;
                padding:1rem 1.2rem; border-radius:.6rem;
                box-shadow:0 8px 24px rgba(16,42,42,.07); margin:.8rem 0 1.2rem; }
    .pill { display:inline-block; padding:.3rem .65rem; border-radius:999px; background:var(--mint);
            color:#087659; font-weight:800; font-size:.78rem; margin-right:.35rem; }
    .note { background:#fff8e8; border:1px solid #f5d78b; padding:.8rem 1rem; border-radius:.5rem; }
    [data-testid="stMetric"] { background:white; border:1px solid #dfece8;
                               border-radius:.7rem; padding:.75rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_bundle(path: str):  # type: ignore[no-untyped-def]
    return load_dashboard_bundle(Path(path))


try:
    bundle = get_bundle(str(BUNDLE_PATH))
except Exception:
    st.error("The presentation bundle is missing or invalid. Run the local export pipeline first.")
    st.stop()

st.sidebar.markdown("## ◉ Experimenter")
st.sidebar.caption("READ-ONLY CONTROL PLANE")
PAGE_NAMES = [
    "Overview",
    "Research Journey",
    "Experiment Comparison",
    "Registry & Lineage",
    "Monitoring",
    "Summary",
]
requested_page = st.query_params.get("page", "Overview")
page = st.sidebar.radio(
    "Navigate",
    PAGE_NAMES,
    index=PAGE_NAMES.index(requested_page) if requested_page in PAGE_NAMES else 0,
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Bundle:** `{bundle.schema_version}`")
st.sidebar.markdown(f"**Trials:** `{len(bundle.trials)}`")
st.sidebar.markdown(f"**Data:** `{bundle.data_manifest.fingerprint[:10]}…`")
st.sidebar.caption("Offline research replay · No live training")


def page_heading(label: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def overview() -> None:
    page_heading(
        "Decision control plane",
        "Autonomous ML Experimenter",
        "Model-agnostic research loops for experimentation, comparison, governance, "
        "and monitoring.",
    )
    metric = primary_metric(bundle)
    champion, challenger = selected_trials(bundle)
    lift = (float(challenger[metric]) - float(champion[metric])) / max(
        abs(float(champion[metric])), 1e-12
    )
    st.markdown(
        f'<div class="decision"><span class="pill">{bundle.decision.status.value.upper()}</span>'
        f'<span class="pill">HUMAN APPROVAL PENDING</span><h3 style="margin:.65rem 0 .2rem">'
        f"{bundle.decision.challenger_model} is the recommended challenger</h3>"
        f"<div>{bundle.narrative.conclusion}</div></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Final test NDCG@10", f"{float(challenger[metric]):.3f}", fmt_percent(lift))
    columns[1].metric("Catalog coverage", fmt_percent(float(challenger["catalog_coverage"])))
    columns[2].metric("p95 inference", f"{float(challenger['p95_latency_ms']):.2f} ms")
    columns[3].metric("Research trials", len(bundle.trials))
    st.subheader("Why this decision")
    left, right = st.columns([1.35, 1])
    with left:
        for reason in bundle.decision.reasons:
            st.markdown(f"✓ {reason}")
    with right:
        guardrails = pd.DataFrame(
            {
                "Guardrail": list(bundle.decision.guardrails),
                "Status": [
                    "PASS" if value else "FAIL" for value in bundle.decision.guardrails.values()
                ],
            }
        )
        st.dataframe(guardrails, hide_index=True, width="stretch")
    st.markdown(
        '<div class="note"><b>Evidence boundary:</b> this is leakage-safe offline replay. '
        "It is not a production monitor or causal A/B-test result.</div>",
        unsafe_allow_html=True,
    )


def research_journey() -> None:
    page_heading(
        "Adaptive optimization",
        "Research Journey",
        "Every point is an executed trial; failures, plateaus, and pivots remain visible.",
    )
    frame = trial_frame(bundle)
    frame = frame[frame["evaluation_split"] == "validation"].reset_index(drop=True)
    metric = primary_metric(bundle)
    colors = {
        "improvement": "#00a878",
        "pivot": "#7557d3",
        "plateau": "#f4b942",
        "failure": "#df5b5b",
        "pruned": "#8b9895",
    }
    symbols = {
        "improvement": "star",
        "pivot": "diamond",
        "plateau": "circle",
        "failure": "x",
        "pruned": "triangle-down",
    }
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["order"],
            y=frame[metric],
            mode="lines",
            line={"color": "#b5c8c2", "width": 3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for outcome, group in frame.groupby("outcome", sort=False):
        figure.add_trace(
            go.Scatter(
                x=group["order"],
                y=group[metric],
                mode="markers",
                name=outcome.title(),
                marker={
                    "color": colors.get(outcome, "#60706e"),
                    "symbol": symbols.get(outcome, "circle"),
                    "size": 13,
                    "line": {"color": "white", "width": 1.5},
                },
                customdata=group[["trial_id", "phase", "model", "hypothesis", "parent_trial_id"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Model: %{customdata[2]}"
                    "<br>NDCG@10: %{y:.4f}<br>%{customdata[3]}<br>Parent: "
                    "%{customdata[4]}<extra></extra>"
                ),
            )
        )
    for _, group in frame.groupby("phase", sort=False):
        figure.add_vrect(
            x0=float(group["order"].min()) - 0.45,
            x1=float(group["order"].max()) + 0.45,
            fillcolor="#dff8ef",
            opacity=0.12,
            line_width=0,
            annotation_text=str(group["phase"].iloc[0]),
            annotation_position="top left",
        )
    figure.update_layout(
        height=510,
        xaxis_title="Ordered experiment",
        yaxis_title="Validation NDCG@10",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 30, "r": 20, "t": 75, "b": 30},
    )
    st.plotly_chart(figure, width="stretch", key="research-journey")
    st.dataframe(
        frame[["trial_id", "phase", "model", "outcome", metric, "parent_trial_id"]],
        hide_index=True,
        width="stretch",
    )


def experiment_comparison() -> None:
    page_heading(
        "Champion / challenger",
        "Experiment Comparison",
        "Quality is optimized; coverage and latency remain explicit decision constraints.",
    )
    summary = model_summary(bundle)
    metric = primary_metric(bundle)
    figure = px.scatter(
        summary,
        x="catalog_coverage",
        y=metric,
        color="model",
        size="p95_latency_ms",
        hover_data=["trial_id", "outcome"],
        color_discrete_sequence=["#00a878", "#7557d3", "#f4b942"],
    )
    figure.update_layout(height=420, template="plotly_white", showlegend=True)
    st.plotly_chart(figure, width="stretch", key="model-comparison")
    st.dataframe(
        summary.style.format(
            {metric: "{:.4f}", "catalog_coverage": "{:.1%}", "p95_latency_ms": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Selected candidate lineage")
    lineage = trial_frame(bundle)
    lineage = lineage[lineage["model"] == bundle.decision.challenger_model]
    st.dataframe(
        lineage[
            [
                "trial_id",
                "parent_trial_id",
                "phase",
                "evaluation_split",
                "outcome",
                "model_version",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


def registry_lineage() -> None:
    page_heading(
        "Audit trail",
        "Registry & Lineage",
        "Code, data, parent trial, model family, and review state remain traceable.",
    )
    st.markdown(
        f'<div class="decision"><span class="pill">CANDIDATE</span>'
        f'<span class="pill">REVIEW PENDING</span><b>{bundle.decision.challenger_model}</b> '
        "has a promotion recommendation; the champion alias is unchanged.</div>",
        unsafe_allow_html=True,
    )
    frame = trial_frame(bundle)
    st.dataframe(
        frame[
            [
                "trial_id",
                "parent_trial_id",
                "phase",
                "model",
                "outcome",
                "run_id",
                "model_version",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Data fingerprint: {bundle.data_manifest.fingerprint} · "
        f"Source: {bundle.data_manifest.source}"
    )


def monitoring() -> None:
    page_heading(
        "Offline observability",
        "Monitoring",
        "Untouched later windows simulate delayed-label monitoring; incidents are "
        "explicitly marked.",
    )
    frame = monitoring_frame(bundle)
    observed = frame[~frame["simulated_incident"]]
    incident = frame[frame["simulated_incident"]]
    status_columns = st.columns(len(frame))
    for column, (_, row) in zip(status_columns, frame.iterrows(), strict=True):
        column.metric(str(row["window"]), str(row["status"]).upper())
    metric = primary_metric(bundle)
    quality = px.line(
        observed,
        x="window",
        y=[metric, "catalog_coverage"],
        markers=True,
        title="Observed ranking quality and coverage",
        color_discrete_sequence=["#00a878", "#7557d3"],
    )
    drift = px.line(
        observed,
        x="window",
        y=["event_type_js", "item_popularity_js"],
        markers=True,
        title="Observed distribution drift",
        color_discrete_sequence=["#f4b942", "#df5b5b"],
    )
    left, right = st.columns(2)
    left.plotly_chart(quality, width="stretch", key="quality-monitoring")
    right.plotly_chart(drift, width="stretch", key="drift-monitoring")
    if not incident.empty:
        st.error(str(incident.iloc[0]["alerts"]))
        st.caption(
            "Synthetic incident for alert-path demonstration; excluded from observed trends."
        )
    st.dataframe(
        frame[["window", "status", metric, "event_type_js", "unknown_item_rate", "p95_latency_ms"]],
        hide_index=True,
        width="stretch",
    )


def summary() -> None:
    page_heading(
        "Evidence-grounded communication",
        "Summary",
        f"Narrative provider: {bundle.narrative.provider} · policy decision remains deterministic.",
    )
    st.markdown(f"## {bundle.narrative.headline}")
    st.write(bundle.narrative.conclusion)
    st.subheader("Trade-offs")
    for item in bundle.narrative.tradeoffs:
        st.markdown(f"- {item}")
    st.subheader("Limitations")
    for item in bundle.narrative.limitations:
        st.markdown(f"- {item}")
    st.subheader("Next experiment")
    st.info(bundle.narrative.next_experiment)


PAGES = {
    "Overview": overview,
    "Research Journey": research_journey,
    "Experiment Comparison": experiment_comparison,
    "Registry & Lineage": registry_lineage,
    "Monitoring": monitoring,
    "Summary": summary,
}
PAGES[page]()
