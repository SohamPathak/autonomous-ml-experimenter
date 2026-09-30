"""Read-only Streamlit control plane for Autonomous ML Experimenter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.view_models import (
    load_dashboard_bundle,
    model_summary,
    monitoring_frame,
    primary_metric,
    selected_trials,
    trial_frame,
)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = Path(os.getenv("PRESENTATION_BUNDLE", str(ROOT / "app/data/demo_bundle.json")))

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
def get_bundle(path: str) -> dict[str, Any]:
    return load_dashboard_bundle(Path(path))


try:
    bundle = get_bundle(str(BUNDLE_PATH))
except Exception as error:
    st.error(
        "The presentation bundle is missing or invalid. Run the local export pipeline "
        f"first (`autonomous-ml-experimenter run --offline`). Details: {error}"
    )
    st.stop()

DECISION = bundle["decision"]
NARRATIVE = bundle["narrative"]
MANIFEST = bundle["data_manifest"]

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
st.sidebar.markdown(f"**Bundle:** `{bundle['schema_version']}`")
st.sidebar.markdown(f"**Trials:** `{len(bundle['trials'])}`")
st.sidebar.markdown(f"**Data:** `{MANIFEST['fingerprint'][:10]}…`")
st.sidebar.caption("Offline research replay · No live training")


def page_heading(label: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


OUTCOME_COLORS = {
    "improvement": "#00a878",
    "pivot": "#7557d3",
    "plateau": "#f4b942",
    "failure": "#df5b5b",
    "pruned": "#8b9895",
}
OUTCOME_SHAPES = {
    "improvement": "triangle-up",
    "pivot": "diamond",
    "plateau": "circle",
    "failure": "cross",
    "pruned": "triangle-down",
}
SERIES_COLORS = ["#00a878", "#7557d3", "#f4b942", "#df5b5b"]


def series_chart(frame: pd.DataFrame, columns: list[str], title: str) -> alt.LayerChart:
    """Build a multi-series line chart from wide monitoring data."""
    melted = frame.melt(
        id_vars=["window"], value_vars=columns, var_name="metric", value_name="value"
    )
    scale = alt.Scale(domain=columns, range=SERIES_COLORS[: len(columns)])
    encoding = {
        "x": alt.X("window:N", title="Window", sort=list(frame["window"])),
        "y": alt.Y("value:Q", title="Value"),
        "color": alt.Color("metric:N", title=None, scale=scale, legend=alt.Legend(orient="bottom")),
        "tooltip": [
            alt.Tooltip("window:N", title="Window"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Value", format=".4f"),
        ],
    }
    base = alt.Chart(melted)
    line = base.mark_line(strokeWidth=3).encode(**encoding)
    points = base.mark_point(size=90, filled=True).encode(**encoding)
    return (line + points).properties(height=320, title=title)


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
        f'<div class="decision"><span class="pill">{str(DECISION["status"]).upper()}</span>'
        f'<span class="pill">HUMAN APPROVAL PENDING</span><h3 style="margin:.65rem 0 .2rem">'
        f"{DECISION['challenger_model']} is the recommended challenger</h3>"
        f"<div>{NARRATIVE['conclusion']}</div></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    columns[0].metric("Final test NDCG@10", f"{float(challenger[metric]):.3f}", fmt_percent(lift))
    columns[1].metric("Catalog coverage", fmt_percent(float(challenger["catalog_coverage"])))
    columns[2].metric("p95 inference", f"{float(challenger['p95_latency_ms']):.2f} ms")
    columns[3].metric("Research trials", len(bundle["trials"]))
    st.subheader("Why this decision")
    left, right = st.columns([1.35, 1])
    with left:
        for reason in DECISION["reasons"]:
            st.markdown(f"✓ {reason}")
    with right:
        guardrails = pd.DataFrame(
            {
                "Guardrail": list(DECISION["guardrails"]),
                "Status": [
                    "PASS" if value else "FAIL" for value in DECISION["guardrails"].values()
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
    outcomes = [value for value in OUTCOME_COLORS if value in set(frame["outcome"])]
    phases = (
        frame.groupby("phase", sort=False)
        .agg(start=("order", "min"), end=("order", "max"))
        .reset_index()
    )
    phases["start"] = phases["start"] - 0.5
    phases["end"] = phases["end"] + 0.5
    base = alt.Chart(frame)
    axis_scale = alt.Scale(
        domain=[float(frame["order"].min()) - 0.6, float(frame["order"].max()) + 0.6],
        nice=False,
    )
    bands = (
        alt.Chart(phases)
        .mark_rect(fill="#00a878", opacity=0.07)
        .encode(
            x=alt.X("start:Q", title="Ordered experiment", scale=axis_scale),
            x2="end:Q",
        )
    )
    labels = (
        alt.Chart(phases)
        .mark_text(align="left", dy=12, dx=6, fontSize=11, color="#4c5f5c", fontWeight="bold")
        .encode(x=alt.X("start:Q", scale=axis_scale), y=alt.value(0), text="phase:N")
    )
    trend = base.mark_line(color="#b5c8c2", strokeWidth=3).encode(
        x=alt.X("order:Q", title="Ordered experiment", scale=axis_scale),
        y=alt.Y(
            f"{metric}:Q",
            title="Validation NDCG@10",
            scale=alt.Scale(zero=False, padding=25),
        ),
    )
    points = base.mark_point(size=260, filled=True, strokeWidth=1.5, stroke="white").encode(
        x=alt.X("order:Q", scale=axis_scale),
        y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False, padding=25)),
        color=alt.Color(
            "outcome:N",
            title="Outcome",
            scale=alt.Scale(domain=outcomes, range=[OUTCOME_COLORS[value] for value in outcomes]),
            legend=alt.Legend(orient="top"),
        ),
        shape=alt.Shape(
            "outcome:N",
            scale=alt.Scale(domain=outcomes, range=[OUTCOME_SHAPES[value] for value in outcomes]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("trial_id:N", title="Trial"),
            alt.Tooltip("phase:N", title="Phase"),
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip(f"{metric}:Q", title="NDCG@10", format=".4f"),
            alt.Tooltip("outcome:N", title="Outcome"),
            alt.Tooltip("hypothesis:N", title="Hypothesis"),
            alt.Tooltip("parent_trial_id:N", title="Parent"),
        ],
    )
    st.altair_chart(
        (bands + labels + trend + points).properties(height=460).configure_view(strokeWidth=0),
        width="stretch",
        key="research-journey",
    )
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
    comparison = (
        alt.Chart(summary)
        .mark_point(filled=True, strokeWidth=1.5, stroke="white", opacity=0.95)
        .encode(
            x=alt.X(
                "catalog_coverage:Q",
                title="Catalog coverage",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False, padding=30),
            ),
            y=alt.Y(f"{metric}:Q", title="NDCG@10", scale=alt.Scale(zero=False, padding=30)),
            color=alt.Color(
                "model:N",
                title="Model",
                scale=alt.Scale(range=SERIES_COLORS),
                legend=alt.Legend(orient="right"),
            ),
            size=alt.Size(
                "p95_latency_ms:Q",
                title="p95 latency (ms)",
                scale=alt.Scale(range=[140, 700]),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(f"{metric}:Q", title="NDCG@10", format=".4f"),
                alt.Tooltip("catalog_coverage:Q", title="Coverage", format=".1%"),
                alt.Tooltip("p95_latency_ms:Q", title="p95 latency (ms)", format=".3f"),
                alt.Tooltip("trial_id:N", title="Trial"),
                alt.Tooltip("outcome:N", title="Outcome"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(comparison, width="stretch", key="model-comparison")
    st.dataframe(
        summary.style.format(
            {metric: "{:.4f}", "catalog_coverage": "{:.1%}", "p95_latency_ms": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Selected candidate lineage")
    lineage = trial_frame(bundle)
    lineage = lineage[lineage["model"] == DECISION["challenger_model"]]
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
        f'<span class="pill">REVIEW PENDING</span><b>{DECISION["challenger_model"]}</b> '
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
    st.caption(f"Data fingerprint: {MANIFEST['fingerprint']} · Source: {MANIFEST['source']}")


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
    left, right = st.columns(2)
    left.altair_chart(
        series_chart(observed, [metric, "catalog_coverage"], "Ranking quality and coverage"),
        width="stretch",
        key="quality-monitoring",
    )
    right.altair_chart(
        series_chart(observed, ["event_type_js", "item_popularity_js"], "Distribution drift"),
        width="stretch",
        key="drift-monitoring",
    )
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
        f"Narrative provider: {NARRATIVE['provider']} · policy decision remains deterministic.",
    )
    st.markdown(f"## {NARRATIVE['headline']}")
    st.write(NARRATIVE["conclusion"])
    st.subheader("Trade-offs")
    for item in NARRATIVE["tradeoffs"]:
        st.markdown(f"- {item}")
    st.subheader("Limitations")
    for item in NARRATIVE["limitations"]:
        st.markdown(f"- {item}")
    st.subheader("Next experiment")
    st.info(NARRATIVE["next_experiment"])


PAGES = {
    "Overview": overview,
    "Research Journey": research_journey,
    "Experiment Comparison": experiment_comparison,
    "Registry & Lineage": registry_lineage,
    "Monitoring": monitoring,
    "Summary": summary,
}
PAGES[page]()
