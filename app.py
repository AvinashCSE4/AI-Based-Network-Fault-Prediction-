"""
app.py  -  NetGuard AI : AI-Based Network Fault Prediction Dashboard
=====================================================================
Run with:   streamlit run app.py

This file only contains the user interface. The machine-learning logic
(data generation, training, prediction, explanation, CSV validation) lives
in train_model.py.

Everything shown as network data is SIMULATED (demo mode) or comes from a
stored CSV file (CSV mode). Nothing here connects to real routers/switches,
and a "Critical" result is a PREDICTION, not a confirmed device failure.
"""

from __future__ import annotations

import html as html_lib
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import train_model as tm

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PRIMARY, DARK_BLUE, LIGHT_BLUE = "#2563EB", "#1E40AF", "#EFF6FF"
TEXT, MUTED, BORDER = "#1E293B", "#64748B", "#E2E8F0"
OK, WARN, ERR = "#16A34A", "#F59E0B", "#EF4444"
COND_COLOR = {"Normal": OK, "Warning": WARN, "Critical": ERR}
SERIES_COLORS = ["#2563EB", "#0EA5E9", "#6366F1", "#14B8A6", "#F59E0B", "#EC4899", "#8B5CF6", "#84CC16"]
CSS_PATH = tm.ASSETS_DIR / "style.css"
LOGO_PATH = tm.ASSETS_DIR / "logo.svg"
DATA_MODES = ["Demo Data", "CSV Upload"]

# Small inline SVG icon set (stroke icons, 24x24 grid)
ICON_PATHS = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "server": '<rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="8 12 11 12 12 9 13.5 15 15 12 16 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "drop": '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "monitor": '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>',
    "router": '<rect x="2" y="14" width="20" height="8" rx="2"/><path d="M6.01 18H6"/><path d="M10.01 18H10"/><path d="M15 10v4"/><path d="M17.84 7.17a4 4 0 0 0-5.66 0"/><path d="M20.66 4.34a8 8 0 0 0-11.31 0"/>',
    "switch": '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
}
DEVICE_ICON = {"Router": "router", "Switch": "switch", "Server": "server", "Endpoint Group": "monitor"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def esc(value) -> str:
    """Escape text before putting it inside HTML."""
    return html_lib.escape(str(value))


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICON_PATHS[name]}</svg>')


def render_html(markup: str) -> None:
    """Render HTML in Streamlit (indentation removed so Markdown does not treat it as code)."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def plot(fig: go.Figure, key: str | None = None) -> None:
    """Show a Plotly chart full width (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def style_fig(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    """Apply the blue dashboard theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=12, r=12, t=52 if title else 16, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, Segoe UI, sans-serif", color=TEXT, size=12),
        title=dict(text=title, font=dict(family="Space Grotesk, Manrope, sans-serif", size=17), x=0.01) if title else None,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), hoverlabel=dict(font_family="Manrope"),
    )
    fig.update_xaxes(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F7", linecolor=BORDER, zeroline=False)
    return fig


def section_title(title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{esc(subtitle)}</p>' if subtitle else ""
    render_html(f'<div class="section-title">{esc(title)}</div>{sub}')


def alert_box(kind: str, message_html: str) -> None:
    """kind: ok | warning | critical | info"""
    icon_name = {"ok": "check", "warning": "alert", "critical": "alert", "info": "info"}[kind]
    color = {"ok": OK, "warning": "#B45309", "critical": ERR, "info": DARK_BLUE}[kind]
    render_html(f'<div class="alert-box a-{kind}"><span class="a-icon">{icon(icon_name, 20, color)}</span>'
                f'<div>{message_html}</div></div>')


def condition_badge(condition: str, offline: bool = False) -> str:
    if offline:
        return '<span class="badge b-offline">Offline</span>'
    text = {"Normal": "Healthy", "Warning": "Warning", "Critical": "Critical"}[condition]
    return f'<span class="badge b-{condition.lower()}">{text}</span>'


def level_for(feature: str, value: float) -> str:
    """Normal / Warning / Critical level of ONE parameter value (from the threshold table)."""
    info = tm.FEATURE_INFO[feature]
    if info["direction"] == "high":
        return "Critical" if value >= info["crit"] else "Warning" if value >= info["warn"] else "Normal"
    return "Critical" if value <= info["crit"] else "Warning" if value <= info["warn"] else "Normal"


def health_text(score: float) -> tuple[str, str]:
    """(label, css class) for an overall health score."""
    if score >= 85:
        return "Healthy", "ok"
    if score >= 55:
        return "Degraded", "warn"
    return "At Risk", "err"


# ----------------------------------------------------------------------------
# Data + model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the AI model (first run can take a few seconds) ...")
def get_default_bundle() -> dict:
    """Load the saved model; train and save it automatically if it does not exist."""
    tm.ensure_project_folders()
    if tm.MODEL_PATH.exists():
        try:
            return tm.load_model(tm.MODEL_PATH)
        except Exception:  # corrupted / incompatible file -> retrain
            pass
    return tm.train_and_save_default()


@st.cache_data(show_spinner=False)
def build_demo_data(seed: int, scenario: str) -> pd.DataFrame:
    """Simulated monitoring data (cached so the charts stay stable between clicks)."""
    return tm.generate_monitoring_dataset(seed=seed, scenario=scenario)


def sample_csv_bytes() -> bytes:
    return tm.ensure_sample_dataset().read_bytes()


def active_bundle() -> dict:
    return st.session_state.get("custom_bundle") or get_default_bundle()


def init_state() -> None:
    sim_defaults = {f: float(tm.FEATURE_INFO[f]["default"]) for f in tm.FEATURES}
    defaults = {
        "history": [], "sim_runs": [], "sim_result": None, "sim_values": sim_defaults,
        "sim_device": "Router", "demo_seed": 42, "demo_scenario": "Mixed (realistic)",
        "data_mode": "Demo Data", "csv_result": None, "csv_sig": None, "csv_name": None,
        "custom_bundle": None, "csv_message": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Context: the data every page uses (built once per run)
# ----------------------------------------------------------------------------
def build_context() -> None:
    bundle = active_bundle()
    csv_result = st.session_state.csv_result
    using_csv = (st.session_state.data_mode == "CSV Upload" and csv_result is not None
                 and not csv_result["errors"] and csv_result["df"] is not None)
    if using_csv:
        raw = csv_result["df"]
        source = f"CSV Upload: {st.session_state.csv_name} (stored records, not live traffic)"
    else:
        raw = build_demo_data(st.session_state.demo_seed, st.session_state.demo_scenario)
        source = "Demo Mode — Simulated Network Data"
        if st.session_state.data_mode == "CSV Upload":
            source = "Demo Mode — Simulated Network Data (no valid CSV uploaded yet)"

    df = tm.predict_dataframe(bundle, raw)
    latest = df.groupby("device", sort=False).tail(1).head(12).reset_index(drop=True)
    health = float(latest["health_score"].mean())
    n_critical = int((latest["predicted_condition"] == "Critical").sum())
    n_warning = int((latest["predicted_condition"] == "Warning").sum())
    worst = "Critical" if n_critical else "Warning" if n_warning else "Normal"
    st.session_state["ctx"] = {
        "bundle": bundle, "df": df, "latest": latest, "is_demo": not using_csv, "source": source,
        "x_col": "timestamp" if "timestamp" in df.columns else "record",
        "health": health, "n_critical": n_critical, "n_warning": n_warning, "worst": worst,
    }


def ctx() -> dict:
    return st.session_state["ctx"]


# ----------------------------------------------------------------------------
# Sidebar: data source controls + footer
# ----------------------------------------------------------------------------
def sidebar_controls() -> None:
    render_html('<div class="side-label">DATA SOURCE</div>')
    st.radio("Data mode", DATA_MODES, key="data_mode", label_visibility="collapsed")

    if st.session_state.data_mode == "Demo Data":
        st.selectbox("Demo network condition", list(tm.SCENARIO_WEIGHTS.keys()), key="demo_scenario",
                     help="Controls how often the simulated devices drift into Warning or Critical behaviour.")
        st.caption("Demo Mode — Simulated Network Data")
    else:
        uploaded = st.file_uploader("Upload network CSV", type=["csv"], key="csv_uploader",
                                    help="Required columns: " + ", ".join(tm.FEATURES))
        if uploaded is None:
            st.session_state.csv_result, st.session_state.csv_sig = None, None
            st.caption("No file uploaded. The dashboard shows demo data until a valid CSV is provided.")
        else:
            signature = (uploaded.name, uploaded.size)
            if st.session_state.csv_sig != signature:  # new file -> validate once
                try:
                    st.session_state.csv_result = tm.validate_uploaded_dataframe(tm.read_uploaded_csv(uploaded))
                except ValueError as exc:
                    st.session_state.csv_result = {"df": None, "errors": [str(exc)], "warnings": [],
                                                   "label_col": None, "labels_valid": False}
                st.session_state.csv_sig, st.session_state.csv_name = signature, uploaded.name
            result = st.session_state.csv_result
            for message in result["errors"]:
                st.error(message)
            for message in result["warnings"]:
                st.warning(message)
            if not result["errors"]:
                st.success(f"{len(result['df'])} records validated. These are stored records, not live traffic.")
                if result["labels_valid"]:
                    st.caption(f"Valid label column found: '{result['label_col']}'.")
                    if st.button("Train model on uploaded labels", key="retrain_btn"):
                        try:
                            with st.spinner("Training on your labelled data ..."):
                                st.session_state.custom_bundle = tm.train_and_evaluate(
                                    result["df"], label_col=result["label_col"],
                                    data_source=f"Uploaded CSV: {st.session_state.csv_name}")
                            st.toast("Model retrained on the uploaded labels.")
                        except ValueError as exc:
                            st.error(str(exc))

    if st.session_state.custom_bundle is not None:
        st.info("Using a model trained on your uploaded CSV.")
        if st.button("Switch back to default model", key="reset_model_btn"):
            st.session_state.custom_bundle = None
            st.rerun()

    st.download_button("Download sample dataset (CSV)", data=sample_csv_bytes(),
                       file_name="network_dataset.csv", mime="text/csv", key="dl_sample")
    render_html('<div class="side-footer"><b>AI-Based Network Fault Prediction</b><br>'
                'Computer Networks Mini Project</div>')


# ----------------------------------------------------------------------------
# Reusable UI blocks
# ----------------------------------------------------------------------------
def refresh_data() -> None:
    """Refresh button: generate a fresh batch of simulated data (demo mode only)."""
    if st.session_state.data_mode == "Demo Data":
        st.session_state.demo_seed += 1
    st.toast("Data refreshed.")


def render_header(page_key: str) -> None:
    c = ctx()
    label, css = health_text(c["health"])
    status_css = {"Normal": "ok", "Warning": "warn", "Critical": "err"}[c["worst"]]
    status_text = {"Normal": "No faults predicted", "Warning": f"{c['n_warning']} device(s) in Warning",
                   "Critical": f"{c['n_critical']} device(s) predicted Critical"}[c["worst"]]
    left, right = st.columns([5, 1.3], vertical_alignment="center")
    with left:
        render_html(f"""
        <div class="ng-header">
          <div class="ng-title-row">
            <div class="ng-logo">{icon('shield', 30, '#fff')}</div>
            <div><div class="ng-title">NetGuard <span>AI</span></div>
            <p class="ng-subtitle">Intelligent Network Monitoring &amp; Predictive Fault Detection</p></div>
          </div>
          <div class="ng-chips">
            <span class="chip">{icon('clock', 14)} {datetime.now().strftime('%a, %d %b %Y · %H:%M')}</span>
            <span class="chip {css}"><span class="dot"></span>Network health: {label} ({c['health']:.0f}/100)</span>
            <span class="chip {status_css}"><span class="dot"></span>Fault status: {esc(status_text)}</span>
            <span class="chip info">{icon('server', 14)} {esc(c['source'])}</span>
          </div>
        </div>""")
    with right:
        st.button("Refresh Data", key=f"refresh_{page_key}", on_click=refresh_data, icon=":material/refresh:")
    render_html('<div class="ng-gradient-line"></div>')


def kpi_card(icon_name: str, label: str, value: str, desc: str, tone: str = "blue") -> str:
    return (f'<div class="kpi-card"><div class="kpi-top"><div class="kpi-icon tone-{tone}">{icon(icon_name, 20)}</div>'
            f'<div class="kpi-label">{esc(label)}</div></div><div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-desc">{esc(desc)}</div></div>')


def tone_of(level: str) -> str:
    return {"Normal": "green", "Warning": "amber", "Critical": "red"}[level]


def render_kpis(frame: pd.DataFrame, devices_desc: str) -> None:
    """Six KPI cards calculated from `frame` (one row per device)."""
    n_dev = frame["device"].nunique()
    health = float(frame["health_score"].mean())
    n_crit = int((frame["predicted_condition"] == "Critical").sum())
    n_warn = int((frame["predicted_condition"] == "Warning").sum())
    lat, loss, bw = (float(frame[f].mean()) for f in ("latency", "packet_loss", "bandwidth_utilization"))
    h_label, h_css = health_text(health)
    h_tone = {"ok": "green", "warn": "amber", "err": "red"}[h_css]
    cards = [
        kpi_card("server", "Total Network Devices", str(n_dev), devices_desc),
        kpi_card("shield", "Network Health Score", f"{health:.0f} / 100", f"Status: {h_label}", h_tone),
        kpi_card("alert", "Active Faults (Predicted)", str(n_crit), f"{n_warn} more device(s) in Warning",
                 "red" if n_crit else ("amber" if n_warn else "green")),
        kpi_card("clock", "Average Latency", f"{lat:.1f} ms", f"Warning threshold: {tm.FEATURE_INFO['latency']['warn']} ms",
                 tone_of(level_for("latency", lat))),
        kpi_card("drop", "Packet Loss", f"{loss:.2f} %", f"Warning threshold: {tm.FEATURE_INFO['packet_loss']['warn']} %",
                 tone_of(level_for("packet_loss", loss))),
        kpi_card("zap", "Bandwidth Utilization", f"{bw:.1f} %",
                 f"Warning threshold: {tm.FEATURE_INFO['bandwidth_utilization']['warn']} %",
                 tone_of(level_for("bandwidth_utilization", bw))),
    ]
    for row in (cards[:3], cards[3:]):
        for col, card in zip(st.columns(3), row):
            col.markdown(card, unsafe_allow_html=True)


def device_card(row: pd.Series, simulated: bool) -> str:
    condition = row["predicted_condition"]
    offline = str(row.get("connection_status", "Online")).lower() == "offline"
    css = "offline" if offline else condition.lower()
    bar_color = "#64748B" if offline else COND_COLOR[condition]
    icon_name = DEVICE_ICON.get(str(row.get("device_type", "")), "server")
    tag = '<span class="tag-sim">SIMULATED</span>' if simulated else '<span class="tag-sim">CSV RECORD</span>'
    cells = [("Latency", f"{row['latency']:.0f} ms"), ("Pkt loss", f"{row['packet_loss']:.1f} %"),
             ("Bandwidth", f"{row['bandwidth_utilization']:.0f} %"), ("CPU", f"{row['cpu_utilization']:.0f} %"),
             ("Memory", f"{row['memory_utilization']:.0f} %"), ("Health", f"{row['health_score']:.0f}/100")]
    grid = "".join(f'<div class="cell"><span>{k}</span><b>{v}</b></div>' for k, v in cells)
    status = "Offline" if offline else "Online"
    return (f'<div class="device-card s-{css}"><div class="dev-head"><div class="dev-icon">{icon(icon_name, 22)}</div>'
            f'<div><div class="dev-name">{esc(row["device"])}</div>'
            f'<div class="dev-type">{esc(row.get("device_type", "Network Device"))} · {status}</div></div>'
            f'{condition_badge(condition, offline)}</div><div class="dev-grid">{grid}</div>'
            f'<div class="health-bar"><div style="width:{max(row["health_score"], 3):.0f}%;background:{bar_color}"></div></div>'
            f'<div class="dev-foot">AI prediction: <b>{condition}</b> · confidence {row["confidence"]:.1f}% {tag}</div></div>')


def probability_bars(row) -> str:
    html = ""
    for label in tm.LABELS:
        value = float(row[f"prob_{label.lower()}"])
        html += (f'<div class="prob-row"><span class="lbl">{label}</span><span class="bar">'
                 f'<div style="width:{value:.1f}%;background:{COND_COLOR[label]}"></div></span>'
                 f'<span class="val">{value:.1f}%</span></div>')
    return html


def result_card(row: pd.Series, device: str, simulated: bool = True) -> None:
    """The 'AI Network Fault Prediction' result card."""
    condition = row["predicted_condition"]
    record = {f: row[f] for f in tm.FEATURES}
    reasons = "".join(f"<li>{esc(r)}</li>" for r in tm.reason_list(record))
    risk_css = {"Low": "ok", "Medium": "warn", "High": "err"}[row["risk_level"]]
    agree = ("agrees with" if row["rule_based_label"] == condition else
             f"differs from (rule says {row['rule_based_label']})")
    render_html(f"""
    <div class="result-card">
      <div class="rc-title">AI Network Fault Prediction · {esc(device)}</div>
      <div class="rc-status c-{condition.lower()}">Predicted Status: {condition}</div>
      <div class="rc-meta">
        <span class="chip {risk_css}"><span class="dot"></span>Risk level: {row['risk_level']}</span>
        <span class="chip info">Prediction confidence: {row['confidence']:.1f}%</span>
        <span class="chip">Health score: {row['health_score']:.0f}/100</span>
      </div>
      {probability_bars(row)}
      <div style="margin-top:.9rem;font-weight:800;font-size:.85rem;">Possible reasons (from the input parameters)</div>
      <ul class="reason-list">{reasons}</ul>
      <div class="disclaimer">This is a <b>prediction</b> made by a machine-learning model on
      {'simulated' if simulated else 'stored'} measurements. It is not a confirmed physical device failure.
      The model output {agree} the reference rule-based severity check.</div>
    </div>""")


def threshold_alerts(row: pd.Series, device: str) -> None:
    """Visual alerts for the predicted condition and for parameters above thresholds."""
    condition = row["predicted_condition"]
    record = {f: row[f] for f in tm.FEATURES}
    exceeded = [i for i in tm.explain_record(record) if i["points"] > 0]
    if condition == "Critical":
        alert_box("critical", f"<b>Critical alert · {esc(device)}</b><br>The model predicts a high risk of a network fault "
                              f"(confidence {row['confidence']:.1f}%). Investigate the parameters listed below.")
    elif condition == "Warning":
        alert_box("warning", f"<b>Warning · {esc(device)}</b><br>Network performance may be degrading "
                             f"(confidence {row['confidence']:.1f}%). Monitor this device closely.")
    else:
        alert_box("ok", f"<b>Normal · {esc(device)}</b><br>No fault is predicted for the current parameters.")
    if exceeded and condition == "Normal":
        alert_box("info", f"{len(exceeded)} parameter(s) are above their warning threshold, but the combined pattern is "
                          "still classified as Normal.")
    elif exceeded:
        names = ", ".join(i["name"].lower() for i in exceeded[:4])
        alert_box("info", f"Parameters above configured thresholds: {esc(names)}"
                          f"{' and more' if len(exceeded) > 4 else ''}.")


def add_history(row: pd.Series, device: str, source: str) -> None:
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Network Device": device,
        "Latency (ms)": round(float(row["latency"]), 1), "Packet Loss (%)": round(float(row["packet_loss"]), 2),
        "Bandwidth Util. (%)": round(float(row["bandwidth_utilization"]), 1),
        "CPU Util. (%)": round(float(row["cpu_utilization"]), 1),
        "Predicted Condition": row["predicted_condition"], "Risk Level": row["risk_level"],
        "Confidence (%)": round(float(row["confidence"]), 1), "Source": source,
    })


def metrics_row(bundle: dict) -> None:
    m = bundle["metrics"]
    for col, (label, key) in zip(st.columns(4), [("Accuracy", "Accuracy"), ("Precision (weighted)", "Precision"),
                                                 ("Recall (weighted)", "Recall"), ("F1-score (weighted)", "F1-score")]):
        col.metric(label, f"{m[key] * 100:.2f}%")


# ----------------------------------------------------------------------------
# PAGE 1: Overview
# ----------------------------------------------------------------------------
def page_overview() -> None:
    render_header("overview")
    c = ctx()
    section_title("Network Overview", "KPIs are calculated from the latest reading of every device in the current dataset.")
    render_kpis(c["latest"], "Devices in the current dataset")

    sim = st.session_state.sim_result
    if sim:
        cond = sim["record"]["predicted_condition"]
        kind = {"Normal": "ok", "Warning": "warning", "Critical": "critical"}[cond]
        alert_box(kind, f"<b>Latest Simulation Lab run ({esc(sim['time'])})</b>: {esc(sim['device'])} predicted "
                        f"<b>{cond}</b> with {sim['record']['confidence']:.1f}% confidence.")

    left, right = st.columns([1, 1.25])
    with left:
        with st.container(border=True):
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=c["health"], number=dict(suffix=" / 100", font=dict(family="Space Grotesk, Manrope, sans-serif", size=34)),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=PRIMARY, thickness=0.28),
                           steps=[dict(range=[0, 55], color="#FEE2E2"), dict(range=[55, 85], color="#FEF3C7"),
                                  dict(range=[85, 100], color="#DCFCE7")])))
            plot(style_fig(gauge, "Overall Network Health Score", 300), key="ov_gauge")
    with right:
        with st.container(border=True):
            latest = c["latest"]
            fig = go.Figure(go.Bar(
                x=latest["device"], y=latest["health_score"], marker_color=[COND_COLOR[v] for v in latest["predicted_condition"]],
                text=latest["predicted_condition"], textposition="outside",
                hovertemplate="%{x}<br>Health score: %{y}<extra></extra>"))
            fig.update_yaxes(range=[0, 115], title="Health score")
            plot(style_fig(fig, "Device Health (colour = predicted condition)", 300), key="ov_devices")

    section_title("Active Alerts", "Generated from the AI predictions of the latest readings. No e-mail/SMS notifications are sent.")
    flagged = c["latest"][c["latest"]["predicted_condition"] != "Normal"].sort_values("severity_points", ascending=False)
    if flagged.empty:
        alert_box("ok", "<b>All devices look healthy.</b> No Warning or Critical predictions in the latest readings.")
    for _, row in flagged.iterrows():
        top = tm.reason_list({f: row[f] for f in tm.FEATURES}, 2)
        kind = "critical" if row["predicted_condition"] == "Critical" else "warning"
        alert_box(kind, f"<b>{esc(row['device'])} · {row['predicted_condition']}</b> "
                        f"(confidence {row['confidence']:.1f}%)<br>{esc(' | '.join(top))}")


# ----------------------------------------------------------------------------
# PAGE 2: Network Monitoring
# ----------------------------------------------------------------------------
def topology_figure(latest: pd.DataFrame) -> go.Figure | None:
    """Simple topology diagram for the demo network (Router - 2 switches - clients/server)."""
    positions = {"Router": (0, 2), "Switch 1": (-1.6, 1), "Switch 2": (1.6, 1),
                 "Client Devices": (-1.6, 0), "Server": (1.6, 0)}
    links = [("Router", "Switch 1"), ("Router", "Switch 2"), ("Switch 1", "Client Devices"), ("Switch 2", "Server")]
    rows = {r["device"]: r for _, r in latest.iterrows()}
    if not all(name in rows for name in positions):
        return None
    fig = go.Figure()
    for a, b in links:
        fig.add_trace(go.Scatter(x=[positions[a][0], positions[b][0]], y=[positions[a][1], positions[b][1]],
                                 mode="lines", line=dict(color="#93C5FD", width=4), hoverinfo="skip", showlegend=False))
    names = list(positions)
    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in names], y=[positions[n][1] for n in names], mode="markers+text",
        text=names, textposition="bottom center", textfont=dict(family="Space Grotesk, Manrope, sans-serif", size=13),
        marker=dict(size=46, color=[COND_COLOR[rows[n]["predicted_condition"]] for n in names],
                    line=dict(color="#fff", width=3)),
        customdata=[[rows[n]["predicted_condition"], rows[n]["latency"], rows[n]["packet_loss"]] for n in names],
        hovertemplate="<b>%{text}</b><br>Predicted: %{customdata[0]}<br>Latency: %{customdata[1]:.0f} ms<br>"
                      "Packet loss: %{customdata[2]:.1f}%<extra></extra>", showlegend=False))
    fig.update_xaxes(visible=False, range=[-2.6, 2.6])
    fig.update_yaxes(visible=False, range=[-0.7, 2.5])
    return style_fig(fig, "Simulated Network Topology (colour = AI predicted condition)", 360)


def page_monitoring() -> None:
    render_header("monitoring")
    c = ctx()
    latest = c["latest"]
    label = "Simulated" if c["is_demo"] else "Uploaded CSV"
    section_title("Network Device Monitoring", f"{label} readings. Cards show the most recent reading of each device and the AI-predicted condition.")
    render_html('<div class="alert-box a-info"><span class="a-icon">' + icon("info", 20, DARK_BLUE) + '</span><div>'
                + ("<b>Demo Mode — Simulated Network Data.</b> No real router or switch is being monitored."
                   if c["is_demo"] else "<b>CSV records.</b> These are stored measurements, not live network traffic.")
                + "</div></div>")

    if c["is_demo"]:
        fig = topology_figure(latest)
        if fig is not None:
            with st.container(border=True):
                plot(fig, key="mon_topology")

    view = st.radio("Show devices", ["All", "Warning / Critical only", "Normal only"], horizontal=True)
    shown = latest
    if view == "Warning / Critical only":
        shown = latest[latest["predicted_condition"] != "Normal"]
    elif view == "Normal only":
        shown = latest[latest["predicted_condition"] == "Normal"]
    if shown.empty:
        st.info("No devices match this filter.")
    for start in range(0, len(shown), 3):
        for col, (_, row) in zip(st.columns(3), shown.iloc[start:start + 3].iterrows()):
            col.markdown(device_card(row, c["is_demo"]), unsafe_allow_html=True)

    with st.expander("Latest readings table"):
        cols = ["device", "device_type", "connection_status"] + tm.FEATURES + ["predicted_condition", "confidence", "risk_level"]
        table(latest[[col for col in cols if col in latest.columns]].round(2))


# ----------------------------------------------------------------------------
# PAGE 3: AI Fault Prediction
# ----------------------------------------------------------------------------
def page_prediction() -> None:
    render_header("prediction")
    c = ctx()
    bundle = c["bundle"]
    tab_live, tab_model, tab_results = st.tabs(["Live Prediction", "Model Performance", "Prediction Results"])

    with tab_live:
        section_title("Prediction for a device", "Choose a device to see the AI prediction for its latest reading.")
        device = st.selectbox("Device", list(c["latest"]["device"]), key="pred_device")
        row = c["latest"][c["latest"]["device"] == device].iloc[0]
        left, right = st.columns([1.1, 1])
        with left:
            result_card(row, device, simulated=c["is_demo"])
        with right:
            threshold_alerts(row, device)
            items = tm.explain_record({f: row[f] for f in tm.FEATURES})
            frame = pd.DataFrame([{"Parameter": i["name"], "Value": f"{i['value']:g} {i['unit']}", "Level": i["status"]} for i in items])
            table(frame)
            if st.button("Add this prediction to Fault History", key="pred_log"):
                add_history(row, device, "Monitoring snapshot" if c["is_demo"] else "CSV record")
                st.toast("Prediction added to Fault History.")

    with tab_model:
        section_title("Model evaluation", "All numbers below are calculated from predictions on a held-out test set.")
        alert_box("info", f"<b>Selected model: {esc(bundle['model_name'])}</b> · training data: {esc(bundle['data_source'])}<br>"
                          f"Train rows: {bundle['train_rows']} · test rows: {bundle['test_rows']} · random seed: {bundle['seed']} · "
                          f"trained at {esc(bundle['trained_at'])}.<br>The initial model is trained on <b>synthetic data</b>, so these scores "
                          "show how well it learned the demonstration rules, not how it would perform on a real network.")
        metrics_row(bundle)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                labels = bundle["labels_present"]
                cm = np.array(bundle["confusion_matrix"])
                fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto",
                                color_continuous_scale=[[0, "#EFF6FF"], [1, DARK_BLUE]],
                                labels=dict(x="Predicted", y="Actual", color="Count"))
                plot(style_fig(fig, "Confusion Matrix (test set)", 360), key="pred_cm")
        with c2:
            with st.container(border=True):
                importance = bundle.get("feature_importance")
                if importance:
                    imp = pd.Series(importance).sort_values()
                    fig = go.Figure(go.Bar(x=imp.values, y=[tm.FEATURE_INFO[f]["short"] for f in imp.index],
                                           orientation="h", marker_color=PRIMARY))
                    plot(style_fig(fig, "Feature Importance (model)", 360), key="pred_imp")
                else:
                    st.info("Feature importance is not available for this model type.")
        section_title("Algorithm comparison", "Same train/test split for every model; the best weighted F1-score is selected (Random Forest is preferred on near-ties).")
        comp = pd.DataFrame(bundle["comparison"]).copy()
        for col in ("Accuracy", "Precision", "Recall", "F1-score"):
            comp[col] = (comp[col] * 100).round(2).astype(str) + "%"
        table(comp)
        section_title("Per-class results")
        per_class = pd.DataFrame(bundle["per_class"]).T[["precision", "recall", "f1-score", "support"]].round(3)
        per_class.insert(0, "Class", per_class.index)
        table(per_class)

    with tab_results:
        section_title("Predictions for the current dataset", f"{len(c['df'])} records · source: {c['source']}")
        out = c["df"].copy()
        show_cols = [col for col in ["timestamp", "record", "device"] if col in out.columns] + tm.FEATURES + \
                    ["predicted_condition", "risk_level", "confidence", "health_score"]
        out = out[show_cols]
        numeric_cols = out.select_dtypes("number").columns
        out[numeric_cols] = out[numeric_cols].round(2)
        table(out)
        st.download_button("Download prediction results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="netguard_prediction_results.csv", mime="text/csv", key="dl_predictions")


# ----------------------------------------------------------------------------
# PAGE 4: Fault History
# ----------------------------------------------------------------------------
def clear_history() -> None:
    st.session_state.history = []


def log_snapshot() -> None:
    c = ctx()
    for _, row in c["latest"].iterrows():
        add_history(row, row["device"], "Demo snapshot" if c["is_demo"] else "CSV snapshot")
    st.toast(f"Logged {len(c['latest'])} device predictions.")


def page_history() -> None:
    render_header("history")
    section_title("Fault History & Alerts", "Predictions made during this browser session (kept in Streamlit session state).")
    b1, b2, _ = st.columns([1.3, 1, 3])
    b1.button("Log current device snapshot", key="log_snapshot", on_click=log_snapshot)
    b2.button("Clear history", key="clear_history", on_click=clear_history)

    history = pd.DataFrame(st.session_state.history)
    if history.empty:
        alert_box("info", "No predictions have been recorded yet. Run a prediction in the <b>Simulation Lab</b> or click "
                          "<b>Log current device snapshot</b>.")
        return
    counts = history["Predicted Condition"].value_counts()
    for col, label in zip(st.columns(4), ["Total"] + tm.LABELS):
        col.metric(label, len(history) if label == "Total" else int(counts.get(label, 0)))

    choice = st.radio("Filter by predicted condition", ["All"] + tm.LABELS, horizontal=True, key="hist_filter")
    filtered = history if choice == "All" else history[history["Predicted Condition"] == choice]
    table(filtered.iloc[::-1])  # newest first
    st.download_button("Download fault history (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="netguard_fault_history.csv", mime="text/csv", key="dl_history")

    section_title("Recent alerts")
    recent = history[history["Predicted Condition"] != "Normal"].tail(5).iloc[::-1]
    if recent.empty:
        alert_box("ok", "No Warning or Critical predictions in the history.")
    for _, r in recent.iterrows():
        kind = "critical" if r["Predicted Condition"] == "Critical" else "warning"
        alert_box(kind, f"<b>{esc(r['Predicted Condition'])} · {esc(r['Network Device'])}</b> at {esc(r['Timestamp'])} "
                        f"(latency {r['Latency (ms)']} ms, packet loss {r['Packet Loss (%)']}%, confidence {r['Confidence (%)']}%)")


# ----------------------------------------------------------------------------
# PAGE 5: Network Analytics
# ----------------------------------------------------------------------------
def threshold_line(fig: go.Figure, feature: str, level: str = "warn") -> None:
    info = tm.FEATURE_INFO[feature]
    fig.add_hline(y=info[level], line_dash="dash", line_color=WARN if level == "warn" else ERR,
                  annotation_text=f"{'Warning' if level == 'warn' else 'Critical'} threshold", annotation_position="top left")


def page_analytics() -> None:
    render_header("analytics")
    c = ctx()
    df, x = c["df"], c["x_col"]
    section_title("Network Analytics", "Interactive charts - hover for details, drag to zoom, double-click to reset.")
    devices = list(dict.fromkeys(df["device"]))
    chosen = st.multiselect("Devices to include", devices, default=devices[:8])
    data = df[df["device"].isin(chosen)]
    if data.empty:
        st.info("Select at least one device.")
        return
    x_title = "Time" if x == "timestamp" else "Record number"

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):  # Chart 1
        fig = px.line(data, x=x, y="latency", color="device", color_discrete_sequence=SERIES_COLORS,
                      labels={"latency": "Latency (ms)", x: x_title})
        threshold_line(fig, "latency", "warn")
        plot(style_fig(fig, "1 · Network Latency", 350), key="an_lat")
    with c2, st.container(border=True):  # Chart 2
        avg = data.groupby(x, as_index=False)["packet_loss"].mean()
        colors = [COND_COLOR[level_for("packet_loss", v)] for v in avg["packet_loss"]]
        fig = go.Figure(go.Bar(x=avg[x], y=avg["packet_loss"], marker_color=colors,
                               hovertemplate="%{x}<br>Packet loss: %{y:.2f}%<extra></extra>"))
        fig.update_yaxes(title="Average packet loss (%)")
        plot(style_fig(fig, "2 · Packet Loss (average of selected devices)", 350), key="an_loss")

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):  # Chart 3
        avg = data.groupby(x, as_index=False)["bandwidth_utilization"].mean()
        fig = px.area(avg, x=x, y="bandwidth_utilization", labels={"bandwidth_utilization": "Bandwidth utilization (%)", x: x_title},
                      color_discrete_sequence=[PRIMARY])
        threshold_line(fig, "bandwidth_utilization", "warn")
        fig.update_yaxes(range=[0, 105])
        plot(style_fig(fig, "3 · Bandwidth Utilization (average)", 350), key="an_bw")
    with c4, st.container(border=True):  # Chart 4
        means = data.groupby("device", sort=False)[["cpu_utilization", "memory_utilization"]].mean().reset_index()
        fig = go.Figure()
        fig.add_bar(x=means["device"], y=means["cpu_utilization"], name="CPU %", marker_color=PRIMARY)
        fig.add_bar(x=means["device"], y=means["memory_utilization"], name="Memory %", marker_color="#93C5FD")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Average utilization (%)", range=[0, 105])
        plot(style_fig(fig, "4 · CPU vs Memory Utilization", 350), key="an_res")

    counts = data["predicted_condition"].value_counts().reindex(tm.LABELS, fill_value=0)
    c5, c6 = st.columns(2)
    with c5, st.container(border=True):  # Chart 5
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, sort=False,
                               marker=dict(colors=[COND_COLOR[k] for k in counts.index], line=dict(color="#fff", width=3)),
                               textinfo="label+percent"))
        plot(style_fig(fig, "5 · Network Health Distribution", 350), key="an_pie")
    with c6, st.container(border=True):  # Chart 6
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[COND_COLOR[k] for k in counts.index],
                               text=counts.values, textposition="outside"))
        fig.update_yaxes(title="Number of records", range=[0, max(counts.max() * 1.15, 1)])
        plot(style_fig(fig, "6 · Fault Prediction Results", 350), key="an_counts")

    with st.container(border=True):  # Chart 7
        corr = data[tm.FEATURES].corr().fillna(0)
        short = [tm.FEATURE_INFO[f]["short"] for f in tm.FEATURES]
        fig = px.imshow(corr.values, x=short, y=short, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=[[0, ERR], [0.5, "#FFFFFF"], [1, PRIMARY]])
        plot(style_fig(fig, "7 · Network Performance Correlation", 480), key="an_corr")

    runs = pd.DataFrame(st.session_state.sim_runs)
    if not runs.empty:
        section_title("Simulation Lab runs (this session)", "Every click on 'Run AI Fault Prediction' adds a point here.")
        with st.container(border=True):
            fig = go.Figure(go.Bar(x=list(range(1, len(runs) + 1)), y=runs["health_score"],
                                   marker_color=[COND_COLOR[v] for v in runs["predicted_condition"]],
                                   text=runs["predicted_condition"], textposition="outside",
                                   hovertemplate="Run %{x}<br>Health score: %{y}<extra></extra>"))
            fig.update_xaxes(title="Run number", dtick=1)
            fig.update_yaxes(title="Health score", range=[0, 115])
            plot(style_fig(fig, "Health score of simulated scenarios", 320), key="an_runs")


# ----------------------------------------------------------------------------
# PAGE 6: Simulation Lab
# ----------------------------------------------------------------------------
def _set_sim_values(values: dict) -> None:
    for feature in tm.FEATURES:
        st.session_state[f"sim_{feature}"] = float(values[feature])
    st.session_state.sim_values = {f: float(values[f]) for f in tm.FEATURES}


def reset_simulation() -> None:
    _set_sim_values({f: tm.FEATURE_INFO[f]["default"] for f in tm.FEATURES})
    st.session_state.sim_result = None


def load_scenario(name: str) -> None:
    _set_sim_values(tm.SIM_SCENARIOS[name])


def run_simulation() -> None:
    record = {f: float(st.session_state[f"sim_{f}"]) for f in tm.FEATURES}
    record["connection_status"] = "Offline" if record["packet_loss"] >= 95 else "Online"
    predicted = tm.predict_dataframe(active_bundle(), pd.DataFrame([record])).iloc[0]
    device = st.session_state.sim_device
    st.session_state.sim_result = {"record": predicted.to_dict(), "device": device, "inputs": record,
                                   "time": datetime.now().strftime("%H:%M:%S")}
    st.session_state.sim_runs.append({"device": device, "predicted_condition": predicted["predicted_condition"],
                                      "health_score": float(predicted["health_score"])})
    add_history(predicted, device, "Simulation Lab")
    if predicted["predicted_condition"] != "Normal":
        st.toast(f"{predicted['predicted_condition']} alert for {device}!", icon="⚠️")


def page_simulation() -> None:
    render_header("simulation")
    section_title("Simulation Lab", "Change the network parameters, then let the trained AI model predict the condition.")
    alert_box("info", "<b>Simulated inputs.</b> The values you set here are hypothetical measurements. "
                      "The prediction is generated by the trained model at the moment you click the button.")

    # Keep slider values when the user visits other pages and comes back.
    for feature in tm.FEATURES:
        st.session_state.setdefault(f"sim_{feature}", st.session_state.sim_values[feature])

    st.write("**Quick scenarios**")
    sc_cols = st.columns(3)
    for col, name in zip(sc_cols, tm.SIM_SCENARIOS):
        col.button(f"Load: {name}", key=f"scn_{name}", on_click=load_scenario, args=(name,))

    left, right = st.columns([1.05, 1])
    with left:
        with st.container(border=True):
            st.selectbox("Simulated device", list(tm.DEVICE_PROFILES.keys()), key="sim_device")
            s1, s2 = st.columns(2)
            for i, feature in enumerate(tm.FEATURES):
                info = tm.FEATURE_INFO[feature]
                with (s1 if i % 2 == 0 else s2):
                    st.slider(f"{info['name']} ({info['unit']})", float(info["min"]), float(info["max"]),
                              step=float(info["step"]), key=f"sim_{feature}")
            st.session_state.sim_values = {f: float(st.session_state[f"sim_{f}"]) for f in tm.FEATURES}
            b1, b2 = st.columns(2)
            b1.button("Run AI Fault Prediction", type="primary", key="run_sim", on_click=run_simulation, width="stretch")
            b2.button("Reset Simulation", key="reset_sim", on_click=reset_simulation, width="stretch")

    result = st.session_state.sim_result
    with right:
        if result is None:
            alert_box("info", "No prediction yet. Click <b>Run AI Fault Prediction</b> to analyse the current slider values.")
        else:
            row = pd.Series(result["record"])
            current = {f: float(st.session_state[f"sim_{f}"]) for f in tm.FEATURES}
            if any(abs(current[f] - result["inputs"][f]) > 1e-9 for f in tm.FEATURES):
                alert_box("warning", "The sliders changed after the last run. Click <b>Run AI Fault Prediction</b> to update the result.")
            result_card(row, result["device"], simulated=True)

    if result is not None:
        row = pd.Series(result["record"])
        section_title("Simulation results", f"Run at {result['time']} for {result['device']} (KPIs are calculated from the simulated input).")
        render_kpis(pd.DataFrame([result["record"]]).assign(device=result["device"]), "Simulated device")
        a1, a2 = st.columns([1, 1])
        with a1:
            threshold_alerts(row, result["device"])
            items = tm.explain_record(result["inputs"])
            frame = pd.DataFrame([{
                "Parameter": i["name"], "Value": f"{i['value']:g} {i['unit']}", "Level": i["status"],
                "Warning limit": f"{tm.FEATURE_INFO[i['feature']]['warn']:g}", "Critical limit": f"{tm.FEATURE_INFO[i['feature']]['crit']:g}",
            } for i in items])
            table(frame)
        with a2:
            with st.container(border=True):
                items = tm.explain_record(result["inputs"])
                fig = go.Figure(go.Bar(
                    x=[i["points"] for i in items][::-1], y=[tm.FEATURE_INFO[i["feature"]]["short"] for i in items][::-1],
                    orientation="h", marker_color=[COND_COLOR[i["status"]] for i in items][::-1]))
                fig.update_xaxes(range=[0, 2.2], tickvals=[0, 1, 2], ticktext=["Normal", "Warning", "Critical"])
                plot(style_fig(fig, "Which parameters drive the prediction?", 340), key="sim_points")


# ----------------------------------------------------------------------------
# PAGE 7: About Project
# ----------------------------------------------------------------------------
ARCH_STAGES = [
    ("Network Devices / Simulated Network", "Routers, switches, servers and clients (here: simulated devices or CSV records)."),
    ("Network Performance Data Collection", "Latency, packet loss, bandwidth, throughput, CPU, memory, traffic and error rate."),
    ("Data Preprocessing", "Check for missing or invalid values and convert everything to numbers."),
    ("Feature Extraction", "Select the 8 performance parameters that the model uses as input features."),
    ("Machine Learning Model", "Random Forest classifier trained with Scikit-learn and saved with Joblib."),
    ("Network Fault Prediction", "The model outputs Normal, Warning or Critical with class probabilities."),
    ("Risk Classification", "Prediction is mapped to a Low, Medium or High risk level."),
    ("Alert Generation", "Warning and Critical predictions create visual alerts and history entries."),
    ("Dashboard Visualization", "KPI cards, device cards and Plotly charts in this Streamlit dashboard."),
]


def page_about() -> None:
    render_header("about")
    bundle = ctx()["bundle"]
    left, right = st.columns([1, 1.1])
    with left:
        section_title("System Architecture", "Conceptual data flow of the project.")
        steps = []
        for i, (title, desc) in enumerate(ARCH_STAGES, start=1):
            steps.append(f'<div class="arch-step"><div class="arch-num">{i:02d}</div><div><b>{esc(title)}</b><small>{esc(desc)}</small></div></div>')
        render_html('<div class="arch-arrow">&#8595;</div>'.join(steps))
    with right:
        section_title("Project Information")
        render_html(f"""
        <div class="ng-card"><h4>Project title</h4><p><b>AI-Based Network Fault Prediction</b> (Computer Networks Mini Project)</p>
        <h4>Objective</h4><p>Use machine learning to analyse network performance parameters, give early warnings about abnormal
        behaviour and help administrators reduce unexpected downtime. Conditions are classified as <b>Normal</b>,
        <b>Warning</b> or <b>Critical (fault predicted)</b>.</p>
        <h4>Technologies used</h4><p>Python · Streamlit · Pandas · NumPy · Scikit-learn · Joblib · Plotly · custom CSS · Google Fonts (Manrope, Space Grotesk, JetBrains Mono)</p>
        <h4>Machine learning algorithm</h4><p>Primary model: <b>Random Forest Classifier</b>. Decision Tree and Logistic Regression are trained and compared on the
        same split. Currently deployed model: <b>{esc(bundle['model_name'])}</b> (accuracy {bundle['metrics']['Accuracy'] * 100:.2f}% on the synthetic test set).</p>
        <h4>Role of Cisco Packet Tracer</h4><p>Cisco Packet Tracer can be used to demonstrate network topology and simulated connectivity.
        Python and Streamlit implement the AI-based network fault prediction logic. The current prototype uses simulated network
        measurements or CSV input and does <b>not</b> automatically control or read physical network devices, and it is not
        integrated with Packet Tracer.</p></div>""")

    section_title("Network parameters and demonstration thresholds",
                  "These thresholds are demonstration assumptions used to label the synthetic data. They are not universal networking standards. Edit FEATURE_INFO in train_model.py to change them.")
    rows = []
    for f in tm.FEATURES:
        info = tm.FEATURE_INFO[f]
        sign = ">=" if info["direction"] == "high" else "<="
        rows.append({"Parameter": info["name"], "Unit": info["unit"], "Warning when": f"{sign} {info['warn']:g}",
                     "Critical when": f"{sign} {info['crit']:g}", "Allowed range": f"{info['min']:g} - {info['max']:g}"})
    table(pd.DataFrame(rows))
    alert_box("info", f"<b>Labelling rule:</b> each parameter earns 0, 1 or 2 severity points (normal / warning / critical level). "
                      f"Total points &ge; {tm.WARNING_MIN_POINTS} &rarr; <b>Warning</b>, &ge; {tm.CRITICAL_MIN_POINTS} &rarr; <b>Critical</b> "
                      f"(maximum {tm.MAX_POINTS}). Health score = 100 &times; (1 &minus; points / {tm.MAX_POINTS}).")

    section_title("Limitations")
    render_html("""
    <div class="ng-card"><ul>
    <li>The model is trained on <b>synthetic data</b>; its labels come from a rule-based function, so the high accuracy mainly shows that it learned those rules.</li>
    <li>The dashboard analyses simulated measurements or uploaded CSV records. It is <b>not</b> connected to live routers, switches or SNMP/NetFlow feeds.</li>
    <li>A Critical result is a <b>fault prediction</b>, not a confirmed hardware or link failure.</li>
    <li>No e-mail, SMS or push notifications are implemented. Alerts are shown inside the dashboard only.</li>
    <li>Real deployments would need real labelled data, time-series features and re-validation of thresholds.</li>
    </ul></div>""")


# ----------------------------------------------------------------------------
# App entry point
# ----------------------------------------------------------------------------
def bootstrap() -> None:
    st.set_page_config(page_title="NetGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    tm.ensure_project_folders()
    load_css()
    init_state()
    if LOGO_PATH.exists():
        try:
            st.logo(str(LOGO_PATH), size="large")
        except Exception:
            pass


def main() -> None:
    bootstrap()
    pages = [
        st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(page_monitoring, title="Network Monitoring", icon=":material/lan:"),
        st.Page(page_prediction, title="AI Fault Prediction", icon=":material/psychology:"),
        st.Page(page_history, title="Fault History", icon=":material/history:"),
        st.Page(page_analytics, title="Network Analytics", icon=":material/insights:"),
        st.Page(page_simulation, title="Simulation Lab", icon=":material/science:"),
        st.Page(page_about, title="About Project", icon=":material/info:"),
    ]
    navigation = st.navigation(pages)
    with st.sidebar:
        sidebar_controls()
    build_context()
    navigation.run()


if __name__ == "__main__":
    main()
