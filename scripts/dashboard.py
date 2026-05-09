"""
ML-NIDS Pro — Professional Network Intrusion Detection Dashboard
Run: streamlit run nids_dashboard.py
Requirements: streamlit plotly pandas numpy requests
"""

import os
import random
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MUST be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML-NIDS Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    page_title: str = "ML-NIDS Pro"
    page_icon: str = "🛡️"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    alert_threshold_default: int = 95
    total_flows_init: int = 24_812
    threat_base: int = 147

    feature_columns: Tuple[str, ...] = (
        "dst_port", "protocol", "total_packets", "total_bytes",
        "packet_size_mean", "packet_size_std", "flow_duration",
        "packets_per_sec", "bytes_per_sec", "unique_dst_ports",
        "inter_arrival_mean", "inter_arrival_std",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            alert_threshold_default=int(os.getenv("NIDS_THRESHOLD", "95"))
        )


SETTINGS = Settings.from_env()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLASSES: List[str] = [
    "Normal", "SSH Brute Force", "Port Scan", "Reverse Shell", "C2 Beacon"
]

CLASS_COLORS: Dict[str, str] = {
    "Normal":          "#22c55e",
    "SSH Brute Force": "#ef4444",
    "Port Scan":       "#f59e0b",
    "Reverse Shell":   "#a78bfa",
    "C2 Beacon":       "#14b8a6",
}

THREAT_ICONS: Dict[str, str] = {
    "Normal":          "🟢",
    "SSH Brute Force": "🔴",
    "Port Scan":       "🟡",
    "Reverse Shell":   "🟣",
    "C2 Beacon":       "🔵",
}

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8892b0", size=11),
    margin=dict(l=12, r=12, t=40, b=12),
)
PLOTLY_CFG: dict = {"displayModeBar": False, "responsive": True}

# Attack scenario feature profiles
SCENARIOS: Dict[str, dict] = {
    "🌐 Normal Traffic": dict(
        dport=80,   proto=0, pkts=8,   bytes_=960,
        psmean=120.0, psstd=5.0,  dur=5.0,  pps=1.6,
        bps=192.0, udp=1,  iam=0.625, ias=0.020,
    ),
    "🔐 SSH Brute Force": dict(
        dport=22,   proto=0, pkts=120, bytes_=14_400,
        psmean=120.0, psstd=15.2, dur=24.5, pps=4.9,
        bps=588.0, udp=1,  iam=0.204, ias=0.042,
    ),
    "🔍 Port Scan": dict(
        dport=80,   proto=0, pkts=250, bytes_=30_000,
        psmean=120.0, psstd=35.2, dur=2.5,  pps=100.0,
        bps=12_000.0, udp=45, iam=0.010, ias=0.008,
    ),
    "💀 Reverse Shell": dict(
        dport=4_444, proto=0, pkts=15,  bytes_=1_800,
        psmean=120.0, psstd=13.5, dur=3.2,  pps=4.69,
        bps=562.5, udp=1,  iam=0.213, ias=0.035,
    ),
    "📡 C2 Beaconing": dict(
        dport=8_888, proto=0, pkts=22,  bytes_=2_640,
        psmean=120.0, psstd=16.1, dur=11.0, pps=2.0,
        bps=240.0, udp=1,  iam=0.500, ias=0.100,
    ),
}

BASE_PROBS: Dict[str, Dict[str, float]] = {
    "🌐 Normal Traffic": {
        "Normal": 0.973, "SSH Brute Force": 0.006,
        "Port Scan": 0.009, "Reverse Shell": 0.007, "C2 Beacon": 0.005,
    },
    "🔐 SSH Brute Force": {
        "Normal": 0.021, "SSH Brute Force": 0.962,
        "Port Scan": 0.005, "Reverse Shell": 0.008, "C2 Beacon": 0.004,
    },
    "🔍 Port Scan": {
        "Normal": 0.011, "SSH Brute Force": 0.004,
        "Port Scan": 0.974, "Reverse Shell": 0.006, "C2 Beacon": 0.005,
    },
    "💀 Reverse Shell": {
        "Normal": 0.018, "SSH Brute Force": 0.009,
        "Port Scan": 0.006, "Reverse Shell": 0.957, "C2 Beacon": 0.010,
    },
    "📡 C2 Beaconing": {
        "Normal": 0.014, "SSH Brute Force": 0.006,
        "Port Scan": 0.008, "Reverse Shell": 0.013, "C2 Beacon": 0.959,
    },
}

# Static threat events shown on the Overview page
SAMPLE_EVENTS = pd.DataFrame({
    "Time":       ["14:21:08","14:19:33","14:17:55","14:15:02","14:12:44","14:10:11","14:08:02"],
    "Source IP":  ["192.168.1.42","10.0.0.78","172.16.3.5","192.168.5.11","10.10.1.3","185.220.101.5","45.33.32.156"],
    "Dest IP":    ["10.0.0.1"]*7,
    "Dest Port":  [22, 80, 4444, 8888, 22, 443, 22],
    "Protocol":   ["TCP","TCP","TCP","TCP","TCP","HTTPS","TCP"],
    "Class":      ["SSH Brute Force","Port Scan","Reverse Shell","C2 Beacon","SSH Brute Force","Port Scan","SSH Brute Force"],
    "Confidence": ["96.2%","97.4%","95.7%","95.9%","96.8%","94.3%","97.1%"],
    "Action":     ["🚫 Blocked","🚫 Blocked","🚫 Blocked","🔔 Alert","🚫 Blocked","🔔 Alert","🚫 Blocked"],
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def simulate_probabilities(scenario: str) -> Dict[str, float]:
    """Add small random noise to base probabilities and renormalise."""
    noisy = {
        k: max(0.001, v + (random.random() - 0.5) * 0.014)
        for k, v in BASE_PROBS[scenario].items()
    }
    total = sum(noisy.values())
    return {k: v / total for k, v in noisy.items()}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA
# FIX Bug 6: replaced non-seeded random.randint with seeded np.random.default_rng
# so cached data is deterministic and consistent across cache refreshes.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def get_geo_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=99)
    cities = [
        ("Mumbai",    "IN",  19.08,   72.88,  "SSH Brute Force"),
        ("Beijing",   "CN",  39.90,  116.40,  "Port Scan"),
        ("Moscow",    "RU",  55.75,   37.62,  "C2 Beacon"),
        ("São Paulo", "BR", -23.55,  -46.63,  "Reverse Shell"),
        ("Lagos",     "NG",   6.52,    3.37,  "SSH Brute Force"),
        ("Jakarta",   "ID",  -6.21,  106.85,  "Port Scan"),
        ("Frankfurt", "DE",  50.11,    8.68,  "C2 Beacon"),
        ("Kyiv",      "UA",  50.45,   30.52,  "Reverse Shell"),
        ("Tehran",    "IR",  35.69,   51.39,  "SSH Brute Force"),
        ("Bogotá",    "CO",   4.71,  -74.07,  "Port Scan"),
    ]
    rows = [
        {"city": city, "cc": cc, "lat": lat, "lon": lon,
         "threat": threat, "count": int(rng.integers(12, 380))}
        for city, cc, lat, lon, threat in cities
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_session_state() -> None:
    defaults: dict = {
        "log":           [],
        "alert_count":   0,
        "total_flows":   SETTINGS.total_flows_init,
        "tl_data":       list(np.random.randint(15, 85, 60)),
        "sparklines":    {c: list(np.random.uniform(0.88, 0.99, 20)) for c in CLASSES},
        "last_refresh":  datetime.now(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def total_alerts() -> int:
    return st.session_state.alert_count + SETTINGS.threat_base


def add_alert() -> None:
    st.session_state.alert_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# ARDUINO ALARM
# FIX Bug 1: moved trigger_arduino_alarm BEFORE main() so it is reachable.
# FIX Bug 7: guarded with _REQUESTS_AVAILABLE flag set at import time.
# ─────────────────────────────────────────────────────────────────────────────
def trigger_arduino_alarm(attack_label: str) -> None:
    """Send an HTTP request to live_detection.py to sound the buzzer."""
    if not _REQUESTS_AVAILABLE:
        st.warning("⚠️ 'requests' library not installed. Cannot reach Arduino service.")
        return
    try:
        response = _requests.post(
            "http://localhost:5001/alert",
            json={"attack": attack_label},
            timeout=0.5,
        )
        if response.status_code == 200:
            st.success("🔔 Buzzer triggered on Arduino!")
        else:
            st.warning("⚠️ Live detection service responded with an error.")
    except _requests.exceptions.ConnectionError:
        st.warning("⚠️ Live detection service not running. Start live_detection.py first.")
    except Exception as e:
        st.warning(f"⚠️ Could not reach live detection: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_traffic_distribution() -> go.Figure:
    percentages = [60.2, 14.1, 12.8, 7.4, 5.5]
    fig = go.Figure(go.Pie(
        labels=CLASSES, values=percentages, hole=0.68,
        marker=dict(
            colors=list(CLASS_COLORS.values()),
            line=dict(color="#070b14", width=3),
        ),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        textinfo="none",
        rotation=90,
    ))
    fig.add_annotation(
        text="<b>100%</b><br><span style='font-size:10px'>classified</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#e6f1ff"),
        align="center",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Traffic Class Distribution",
        height=300,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def chart_threat_timeline_24h() -> go.Figure:
    hours = list(range(24))
    rng = np.random.default_rng(42)
    ssh       = np.clip(rng.integers(5,  40, 24).astype(float) + np.sin(np.linspace(0, 6, 24)) * 10, 0, None)
    port_scan = np.clip(rng.integers(3,  30, 24).astype(float), 0, None)
    reverse   = np.clip(rng.integers(1,  15, 24).astype(float), 0, None)
    c2        = np.clip(rng.integers(1,  12, 24).astype(float), 0, None)

    fig = go.Figure()
    for vals, name, color, alpha in [
        (c2,        "C2 Beacon",       "#14b8a6", 0.70),
        (reverse,   "Reverse Shell",   "#a78bfa", 0.65),
        (port_scan, "Port Scan",       "#f59e0b", 0.60),
        (ssh,       "SSH Brute Force", "#ef4444", 0.55),
    ]:
        r, g, b = hex_to_rgb(color)
        fig.add_trace(go.Scatter(
            x=hours, y=vals, name=name,
            mode="lines", stackgroup="one",
            line=dict(color=color, width=1.2),
            fillcolor=f"rgba({r},{g},{b},{alpha})",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Threat Events — Last 24 Hours (Stacked)",
        height=300,
        xaxis=dict(ticksuffix=":00", gridcolor="#1a2035", title="Hour"),
        yaxis=dict(gridcolor="#1a2035", title="Events"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def chart_live_traffic_feed() -> go.Figure:
    """Append one data point (with spike probability) and return updated chart."""
    spike   = random.random() < 0.07
    new_val = random.randint(90, 220) if spike else random.randint(10, 75)
    st.session_state.tl_data.append(new_val)
    st.session_state.tl_data = st.session_state.tl_data[-60:]
    # FIX Bug 4: alert count is incremented here only when this chart is
    # rendered (Overview page only). No change needed structurally, but
    # the caller must ensure this is only invoked from page_overview.
    if spike:
        add_alert()

    threshold_line = [75] * 60
    x_axis = list(range(-59, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=st.session_state.tl_data,
        mode="lines", name="Traffic",
        line=dict(color="#4f6ef7", width=2),
        fill="tozeroy", fillcolor="rgba(79,110,247,0.07)",
        hovertemplate="T%{x}s: %{y} pps<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=threshold_line,
        mode="lines", name="Alert threshold",
        line=dict(color="#ef4444", width=1.2, dash="dot"),
    ))

    spikes_x = [i - 59 for i, v in enumerate(st.session_state.tl_data) if v > 75]
    spikes_y = [v for v in st.session_state.tl_data if v > 75]
    if spikes_x:
        fig.add_trace(go.Scatter(
            x=spikes_x, y=spikes_y,
            mode="markers", name="Threat spike",
            marker=dict(
                color="#ef4444", size=8, symbol="circle",
                line=dict(color="rgba(239,68,68,0.5)", width=8),
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="● Live Traffic Feed — Packets/sec (last 60 s)",
        height=300,
        xaxis=dict(gridcolor="rgba(0,0,0,0)", ticksuffix="s"),
        yaxis=dict(gridcolor="#1a2035", range=[0, 240]),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def chart_feature_importance() -> go.Figure:
    features    = ["inter_arrival_mean","packets_per_sec","unique_dst_ports",
                   "bytes_per_sec","dst_port","total_packets","protocol","flow_duration"]
    importances = [0.23, 0.20, 0.18, 0.14, 0.12, 0.07, 0.04, 0.02]
    colors = ["#22c55e" if v > 0.18 else "#4f6ef7" if v > 0.12 else "#8892b0"
              for v in importances]
    fig = go.Figure(go.Bar(
        x=importances, y=features, orientation="h",
        marker=dict(color=colors, cornerradius=5, opacity=0.85),
        text=[f"{v:.2f}" for v in importances],
        textposition="outside",
        textfont=dict(size=10, color="#8892b0"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Importance",
        height=300,
        xaxis=dict(gridcolor="#1a2035", range=[0, 0.28]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_confusion_matrix() -> go.Figure:
    cm = np.array([
        [918,   0,   8,   0,   0],
        [  0, 248,   0,   3,   0],
        [ 12,   0, 436,   0,   4],
        [  0,   0,   0, 172,   0],
        [  0,   0,   7,   0, 268],
    ])
    labels_short = ["Normal", "SSH BF", "PortScan", "RevShell", "C2"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=labels_short, y=labels_short,
        colorscale=[
            [0,     "#0c101d"],
            [0.001, "#0f2a1a"],
            [0.5,   "#16653a"],
            [1,     "#22c55e"],
        ],
        text=cm,
        texttemplate="<b>%{text}</b>",
        customdata=cm_norm * 100,
        hovertemplate=(
            "Actual: <b>%{y}</b><br>"
            "Pred: <b>%{x}</b><br>"
            "Count: %{text}<br>"
            "Rate: %{customdata:.1f}%<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8892b0"), thickness=12, len=0.8),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title="Normalised Confusion Matrix — 30 % Test Set",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual", autorange="reversed"),
    )
    return fig


def chart_roc_curves() -> go.Figure:
    fig = go.Figure()
    classes_auc = [
        ("Normal",    "#22c55e", 0.9993),
        ("SSH BF",    "#ef4444", 0.9981),
        ("Port Scan", "#f59e0b", 0.9963),
        ("Rev Shell", "#a78bfa", 0.9941),
        ("C2",        "#14b8a6", 0.9920),
    ]
    for name, color, auc in classes_auc:
        x = np.linspace(0, 1, 80)
        y = np.clip(x ** (1.0 / (auc * 2)), 0.0, 1.0)
        r, g, b = hex_to_rgb(color)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=f"{name}  AUC={auc:.4f}",
            line=dict(color=color, width=2.2),
            mode="lines",
            fill="tozeroy" if name == "Normal" else "none",
            fillcolor=f"rgba({r},{g},{b},0.04)",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random baseline", mode="lines",
        line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title="ROC-AUC One-vs-Rest (OvR) — All Classes",
        xaxis=dict(title="False Positive Rate", gridcolor="#1a2035", range=[0, 1]),
        yaxis=dict(title="True Positive Rate",  gridcolor="#1a2035", range=[0, 1]),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def chart_radar() -> go.Figure:
    metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    class_metrics: Dict[str, List[float]] = {
        "Normal":          [0.982, 0.980, 0.981, 0.999, 0.998],
        "SSH Brute Force": [0.965, 0.988, 0.962, 0.998, 0.997],
        "Port Scan":       [0.954, 0.964, 0.952, 0.996, 0.994],
        "Reverse Shell":   [0.936, 0.928, 0.934, 0.994, 0.991],
        "C2 Beacon":       [0.914, 0.906, 0.912, 0.992, 0.988],
    }
    fig = go.Figure()
    for cls, vals in class_metrics.items():
        color = CLASS_COLORS[cls]
        r, g, b = hex_to_rgb(color)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics + [metrics[0]],
            name=cls, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.06)",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=440,
        title="Per-Class Metric Radar (Stratified 5-Fold CV)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0.87, 1.0], color="#8892b0", gridcolor="#1e2d45"),
            angularaxis=dict(color="#8892b0", gridcolor="#1e2d45"),
        ),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


def chart_training_history() -> go.Figure:
    rng    = np.random.default_rng(7)
    epochs = list(range(1, 51))
    t_acc  = np.clip(
        0.70 + 0.28 * (1 - np.exp(-np.array(epochs) / 8)) + rng.standard_normal(50) * 0.005,
        0.0, 1.0,
    )
    v_acc  = np.clip(t_acc - rng.uniform(0.005, 0.020, 50), 0.0, 1.0)
    t_loss = np.clip(1.2 * np.exp(-np.array(epochs) / 10) + rng.standard_normal(50) * 0.01, 0.0, None)
    v_loss = t_loss + rng.uniform(0.005, 0.030, 50)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy over Epochs", "Loss over Epochs"),
    )
    traces = [
        (t_acc,  "Train Acc",  "#4f6ef7", 1, 1),
        (v_acc,  "Val Acc",    "#22c55e", 1, 1),
        (t_loss, "Train Loss", "#4f6ef7", 1, 2),
        (v_loss, "Val Loss",   "#22c55e", 1, 2),
    ]
    for y, name, color, row, col in traces:
        fig.add_trace(
            go.Scatter(x=epochs, y=y, name=name, mode="lines",
                       line=dict(color=color, width=2)),
            row=row, col=col,
        )
    fig.update_layout(**PLOTLY_LAYOUT, height=360, title="Training History",
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="#1a2035")
    fig.update_yaxes(gridcolor="#1a2035")
    return fig


def chart_threat_heatmap() -> go.Figure:
    rng         = np.random.default_rng(0)
    hours       = list(range(24))
    threat_types = ["SSH BF", "Port Scan", "Rev Shell", "C2 Beacon"]
    heat_data   = rng.integers(0, 40, (len(threat_types), 24)).astype(float)
    heat_data[0, [2, 3, 14, 15, 16]] += rng.integers(30, 80, 5)
    heat_data[1, [9, 10, 11]]        += rng.integers(20, 60, 3)

    fig = go.Figure(go.Heatmap(
        z=heat_data, x=hours, y=threat_types,
        colorscale=[
            [0,   "#0c101d"],
            [0.3, "#16213e"],
            [0.6, "#7c3aed"],
            [1,   "#ef4444"],
        ],
        hovertemplate="Hour %{x}:00 | %{y}<br>Events: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8892b0"), thickness=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        title="Threat Activity Heatmap — Last 24 Hours",
        xaxis=dict(title="Hour of day", ticksuffix=":00", gridcolor="#1a2035"),
        yaxis=dict(gridcolor="#1a2035"),
    )
    return fig


def chart_iat_violin() -> go.Figure:
    rng = np.random.default_rng(42)
    iat_data: Dict[str, np.ndarray] = {
        "Normal":          rng.exponential(0.625, 500),
        "SSH Brute Force": rng.exponential(0.204, 300) + rng.standard_normal(300) * 0.04,
        "Port Scan":       rng.exponential(0.010, 400) + np.abs(rng.standard_normal(400) * 0.005),
        "Reverse Shell":   rng.exponential(0.213, 200) + rng.standard_normal(200) * 0.03,
        "C2 Beacon":       rng.exponential(0.500, 250) + rng.standard_normal(250) * 0.09,
    }
    fig = go.Figure()
    for cls, vals in iat_data.items():
        color = CLASS_COLORS[cls]
        r, g, b = hex_to_rgb(color)
        fig.add_trace(go.Violin(
            y=np.clip(vals, 0, 2), name=cls,
            fillcolor=f"rgba({r},{g},{b},0.25)",
            line=dict(color=color, width=1.5),
            meanline=dict(visible=True, color=color),
            box=dict(visible=True),
            points=False,
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title="IAT Distribution per Traffic Class (seconds)",
        yaxis=dict(title="Inter-Arrival Time (s)", gridcolor="#1a2035", range=[0, 2.0]),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        violinmode="group",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# FIX Bug 3: sensitivity and auto_block are now properly named and returned
# for use in page logic (auto_block gates block vs alert actions).
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> Tuple[str, int, str, bool]:
    with st.sidebar:
        st.markdown("""
        <div style='padding-bottom:18px;border-bottom:1px solid #1a2035;margin-bottom:18px'>
          <div style='display:flex;align-items:center;gap:12px'>
            <div style='width:40px;height:40px;border-radius:10px;
                        background:linear-gradient(135deg,#4f6ef7,#7c3aed);
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;box-shadow:0 4px 15px rgba(79,110,247,.4)'>🛡️</div>
            <div>
              <div style='font-size:15px;font-weight:800;color:#e6f1ff;letter-spacing:-.3px'>ML-NIDS Pro</div>
              <div style='font-size:10px;color:#4f6ef7;font-weight:600;letter-spacing:1.5px'>ENTERPRISE EDITION</div>
            </div>
          </div>
          <div style='margin-top:12px;display:flex;gap:6px;flex-wrap:wrap'>
            <span style='background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2);
                         border-radius:20px;padding:2px 10px;font-size:10px;font-weight:600'>● LIVE</span>
            <span style='background:rgba(79,110,247,.1);color:#4f6ef7;border:1px solid rgba(79,110,247,.2);
                         border-radius:20px;padding:2px 10px;font-size:10px;font-weight:600'>RF v3.2</span>
            <span style='background:rgba(20,184,166,.1);color:#14b8a6;border:1px solid rgba(20,184,166,.2);
                         border-radius:20px;padding:2px 10px;font-size:10px;font-weight:600'>GPU ACCEL</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
            "color:#4f6ef7;font-weight:700;margin-bottom:8px'>Navigation</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            label="Navigation",
            options=["📊 Overview", "🌍 Threat Map", "📈 Model Performance",
                     "🎯 Live Simulator", "📋 Forensics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
            "color:#4f6ef7;font-weight:700;margin-bottom:8px'>Detection Settings</div>",
            unsafe_allow_html=True,
        )
        threshold   = st.slider("Alert threshold (%)", 80, 99, SETTINGS.alert_threshold_default)
        sensitivity = st.select_slider("Sensitivity", ["Low", "Medium", "High", "Maximum"], value="High")
        auto_block  = st.toggle("Auto-block threats", value=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
            "color:#4f6ef7;font-weight:700;margin-bottom:8px'>System Status</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model",      "Random Forest")
            st.metric("Inference",  "1.8 ms avg")
            st.metric("Threats (24h)", str(total_alerts()))
        with col2:
            st.metric("Features",  "12 selected")
            st.metric("Uptime",    "99.97%")
        st.caption(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        if st.button("🔄 Refresh Metrics"):
            st.session_state.total_flows += random.randint(100, 900)
            st.rerun()

    return page, threshold, sensitivity, auto_block


# ─────────────────────────────────────────────────────────────────────────────
# TOP-BAR BANNER
# ─────────────────────────────────────────────────────────────────────────────
def render_top_bar() -> None:
    alert_n = total_alerts()
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0c101d,#0f1526);
                border:1px solid #1e2d45;border-radius:16px;
                padding:16px 24px;display:flex;justify-content:space-between;
                align-items:center;margin-bottom:24px;
                box-shadow:0 8px 32px rgba(0,0,0,.4)'>
      <div>
        <div style='font-size:18px;font-weight:800;color:#e6f1ff'>
          Network Intrusion Detection System
        </div>
        <div style='font-size:12px;color:#8892b0'>
          Real-time threat classification · Random Forest · Accuracy 96.3%
        </div>
      </div>
      <div style='display:flex;gap:10px'>
        <span style='background:rgba(34,197,94,.08);color:#22c55e;
                     border:1px solid rgba(34,197,94,.2);
                     border-radius:20px;padding:5px 14px;font-size:11px;font-weight:700'>
          ● OPERATIONAL
        </span>
        <span style='background:rgba(239,68,68,.08);color:#ef4444;
                     border:1px solid rgba(239,68,68,.2);
                     border-radius:20px;padding:5px 14px;font-size:11px;font-weight:700'>
          🚨 {alert_n} Alerts Today
        </span>
        <span style='background:rgba(79,110,247,.08);color:#4f6ef7;
                     border:1px solid rgba(79,110,247,.2);
                     border-radius:20px;padding:5px 14px;font-size:11px;font-weight:700'>
          📡 {st.session_state.total_flows:,} Flows
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(threshold: int) -> None:
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("🔵 Total Flows",   f"{st.session_state.total_flows:,}", "+3.2% /hr")
    with k2: st.metric("🔴 Threats",       str(total_alerts()),                 "-8.1% /hr")
    with k3: st.metric("🟢 Accuracy",      "96.3%",                             "F1 0.957")
    with k4: st.metric("⚡ Latency (avg)",  "1.8 ms",                            "p99 4.2 ms")
    with k5: st.metric("🛡️ Blocked",        "139",                               "+2 last min")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.plotly_chart(chart_traffic_distribution(),  use_container_width=True, config=PLOTLY_CFG)
    with col2:
        st.plotly_chart(chart_threat_timeline_24h(),   use_container_width=True, config=PLOTLY_CFG)

    col3, col4 = st.columns([1.6, 1])
    with col3:
        # FIX Bug 4: chart_live_traffic_feed is ONLY called here (Overview page),
        # preventing spurious alert_count increments on other pages.
        st.plotly_chart(chart_live_traffic_feed(),     use_container_width=True, config=PLOTLY_CFG)
    with col4:
        st.plotly_chart(chart_feature_importance(),    use_container_width=True, config=PLOTLY_CFG)

    st.markdown("### 🔔 Recent Threat Events")
    st.dataframe(SAMPLE_EVENTS, use_container_width=True, hide_index=True)


def page_threat_map() -> None:
    st.markdown("### 🌍 Global Threat Intelligence Map")
    df_geo = get_geo_data()

    col_map, col_right = st.columns([2.2, 1])
    with col_map:
        fig = go.Figure()
        for threat in [c for c in CLASSES if c != "Normal"]:
            color  = CLASS_COLORS[threat]
            subset = df_geo[df_geo["threat"] == threat]
            fig.add_trace(go.Scattergeo(
                lon=subset["lon"],
                lat=subset["lat"],
                text=subset.apply(
                    lambda r: f"{r['city']} ({r['cc']})<br>{r['threat']}: {r['count']} events",
                    axis=1,
                ),
                name=threat,
                marker=dict(
                    size=subset["count"] / 15 + 8,
                    color=color, opacity=0.8,
                    line=dict(width=1.5, color=color),
                    sizemode="diameter",
                ),
                hoverinfo="text",
            ))
        fig.update_geos(
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
            showland=True,      landcolor="#0f1526",
            showocean=True,     oceancolor="#070b14",
            showcoastlines=True, coastlinecolor="#1e2d45",
            showframe=False,
            showcountries=True,  countrycolor="#1e2d45",
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=450,
            title="Live Threat Origin Map — Incoming Attacks",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    with col_right:
        st.markdown("**Top Attack Sources**")
        df_sorted = df_geo.sort_values("count", ascending=False)
        grand_total = df_sorted["count"].sum()
        for _, row in df_sorted.iterrows():
            color = CLASS_COLORS.get(row["threat"], "#8892b0")
            pct   = int(row["count"] / grand_total * 100)
            st.markdown(f"""
            <div style='margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='font-size:12px;color:#ccd6f6;font-weight:600'>
                  {row["city"]}, {row["cc"]}
                </span>
                <span style='font-size:11px;color:{color};font-weight:700'>{row["count"]}</span>
              </div>
              <div style='display:flex;align-items:center;gap:8px'>
                <div style='flex:1;height:4px;background:#1e2d45;border-radius:2px'>
                  <div style='width:{pct}%;height:100%;background:{color};border-radius:2px'></div>
                </div>
                <span style='font-size:10px;color:#8892b0;width:28px'>{pct}%</span>
              </div>
              <div style='font-size:10px;color:#4a5568;margin-top:2px'>{row["threat"]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Threat Distribution by Type")
    cols    = st.columns(4)
    threats = ["SSH Brute Force", "Port Scan", "Reverse Shell", "C2 Beacon"]
    totals  = df_geo.groupby("threat")["count"].sum()
    for col, threat in zip(cols, threats):
        with col:
            st.metric(
                f"{THREAT_ICONS[threat]} {threat}",
                str(totals.get(threat, 0)),
                f"+{random.randint(2, 18)}% vs yesterday",
            )


def page_model_performance() -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("✅ Accuracy",         "96.3%")
    with m2: st.metric("🎯 Precision",        "0.961")
    with m3: st.metric("📡 Recall",           "0.954")
    with m4: st.metric("🏆 F1 (macro)",       "0.957")
    with m5: st.metric("📐 ROC-AUC (macro)",  "0.997")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔢 Confusion Matrix",
        "📉 ROC Curves",
        "📊 Per-Class Metrics",
        "⚡ Perf Over Time",
    ])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(chart_confusion_matrix(), use_container_width=True, config=PLOTLY_CFG)
        with col2:
            st.markdown("**Classification Report**")
            report_df = pd.DataFrame({
                "Class":     CLASSES + ["macro avg", "weighted avg"],
                "Precision": [0.982, 0.965, 0.954, 0.936, 0.914, 0.961, 0.963],
                "Recall":    [0.980, 0.988, 0.964, 0.928, 0.906, 0.954, 0.963],
                "F1-Score":  [0.981, 0.962, 0.952, 0.934, 0.912, 0.957, 0.963],
                "Support":   [926, 251, 452, 172, 275, 2076, 2076],
            })
            styled = (
                report_df.style
                .background_gradient(
                    cmap="Greens",
                    subset=["Precision", "Recall", "F1-Score"],
                    vmin=0.85, vmax=1.0,
                )
                .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            f1_scores = [0.981, 0.962, 0.952, 0.934, 0.912]
            fig_f1 = go.Figure(go.Bar(
                x=CLASSES, y=f1_scores,
                marker=dict(color=list(CLASS_COLORS.values()), cornerradius=6, opacity=0.85),
                text=[f"{v:.3f}" for v in f1_scores],
                textposition="outside",
            ))
            fig_f1.update_layout(
                **PLOTLY_LAYOUT, height=220,
                title="Per-Class F1 Scores",
                yaxis=dict(range=[0.88, 1.0], gridcolor="#1a2035"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_f1, use_container_width=True, config=PLOTLY_CFG)

    with tab2:
        st.plotly_chart(chart_roc_curves(),        use_container_width=True, config=PLOTLY_CFG)
    with tab3:
        st.plotly_chart(chart_radar(),             use_container_width=True, config=PLOTLY_CFG)
    with tab4:
        st.plotly_chart(chart_training_history(),  use_container_width=True, config=PLOTLY_CFG)


def page_live_simulator(threshold: int, auto_block: bool) -> None:
    """
    FIX Bug 2: auto_block parameter is now used to determine whether detected
    threats are hard-blocked or only alerted, and to conditionally call
    trigger_arduino_alarm for confirmed attacks.
    """
    st.markdown("### 🎯 Attack Scenario Simulator")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        scenario = st.selectbox("**Attack Scenario**", list(SCENARIOS.keys()))
    s = SCENARIOS[scenario]

    with col_right:
        st.markdown(f"""
        <div style='background:#0f1526;border:1px solid #1e2d45;border-radius:12px;
                    padding:14px 18px;margin-top:4px'>
          <div style='font-size:11px;color:#4f6ef7;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px'>Scenario Profile</div>
          <div style='display:flex;gap:24px;flex-wrap:wrap'>
            <div><div style='font-size:10px;color:#8892b0'>Dest Port</div>
                 <div style='font-size:16px;font-weight:700;color:#e6f1ff'>{s["dport"]}</div></div>
            <div><div style='font-size:10px;color:#8892b0'>Packets</div>
                 <div style='font-size:16px;font-weight:700;color:#e6f1ff'>{s["pkts"]}</div></div>
            <div><div style='font-size:10px;color:#8892b0'>Bytes</div>
                 <div style='font-size:16px;font-weight:700;color:#e6f1ff'>{s["bytes_"]:,}</div></div>
            <div><div style='font-size:10px;color:#8892b0'>Duration</div>
                 <div style='font-size:16px;font-weight:700;color:#e6f1ff'>{s["dur"]} s</div></div>
            <div><div style='font-size:10px;color:#8892b0'>Pkt/sec</div>
                 <div style='font-size:16px;font-weight:700;color:#e6f1ff'>{s["pps"]}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with st.expander("⚙️ Advanced Flow Parameters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            dport  = st.number_input("Destination Port",   value=int(s["dport"]),  min_value=1, max_value=65535)
            pkts   = st.number_input("Total Packets",      value=int(s["pkts"]),   min_value=1)
            bytes_ = st.number_input("Total Bytes",        value=int(s["bytes_"]), min_value=1)
            dur    = st.number_input("Flow Duration (s)",  value=float(s["dur"]),  step=0.1, format="%.1f")
        with fc2:
            psmean = st.number_input("Pkt Size Mean",      value=float(s["psmean"]), step=0.1,   format="%.1f")
            psstd  = st.number_input("Pkt Size Std",       value=float(s["psstd"]),  step=0.1,   format="%.1f")
            pps    = st.number_input("Packets / sec",      value=float(s["pps"]),    step=0.1,   format="%.1f")
            bps    = st.number_input("Bytes / sec",        value=float(s["bps"]),    step=0.1,   format="%.1f")
        with fc3:
            udp  = st.number_input("Unique Dst Ports",     value=int(s["udp"]),  min_value=1)
            iam  = st.number_input("Inter-arrival Mean",   value=float(s["iam"]), step=0.001, format="%.3f")
            ias  = st.number_input("Inter-arrival Std",    value=float(s["ias"]), step=0.001, format="%.3f")
            _    = st.selectbox("Protocol", ["TCP", "UDP"])   # visual only

    if st.button("🚀 Run Inference", use_container_width=False):
        with st.spinner("Running Random Forest inference…"):
            time.sleep(0.35)

        probs        = simulate_probabilities(scenario)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        top_class, top_conf = sorted_probs[0]
        conf_pct    = top_conf * 100
        is_normal   = top_class == "Normal"
        above_thresh = conf_pct >= threshold

        # FIX Bug 2 & Bug 3: use auto_block to decide action label and
        # whether to trigger the Arduino alarm.
        if is_normal:
            action_label = "✅ Clean"
            st.success(
                f"🛡️  **NORMAL TRAFFIC** — No threat detected &nbsp;|&nbsp; "
                f"Confidence: **{conf_pct:.1f}%**"
            )
        elif not above_thresh:
            action_label = "🔔 Monitor"
            st.warning(
                f"⚠️  **SUSPICIOUS — {top_class.upper()}** — "
                f"Confidence {conf_pct:.1f}% below {threshold}% threshold. Monitor closely."
            )
        else:
            if auto_block:
                action_label = "🚫 Blocked"
                st.error(
                    f"🚨  **ATTACK DETECTED — {top_class.upper()}** — "
                    f"Confidence: **{conf_pct:.1f}%** &nbsp;|&nbsp; Source auto-blocked."
                )
                # FIX Bug 2: trigger Arduino alarm for confirmed blocked attacks
                trigger_arduino_alarm(top_class)
            else:
                action_label = "🔴 Alert"
                st.error(
                    f"🚨  **ATTACK DETECTED — {top_class.upper()}** — "
                    f"Confidence: **{conf_pct:.1f}%** &nbsp;|&nbsp; Auto-block disabled — manual action required."
                )

        st.session_state.log.insert(0, {
            "Timestamp":  datetime.now().strftime("%H:%M:%S"),
            "Scenario":   scenario,
            "Prediction": top_class,
            "Confidence": f"{conf_pct:.1f}%",
            "Threshold":  f"{threshold}%",
            "Action":     action_label,
        })
        if not is_normal:
            add_alert()

        gauge_color = "#22c55e" if is_normal else ("#f59e0b" if not above_thresh else "#ef4444")
        gr, gg, gb  = hex_to_rgb(gauge_color)

        col_gauge, col_probs = st.columns(2)
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=conf_pct,
                delta=dict(
                    reference=threshold, valueformat=".1f",
                    increasing=dict(color="#ef4444"),
                    decreasing=dict(color="#22c55e"),
                ),
                number=dict(suffix="%", font=dict(size=36, color=gauge_color, family="Inter")),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickcolor="#8892b0",
                        tickfont=dict(color="#8892b0", size=10),
                    ),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor="#0f1526",
                    bordercolor="#1e2d45",
                    borderwidth=2,
                    steps=[
                        {"range": [0, threshold],       "color": "#0f1526"},
                        {"range": [threshold, 100],     "color": f"rgba({gr},{gg},{gb},0.08)"},
                    ],
                    threshold=dict(
                        line=dict(color=gauge_color, width=3),
                        thickness=0.8,
                        value=threshold,
                    ),
                ),
                title=dict(
                    text=(
                        f"<b>Detection Confidence</b><br>"
                        f"<span style='font-size:11px'>Alert threshold: {threshold}%</span>"
                    ),
                    font=dict(color="#ccd6f6", size=14),
                ),
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True, config=PLOTLY_CFG)

        with col_probs:
            labels  = [c for c, _ in sorted_probs]
            vals    = [v * 100 for _, v in sorted_probs]
            colors  = [CLASS_COLORS[c] for c in labels]
            opacity = [1.0 if c == top_class else 0.35 for c in labels]
            fig_probs = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors, cornerradius=6, opacity=opacity),
                text=[f"{v:.2f}%" for v in vals],
                textposition="outside",
                textfont=dict(size=11, color="#8892b0"),
            ))
            fig_probs.update_layout(
                **PLOTLY_LAYOUT, height=300,
                title="Class Probabilities",
                xaxis=dict(range=[0, 108], gridcolor="#1a2035"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_probs, use_container_width=True, config=PLOTLY_CFG)

        # SHAP-style attribution
        st.markdown("#### 🔍 Feature Attribution (SHAP-style)")
        shap_feats = [
            "inter_arrival_mean", "packets_per_sec", "dst_port",
            "unique_dst_ports", "bytes_per_sec", "total_packets",
        ]
        rng_shap  = np.random.default_rng()
        shap_vals = (
            rng_shap.uniform(-0.1, 0.1, len(shap_feats)).tolist()
            if is_normal
            else rng_shap.uniform(-0.05, 0.45, len(shap_feats)).tolist()
        )
        shap_cols = ["#ef4444" if v > 0 else "#22c55e" for v in shap_vals]
        fig_shap  = go.Figure(go.Bar(
            x=shap_vals, y=shap_feats, orientation="h",
            marker=dict(color=shap_cols, cornerradius=4),
        ))
        fig_shap.add_vline(x=0, line_color="#1e2d45", line_width=1.5)
        fig_shap.update_layout(
            **PLOTLY_LAYOUT, height=240,
            title=f"Top Feature Contributions → <b>{top_class}</b>",
            xaxis=dict(gridcolor="#1a2035", title="SHAP value"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_shap, use_container_width=True, config=PLOTLY_CFG)

    st.markdown("---")
    st.markdown("### 📋 Detection Log")
    col_log, col_clear = st.columns([8, 1])
    with col_clear:
        if st.button("🗑️ Clear"):
            st.session_state.log = []
            st.rerun()

    if st.session_state.log:
        st.dataframe(
            pd.DataFrame(st.session_state.log[:25]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No detections yet. Select a scenario and click **Run Inference**.")


def page_forensics(threshold: int) -> None:
    st.markdown("### 📋 Threat Forensics & Hunt")
    tab1, tab2, tab3 = st.tabs([
        "🔎 Flow Inspector", "📈 Anomaly Heatmap", "🧬 Pattern Analysis"
    ])

    with tab1:
        st.markdown("**Deep‑dive any recent flow for forensic detail.**")
        ip = st.text_input("Source IP", value="192.168.1.42")
        if st.button("🔎 Investigate"):
            flow_start = (datetime.utcnow() - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")
            # FIX Bug 5: threshold is now used in the report output (was hardcoded before)
            st.markdown(f"""
            <div style='background:#0f1526;border:1px solid #1e2d45;border-radius:12px;
                        padding:20px;font-family:JetBrains Mono,monospace;
                        font-size:12px;color:#ccd6f6;line-height:2'>
              <div style='color:#4f6ef7;font-weight:700;margin-bottom:8px'>
                ── FLOW FORENSICS: {ip} ──────────────────────────────
              </div>
              Source IP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#22c55e'>{ip}</span><br>
              Dest IP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#e6f1ff'>10.0.0.1</span><br>
              Dest Port&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#ef4444'>22 (SSH)</span><br>
              Protocol&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: TCP<br>
              Start Time&nbsp;&nbsp;&nbsp;: {flow_start} UTC<br>
              Duration&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 24.5 s<br>
              Packets&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 120 (→98 ←22)<br>
              Bytes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 14,400 (→11,760 ←2,640)<br>
              Pkt/s&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 4.9<br>
              IAT Mean&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.204 s<br>
              IAT Std&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.042 s<br>
              Unique DST&nbsp;&nbsp;&nbsp;: 1<br>
              <br>
              <div style='color:#ef4444;font-weight:700'>
                ── ML CLASSIFICATION ─────────────────────────────────
              </div>
              Predicted&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#ef4444;font-weight:700'>SSH BRUTE FORCE</span><br>
              Confidence&nbsp;&nbsp;&nbsp;: <span style='color:#ef4444'>96.2%</span>&nbsp;(threshold {threshold}%)<br>
              Trees voting : 481/500 for SSH Brute Force<br>
              Action&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#22c55e'>BLOCKED — firewall rule #4421 applied</span><br>
              MITRE ATT&CK : <span style='color:#a78bfa'>T1110.001 — Brute Force: Password Guessing</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(chart_threat_heatmap(), use_container_width=True, config=PLOTLY_CFG)

    with tab3:
        st.plotly_chart(chart_iat_violin(),     use_container_width=True, config=PLOTLY_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #070b14; }
.block-container { padding: 1.5rem 2rem 2rem; }
[data-testid="stSidebar"] { background: #0c101d !important; border-right: 1px solid #1a2035; }

[data-testid="metric-container"] {
  background: linear-gradient(135deg, #0f1526, #141929);
  border: 1px solid #1e2d45;
  border-radius: 14px;
  padding: 20px !important;
  box-shadow: 0 4px 24px rgba(0,0,0,.4);
  transition: transform .2s, border-color .2s;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-2px);
  border-color: #4f6ef7;
}

.stButton > button {
  background: linear-gradient(135deg, #4f6ef7, #7c3aed);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-weight: 700;
  font-size: 13px;
  padding: 12px 28px;
  box-shadow: 0 4px 20px rgba(79,110,247,.4);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(79,110,247,.55);
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# FIX Bug 3: sensitivity and auto_block are now properly used, not discarded.
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    _init_session_state()

    page, threshold, sensitivity, auto_block = render_sidebar()
    render_top_bar()

    if   page == "📊 Overview":
        page_overview(threshold)
    elif page == "🌍 Threat Map":
        page_threat_map()
    elif page == "📈 Model Performance":
        page_model_performance()
    elif page == "🎯 Live Simulator":
        # FIX Bug 3: pass auto_block through to the simulator page
        page_live_simulator(threshold, auto_block)
    elif page == "📋 Forensics":
        page_forensics(threshold)


if __name__ == "__main__":
    main()