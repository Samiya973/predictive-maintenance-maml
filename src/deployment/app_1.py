# """
# src\deployment\app.py
# Predictive Maintenance Demo — MAML few-shot RUL + Early Warning
# """

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import torch, os, sys

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from src.inference.predictor import PredictiveMaintenancePredictor
# from src.data.data_loader import load_preprocessed_data

# # ── Config ────────────────────────────────────────────────────────────────────
# CNN_CKPT   = "results/saved_models/early_pred_best.pth"
# MAML_CKPT  = "results/saved_models/maml_meta_model_best.pth"
# VAE_CKPT   = "results/saved_models/vae_early_warning.pth"
# STATS_PATH = "results/saved_models/baseline_stats.npz"
# DATASET    = "data/processed/FD001_preprocessed.npz"

# STATE_COLORS = {
#     "CRITICAL":      "#EF4444",
#     "EARLY WARNING": "#F59E0B",
#     "MONITOR":       "#3B82F6",
#     "HEALTHY":       "#10B981",
# }

# # ── Helper functions ──────────────────────────────────────────────────────────
# def get_confirmed_detection_cycle(cusum_values, threshold_h, cycles, confirm_cycles=3):
#     streak = 0
#     for c, s in zip(cycles, cusum_values):
#         if s >= threshold_h:
#             streak += 1
#         else:
#             streak = 0

#         if streak >= confirm_cycles:
#             return int(c - confirm_cycles + 1)
#     return None


# def get_display_state(rul_cycles, fault_prob, vae_confirmed):
#     if rul_cycles <= 15 and fault_prob >= 0.90:
#         return "CRITICAL", "Immediate intervention — schedule unplanned maintenance"

#     if rul_cycles <= 30 and (fault_prob >= 0.70 or vae_confirmed):
#         return "EARLY WARNING", "Degradation confirmed — schedule inspection"

#     if vae_confirmed or fault_prob >= 0.50:
#         return "MONITOR", "Emerging anomaly — monitor closely"

#     return "HEALTHY", "No action required"


# @st.cache_resource
# def load_predictor():
#     return PredictiveMaintenancePredictor(CNN_CKPT, MAML_CKPT, VAE_CKPT, STATS_PATH)


# @st.cache_data
# def load_data():
#     return load_preprocessed_data(DATASET)


# # ── Page setup ────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Predictive Maintenance · MAML", layout="wide")
# st.title("🛠 Turbofan Engine — Predictive Maintenance")
# st.caption("MAML few-shot RUL · CNN-LSTM Early Prediction · VAE+CUSUM Early Warning · NASA C-MAPSS FD001")

# predictor = load_predictor()
# data      = load_data()
# max_rul   = float(data["max_rul"])
# test_ids  = np.array(data["test_engine_ids"])
# engine_list = sorted(np.unique(test_ids).tolist())

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.header("Engine Selection")
#     engine_id = st.selectbox("Engine ID", engine_list, index=0)
#     mask      = test_ids == engine_id
#     n_windows = int(mask.sum())
#     max_cycle = st.slider("Up to cycle", min_value=5, max_value=n_windows,
#                            value=n_windows, step=1)
#     st.markdown("---")
#     st.markdown("**Model info**")
#     st.markdown("- CNN-LSTM · F1=0.876 · AUC=0.990")
#     st.markdown("- MAML CNN-LSTM · few-shot adaptation (K=5)")
#     st.markdown("- VAE+CUSUM · FAR=10%")

# # ── Data slicing ──────────────────────────────────────────────────────────────
# X_eng        = data["X_test"][mask][:max_cycle]
# y_eng        = data["y_test"][mask][:max_cycle]
# y_eng_cycles = y_eng * max_rul

# # ── MAML few-shot adaptation ──────────────────────────────────────────────────
# predictor.adapt_to_engine(X_eng, y_eng, K=5)

# # ── Inference loop ─────────────────────────────────────────────────────────────
# results = []
# for i, (window, rul_true_cycles) in enumerate(zip(X_eng, y_eng_cycles)):
#     out = predictor.predict(window, cycle=i + 1)

#     raw_rul = float(out["rul_cycles"])

#     if raw_rul <= 1.5:
#         out["rul_cycles"] = raw_rul * max_rul
#     elif raw_rul > max_rul * 1.5:
#         out["rul_cycles"] = raw_rul / max_rul

#     out["rul_true"] = float(rul_true_cycles)
#     results.append(out)

# df = pd.DataFrame(results)

# # ── Load threshold and compute confirmed detection ────────────────────────────
# stats = np.load(STATS_PATH, allow_pickle=True)
# h = float(stats["threshold_h"])

# confirmed_det_cycle = get_confirmed_detection_cycle(
#     cusum_values=df["vae_cusum"].values,
#     threshold_h=h,
#     cycles=df["cycle"].values,
#     confirm_cycles=3
# )

# if confirmed_det_cycle is not None:
#     df["vae_confirmed"] = df["cycle"] >= confirmed_det_cycle
# else:
#     df["vae_confirmed"] = False

# # ── Debug expander: verify predictions look sane ──────────────────────────────
# with st.expander("Debug — RUL sanity check (collapse when satisfied)", expanded=False):
#     st.caption(f"max_rul = {max_rul:.1f} cycles")
#     st.caption(
#         f"Predicted RUL range: {df['rul_cycles'].min():.1f} – {df['rul_cycles'].max():.1f} cycles"
#     )
#     st.caption(
#         f"True RUL range:      {df['rul_true'].min():.1f} – {df['rul_true'].max():.1f} cycles"
#     )

#     if df["rul_cycles"].max() < 5:
#         st.error(
#             "Predicted RUL is still near 0 after scaling fix. "
#             "The predictor is returning near-zero values. Check adapt_to_engine() "
#             "and predict() implementations."
#         )
#     elif abs(df["rul_cycles"].mean() - df["rul_true"].mean()) > max_rul * 0.6:
#         st.warning(
#             f"Predicted mean ({df['rul_cycles'].mean():.1f}) is far from true mean "
#             f"({df['rul_true'].mean():.1f}). Model may still have a scaling issue."
#         )
#     else:
#         st.success("Predicted RUL range looks reasonable.")

#     if confirmed_det_cycle is not None:
#         st.info(f"Confirmed VAE detection cycle = {confirmed_det_cycle} (3 consecutive breaches)")
#     else:
#         st.info("No confirmed VAE detection in selected window.")

# # ── Current state badge ───────────────────────────────────────────────────────
# latest = results[-1]
# latest_vae_confirmed = bool(df["vae_confirmed"].iloc[-1]) if len(df) > 0 else False

# display_state, display_action = get_display_state(
#     rul_cycles=float(latest["rul_cycles"]),
#     fault_prob=float(latest["fault_prob"]),
#     vae_confirmed=latest_vae_confirmed
# )

# color = STATE_COLORS[display_state]

# st.markdown(
#     f"""
#     <div style="background:{color}22;border:2px solid {color};border-radius:10px;
#                 padding:16px 24px;margin-bottom:16px">
#         <span style="font-size:22px;font-weight:700;color:{color}">● {display_state}</span>
#         &nbsp;&nbsp;
#         <span style="font-size:15px;color:#ccc">{display_action}</span>
#         &nbsp;&nbsp;|&nbsp;&nbsp;
#         <span style="font-size:14px;color:#aaa">
#             RUL ≈ <b>{latest['rul_cycles']:.0f}</b> cycles &nbsp;
#             Fault P = <b>{latest['fault_prob']:.3f}</b> &nbsp;
#             CUSUM = <b>{latest['vae_cusum']:.2f}</b>
#         </span>
#     </div>
#     """, unsafe_allow_html=True
# )

# # ── RUL time-series plot ──────────────────────────────────────────────────────
# ew_cycle = confirmed_det_cycle

# fig_rul = go.Figure()
# fig_rul.add_trace(go.Scatter(
#     x=df["cycle"], y=df["rul_true"],
#     name="RUL true", line=dict(color="#94A3B8", dash="dot"), mode="lines"
# ))
# fig_rul.add_trace(go.Scatter(
#     x=df["cycle"], y=df["rul_cycles"],
#     name="RUL predicted", line=dict(color="#6366F1", width=2), mode="lines"
# ))
# fig_rul.add_hline(
#     y=30, line_dash="dash", line_color="#EF4444",
#     annotation_text="Fault threshold (30 cycles)",
#     annotation_position="top right"
# )

# if ew_cycle is not None:
#     rul_true_at_trigger = float(df[df["cycle"] == ew_cycle]["rul_true"].values[0])
#     lead_time = rul_true_at_trigger - 30

#     fig_rul.add_vline(
#         x=ew_cycle,
#         line_dash="dash",
#         line_color="#F59E0B",
#         annotation_text=(
#             f"Early warning triggered (+{lead_time:.0f} cycle lead)"
#             if lead_time >= 0
#             else f"Early warning triggered ({lead_time:.0f} cycles LATE)"
#         ),
#         annotation_font_color="#F59E0B",
#     )

# fig_rul.update_layout(
#     title=f"Engine {engine_id} — RUL Trajectory",
#     xaxis_title="Cycle", yaxis_title="RUL (cycles)",
#     legend=dict(orientation="h"), height=380,
#     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
#     font=dict(color="#e2e8f0"),
# )
# st.plotly_chart(fig_rul, width='stretch')

# # ── Model comparison panel ────────────────────────────────────────────────────
# st.subheader("Model Comparison Panel")
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("**CNN-LSTM · Early Prediction**")
#     fig_cnn = go.Figure()
#     fig_cnn.add_trace(go.Scatter(
#         x=df["cycle"], y=df["fault_prob"],
#         name="Fault probability", line=dict(color="#6366F1", width=2)
#     ))
#     fig_cnn.add_hline(
#         y=predictor.cnn_threshold, line_dash="dash", line_color="#EF4444",
#         annotation_text=f"Threshold {predictor.cnn_threshold:.2f}"
#     )
#     fig_cnn.update_layout(
#         xaxis_title="Cycle", yaxis_title="P(fault)",
#         height=280, paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0.05)", font=dict(color="#e2e8f0"),
#         margin=dict(t=20, b=40),
#     )
#     st.plotly_chart(fig_cnn, width='stretch')
#     st.metric("F1 Score", "0.8756")
#     st.metric("AUC-ROC",  "0.9899")
#     st.metric("FAR",      "3.65%")

# with col2:
#     st.markdown("**VAE + CUSUM · Early Warning (unsupervised)**")

#     onset_rows = df[df["rul_true"] <= 30]
#     onset_cycle = int(onset_rows["cycle"].iloc[0]) if len(onset_rows) > 0 else None

#     det_cycle = confirmed_det_cycle

#     if det_cycle is not None and onset_cycle is not None:
#         latency = onset_cycle - det_cycle
#     else:
#         latency = None

#     fig_vae = go.Figure()

#     fig_vae.add_trace(go.Scatter(
#         x=df["cycle"], y=df["vae_cusum"],
#         name="CUSUM statistic", line=dict(color="#F59E0B", width=2)
#     ))

#     fig_vae.add_hline(
#         y=h, line_dash="dash", line_color="#EF4444",
#         annotation_text=f"Threshold h={h:.2f}",
#         annotation_font_color="#EF4444",
#         annotation_position="bottom right",
#     )

#     if onset_cycle is not None:
#         fig_vae.add_vline(
#             x=onset_cycle,
#             line_dash="dash",
#             line_color="#EF4444",
#             line_width=1.5,
#             annotation_text="Onset",
#             annotation_font_color="#EF4444",
#             annotation_position="top left",
#             annotation_yshift=0,
#         )

#     if det_cycle is not None:
#         lat_label = f"Det(lat={latency:+d})" if latency is not None else "Detection"
#         fig_vae.add_vline(
#             x=det_cycle,
#             line_dash="solid",
#             line_color="#F59E0B",
#             line_width=2,
#             annotation_text=lat_label,
#             annotation_font_color="#F59E0B",
#             annotation_position="top right",
#             annotation_yshift=20,
#         )

#         if onset_cycle is not None and det_cycle != onset_cycle:
#             y_bracket = df["vae_cusum"].max() * 0.85
#             shade_x0, shade_x1 = sorted([det_cycle, onset_cycle])

#             fig_vae.add_vrect(
#                 x0=shade_x0, x1=shade_x1,
#                 fillcolor="#F59E0B", opacity=0.10,
#                 line_width=0, layer="below",
#             )

#             mid_cycle = (det_cycle + onset_cycle) / 2
#             offset_text = (
#                 f"Δ={abs(latency)} cycles ({'early' if latency > 0 else 'late'})"
#             )

#             fig_vae.add_annotation(
#                 x=mid_cycle, y=y_bracket,
#                 text=offset_text,
#                 showarrow=False,
#                 font=dict(color="#F59E0B", size=11),
#                 bgcolor="rgba(0,0,0,0.45)",
#                 bordercolor="#F59E0B",
#                 borderwidth=1,
#                 borderpad=4,
#             )

#     fig_vae.update_layout(
#         xaxis_title="Cycle", yaxis_title="CUSUM",
#         height=280, paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0.05)", font=dict(color="#e2e8f0"),
#         margin=dict(t=20, b=40),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02),
#     )
#     st.plotly_chart(fig_vae, width='stretch')

#     st.metric("Architecture", "Unsupervised")
#     st.metric("Approach",     "VAE recon error + CUSUM")

#     if latency is not None:
#         if latency > 0:
#             st.success(
#                 f"✅ Early detection: **{latency} cycles** before fault onset "
#                 f"(cycle {det_cycle} → onset at cycle {onset_cycle})"
#             )
#         elif latency < 0:
#             st.warning(
#                 f"⚠️ Late detection: **{abs(latency)} cycles** after fault onset "
#                 f"(onset at cycle {onset_cycle} → detected at cycle {det_cycle})"
#             )
#         else:
#             st.info(f"ℹ️ Detection coincides exactly with fault onset at cycle {onset_cycle}")
#     elif det_cycle is None:
#         st.caption("ℹ️ CUSUM threshold not crossed in selected window.")

#     st.caption(
#         "⚠️ Detection shown here uses confirmation logic (3 consecutive CUSUM breaches), "
#         "so isolated spikes are ignored."
#     )
"""
app/streamlit_app.py — REAL-TIME SIMULATION + AUDIO ALARM
Predictive Maintenance Demo — MAML few-shot RUL + Early Warning

FIXES:
  1. Alarm now fires on MONITOR state too (VAE early detection fires well before
     RUL=30). Previously ALARM_STATES only included CRITICAL and EARLY WARNING,
     so anomaly detections at RUL=80+ were silently ignored.
  2. Compact single-screen layout — charts are shorter, panels collapsed by default.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os, sys, time, base64

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.inference.predictor import PredictiveMaintenancePredictor
from src.data.data_loader import load_preprocessed_data

# ── Config ────────────────────────────────────────────────────────────────────
CNN_CKPT    = "results/saved_models/early_pred_best.pth"
MAML_CKPT   = "results/saved_models/maml_meta_model_best.pth"
VAE_CKPT    = "results/saved_models/vae_early_warning.pth"
STATS_PATH  = "results/saved_models/baseline_stats.npz"
DATASET     = "data/processed/FD001_preprocessed.npz"
ALARM_SOUND = "src/deployment/alarm_sound.mp3"

MAML_K = 5

STATE_COLORS = {
    "CRITICAL":      "#EF4444",
    "EARLY WARNING": "#F59E0B",
    "MONITOR":       "#3B82F6",
    "HEALTHY":       "#10B981",
}

# FIX 1: MONITOR is now an alarm state — VAE/CUSUM early detection at high RUL
# should alert the operator, not silently pass.
ALARM_STATES = {"CRITICAL", "EARLY WARNING", "MONITOR"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_confirmed_detection_cycle(cusum_values, threshold_h, cycles, confirm_cycles=3):
    streak = 0
    for c, s in zip(cycles, cusum_values):
        streak = streak + 1 if s >= threshold_h else 0
        if streak >= confirm_cycles:
            return int(c - confirm_cycles + 1)
    return None


def get_display_state(rul_cycles, fault_prob, vae_confirmed):
    if rul_cycles <= 15 and fault_prob >= 0.90:
        return "CRITICAL", "Immediate intervention — schedule unplanned maintenance"
    if rul_cycles <= 30 and (fault_prob >= 0.70 or vae_confirmed):
        return "EARLY WARNING", "Degradation confirmed — schedule inspection"
    # FIX: vae_confirmed OR elevated fault_prob triggers MONITOR alarm early
    if vae_confirmed or fault_prob >= 0.50:
        return "MONITOR", "Emerging anomaly — monitor closely"
    return "HEALTHY", "No action required"


def render_alarm(display_state, display_cycle, alarm_slot):
    """Fire audio alarm once per unique cycle that enters any alarm state."""
    if (
        display_state in ALARM_STATES
        and display_cycle not in st.session_state.alarm_fired_at
    ):
        st.session_state.alarm_fired_at.add(display_cycle)
        if os.path.exists(ALARM_SOUND):
            with open(ALARM_SOUND, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            uid = f"alm{display_cycle}"
            alarm_slot.markdown(
                f"""
                <audio id="{uid}" autoplay>
                    <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
                </audio>
                <script>
                    (function(){{
                        var a = document.getElementById('{uid}');
                        if (!a) return;
                        a.volume = 0.9;
                        var p = a.play();
                        if (p) p.catch(function(){{}});
                    }})();
                </script>
                """,
                unsafe_allow_html=True,
            )


@st.cache_resource
def load_predictor():
    return PredictiveMaintenancePredictor(CNN_CKPT, MAML_CKPT, VAE_CKPT, STATS_PATH)


@st.cache_data
def load_data():
    return load_preprocessed_data(DATASET)


@st.cache_data(show_spinner="Running inference for engine…")
def precompute_engine(engine_id: int) -> pd.DataFrame:
    predictor = load_predictor()
    data      = load_data()
    max_rul   = float(data["max_rul"])
    test_ids  = np.array(data["test_engine_ids"])

    mask         = test_ids == engine_id
    X_eng        = data["X_test"][mask]
    y_eng        = data["y_test"][mask]
    y_eng_cycles = y_eng * max_rul
    n            = len(X_eng)

    if n < MAML_K:
        raise ValueError(f"Engine {engine_id} only has {n} windows — need ≥{MAML_K}.")

    predictor.adapt_to_engine(X_eng, y_eng, K=MAML_K)

    rows = []
    for i, (window, rul_true_cyc) in enumerate(zip(X_eng, y_eng_cycles)):
        out     = predictor.predict(window, cycle=i + 1)
        raw_rul = float(out["rul_cycles"])
        if raw_rul <= 1.5:
            out["rul_cycles"] = raw_rul * max_rul
        elif raw_rul > max_rul * 1.5:
            out["rul_cycles"] = raw_rul / max_rul
        out["rul_true"] = float(rul_true_cyc)
        rows.append(out)

    df    = pd.DataFrame(rows)
    stats = np.load(STATS_PATH, allow_pickle=True)
    h     = float(stats["threshold_h"])
    det   = get_confirmed_detection_cycle(df["vae_cusum"].values, h, df["cycle"].values)
    df["vae_confirmed"] = False if det is None else (df["cycle"] >= det)
    df["threshold_h"]   = h
    df["det_cycle"]     = np.nan if det is None else det
    return df


# ── Page setup ────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="EngineWatch · MAML", layout="wide", initial_sidebar_state="expanded")

# # Compact CSS — tighten vertical spacing so everything fits one screen
# st.markdown("""
# <style>
#     .block-container { padding-top: 1.2rem !important; padding-bottom: 0.4rem !important; }
#     .stMetric { padding: 4px 0 !important; }
#     .stMetric label { font-size: 0.7rem !important; }
#     .stMetric [data-testid="metric-container"] > div:first-child { font-size: 1rem !important; }
#     div[data-testid="stExpander"] { margin-bottom: 4px !important; }
#     h1 { margin-bottom: 0 !important; font-size: 1.3rem !important; }
#     h2, h3 { font-size: 1.2rem !important; margin: 4px 0 !important; }
#     .stCaption { font-size: 0.68rem !important; }
#     .element-container { margin-bottom: 2px !important; }
# </style>
# """, unsafe_allow_html=True)
st.set_page_config(
    page_title="EngineWatch · MAML",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem !important; padding-bottom: 0.4rem !important; }
    .stMetric { padding: 4px 0 !important; }
    .stMetric label { font-size: 0.7rem !important; }
    .stMetric [data-testid="metric-container"] > div:first-child { font-size: 1rem !important; }
    div[data-testid="stExpander"] { margin-bottom: 4px !important; }
    h2, h3 { font-size: 1.2rem !important; margin: 4px 0 !important; }
    .stCaption { font-size: 0.68rem !important; }
    .element-container { margin-bottom: 2px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='font-size:48px; margin-bottom:0;'>🛠 EngineWatch — Predictive Maintenance</h1>",
    unsafe_allow_html=True
)
# st.title("🛠 EngineWatch — Predictive Maintenance")

predictor   = load_predictor()
data        = load_data()
max_rul     = float(data["max_rul"])
test_ids    = np.array(data["test_engine_ids"])
engine_list = sorted(np.unique(test_ids).tolist())

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("rt_running",     False),
    ("rt_cycle_idx",   MAML_K),
    ("rt_engine_id",   engine_list[0]),
    ("alarm_fired_at", set()),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Engine")
    engine_id = st.selectbox("Engine ID", engine_list, index=0, label_visibility="collapsed")

    if engine_id != st.session_state.rt_engine_id:
        st.session_state.rt_running     = False
        st.session_state.rt_cycle_idx   = MAML_K
        st.session_state.alarm_fired_at = set()
        st.session_state.rt_engine_id   = engine_id

    mask      = test_ids == engine_id
    n_windows = int(mask.sum())

    st.markdown("### Playback")
    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("▶", use_container_width=True, disabled=st.session_state.rt_running)
    pause_btn = c2.button("⏸", use_container_width=True, disabled=not st.session_state.rt_running)
    reset_btn = c3.button("⏮", use_container_width=True)

    if start_btn:
        st.session_state.rt_running   = True
        st.session_state.rt_engine_id = engine_id
        st.rerun()
    if pause_btn:
        st.session_state.rt_running = False
        st.rerun()
    if reset_btn:
        st.session_state.rt_running     = False
        st.session_state.rt_cycle_idx   = MAML_K
        st.session_state.alarm_fired_at = set()
        st.rerun()

    rt_speed = st.slider("Speed (s/cycle)", 0.3, 5.0, 1.5, 0.1, format="%.1f s")
    st.progress(
        min(st.session_state.rt_cycle_idx / max(n_windows, 1), 1.0),
        text=f"Cycle {st.session_state.rt_cycle_idx}/{n_windows}",
    )

    st.markdown("---")
    st.markdown(
        "**Models**\n"
        "- CNN-LSTM · F1=0.876 · AUC=0.990\n"
        "- MAML · few-shot (K=5)\n"
        "- VAE+CUSUM · FAR=10%\n\n"
        "**Alarm states**\n"
        "- 🔴 CRITICAL · 🟠 EARLY WARNING\n"
        "- 🔵 MONITOR ← *early detection alarm*"
    )

# ── Precompute ────────────────────────────────────────────────────────────────
try:
    full_df = precompute_engine(engine_id)
except ValueError as e:
    st.error(str(e))
    st.stop()

n_total             = len(full_df)
h                   = float(full_df["threshold_h"].iloc[0])
_det_raw            = full_df["det_cycle"].iloc[0]
confirmed_det_cycle = None if pd.isna(_det_raw) else int(_det_raw)

# ── Display cycle ─────────────────────────────────────────────────────────────
if st.session_state.rt_running:
    display_cycle = int(st.session_state.rt_cycle_idx)
else:
    display_cycle = st.sidebar.slider(
        "Manual cycle", MAML_K, n_total, n_total, 1, key="static_slider"
    )

display_cycle = max(MAML_K, min(display_cycle, n_total))
df = full_df.iloc[:display_cycle].copy()

# ── Current state ─────────────────────────────────────────────────────────────
latest               = df.iloc[-1]
latest_vae_confirmed = bool(latest["vae_confirmed"])
display_state, display_action = get_display_state(
    rul_cycles=float(latest["rul_cycles"]),
    fault_prob=float(latest["fault_prob"]),
    vae_confirmed=latest_vae_confirmed,
)
color = STATE_COLORS[display_state]

# ── Alarm (FIX: MONITOR now included) ────────────────────────────────────────
alarm_slot = st.empty()
render_alarm(display_state, display_cycle, alarm_slot)

# ── Status bar ────────────────────────────────────────────────────────────────
icon_map = {"CRITICAL": "🚨", "EARLY WARNING": "⚠️", "MONITOR": "🔔", "HEALTHY": "✅"}
icon = icon_map[display_state]

st.markdown(
    f"""
    <div style="background:{color}18;border:2px solid {color};border-radius:8px;
                padding:8px 16px;margin-bottom:6px;display:flex;align-items:center;gap:16px">
        <span style="font-size:20px;font-weight:700;color:{color}">{icon} {display_state}</span>
        <span style="font-size:13px;color:#ccc">{display_action}</span>
        <span style="font-size:12px;color:#aaa;margin-left:auto">
            RUL ≈ <b>{latest['rul_cycles']:.0f}</b> cycles &nbsp;
            Fault P = <b>{latest['fault_prob']:.3f}</b> &nbsp;
            CUSUM = <b>{latest['vae_cusum']:.2f}</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Metrics row ───────────────────────────────────────────────────────────────
rul_pct = max(0, min(100, int((1 - float(latest["rul_cycles"]) / max_rul) * 100)))
fp_pct  = int(float(latest["fault_prob"]) * 100)
triggered = bool(confirmed_det_cycle and confirmed_det_cycle <= display_cycle)

m1, m2, m3, m4 = st.columns(4)
m1.metric("RUL", f"{int(latest['rul_cycles'])} cycles", f"{rul_pct}% worn")
m2.metric("Fault probability", f"{fp_pct}%")
m3.metric("Anomaly detector", "TRIGGERED" if triggered else "CLEAR",
          f"from cycle {confirmed_det_cycle}" if triggered else "awaiting 3-cycle streak")
m4.metric("Stage", display_state)

# ── Charts — two columns, compact height ─────────────────────────────────────
col_rul, col_vae = st.columns(2)

with col_rul:
    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(
        x=df["cycle"], y=df["rul_true"],
        name="True", line=dict(color="#94A3B8", dash="dot"), mode="lines"
    ))
    fig_rul.add_trace(go.Scatter(
        x=df["cycle"], y=df["rul_cycles"],
        name="Predicted", line=dict(color="#6366F1", width=2), mode="lines"
    ))
    fig_rul.add_hline(
        y=30, line_dash="dash", line_color="#EF4444",
        annotation_text="30-cycle threshold", annotation_position="top right",
    )
    if confirmed_det_cycle and confirmed_det_cycle <= display_cycle:
        row  = full_df[full_df["cycle"] == confirmed_det_cycle]
        lead = float(row["rul_true"].values[0]) - 30 if len(row) else 0
        fig_rul.add_vline(
            x=confirmed_det_cycle, line_dash="dash", line_color="#F59E0B",
            annotation_text=f"Early det (+{lead:.0f})" if lead >= 0 else f"Late ({lead:.0f})",
            annotation_font_color="#F59E0B",
        )
    fig_rul.update_layout(
        title=f"RUL — Engine {engine_id}  (cycle {display_cycle}/{n_total})",
        xaxis_title="Cycle", yaxis_title="RUL (cycles)",
        legend=dict(orientation="h", y=1.12), height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="#e2e8f0"), margin=dict(t=36, b=36, l=48, r=16),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

with col_vae:
    onset_rows  = df[df["rul_true"] <= 30]
    onset_cycle = int(onset_rows["cycle"].iloc[0]) if len(onset_rows) else None
    det_cycle   = confirmed_det_cycle
    latency     = (onset_cycle - det_cycle) if (det_cycle and onset_cycle) else None

    fig_vae = go.Figure()
    fig_vae.add_trace(go.Scatter(
        x=df["cycle"], y=df["vae_cusum"],
        name="CUSUM", line=dict(color="#F59E0B", width=2)
    ))
    fig_vae.add_trace(go.Scatter(
        x=df["cycle"], y=df["fault_prob"],
        name="Fault P", line=dict(color="#6366F1", width=1.5, dash="dot"),
        yaxis="y2",
    ))
    fig_vae.add_hline(
        y=h, line_dash="dash", line_color="#EF4444",
        annotation_text=f"h={h:.2f}", annotation_font_color="#EF4444",
        annotation_position="bottom right",
    )
    if onset_cycle and onset_cycle <= display_cycle:
        fig_vae.add_vline(x=onset_cycle, line_dash="dash", line_color="#EF4444",
                          line_width=1.5, annotation_text="Onset",
                          annotation_font_color="#EF4444", annotation_position="top left")
    if det_cycle and det_cycle <= display_cycle:
        lat_label = f"Det(Δ={latency:+d})" if latency is not None else "Detection"
        fig_vae.add_vline(x=det_cycle, line_dash="solid", line_color="#F59E0B",
                          line_width=2, annotation_text=lat_label,
                          annotation_font_color="#F59E0B", annotation_position="top right",
                          annotation_yshift=12)
        if onset_cycle and det_cycle != onset_cycle and onset_cycle <= display_cycle:
            x0, x1 = sorted([det_cycle, onset_cycle])
            fig_vae.add_vrect(x0=x0, x1=x1, fillcolor="#F59E0B",
                              opacity=0.10, line_width=0, layer="below")

    fig_vae.update_layout(
        title="VAE + CUSUM · Early Warning",
        xaxis_title="Cycle", yaxis_title="CUSUM",
        yaxis2=dict(title="Fault P", overlaying="y", side="right",
                    range=[0, 1], showgrid=False),
        legend=dict(orientation="h", y=1.12), height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)",
        font=dict(color="#e2e8f0"), margin=dict(t=36, b=36, l=48, r=48),
    )
    st.plotly_chart(fig_vae, use_container_width=True)

    if latency is not None and det_cycle and det_cycle <= display_cycle:
        if latency > 0:
            st.success(f"✅ Early: **{latency} cycles** before onset (det@{det_cycle}, onset@{onset_cycle})")
        elif latency < 0:
            st.warning(f"⚠️ Late: **{abs(latency)} cycles** after onset")
        else:
            st.info(f"ℹ️ Exact onset detection @ cycle {onset_cycle}")

# ── Debug (collapsed by default) ─────────────────────────────────────────────
with st.expander("Debug", expanded=False):
    st.caption(
        f"max_rul={max_rul:.1f} | cycles={n_total} | "
        f"pred {df['rul_cycles'].min():.0f}–{df['rul_cycles'].max():.0f} | "
        f"true {df['rul_true'].min():.0f}–{df['rul_true'].max():.0f}"
    )
    if df["rul_cycles"].max() < 5:
        st.error("Predicted RUL near 0 — check predictor output units.")
    elif abs(df["rul_cycles"].mean() - df["rul_true"].mean()) > max_rul * 0.6:
        st.warning("Large RUL offset — possible scaling issue.")
    else:
        st.success("RUL range looks reasonable.")

# ── RT tick ───────────────────────────────────────────────────────────────────
if st.session_state.rt_running:
    next_idx = display_cycle + 1
    if next_idx <= n_total:
        time.sleep(rt_speed)
        st.session_state.rt_cycle_idx = next_idx
        st.rerun()
    else:
        st.session_state.rt_running = False
        st.success(f"✅ Playback complete — all {n_total} cycles processed.")