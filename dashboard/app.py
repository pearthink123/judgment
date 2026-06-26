"""
Streamlit dashboard for the 3-layer DecisionEngine.

Visualises:
  - HMM belief evolution (Layer 2)
  - CUSUM drift + alarms (Layer 1)
  - Decision trace (Layer 3)
  - Hawkes per-type intensities

Run:
    pip install -r requirements.txt
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.engine import DecisionEngine
from examples.coding_agent_demo import simulate_coding_harness

st.set_page_config(page_title="DecisionEngine Dashboard", layout="wide")
st.title("DecisionEngine — 3-Layer Math-Driven Dashboard")
st.markdown(
    "**Layer 1:** CUSUM + Hawkes anomaly detection  |  "
    "**Layer 2:** 3-State HMM (Healthy / Degraded / Broken)  |  "
    "**Layer 3:** Threshold-gate decisions"
)

# Sidebar
st.sidebar.header("Simulation Controls")
max_steps = st.sidebar.slider("Max steps", 6, 40, 20, 1)
seed = st.sidebar.number_input("Random seed", 1, 99999, 42)
inject_errors = st.sidebar.checkbox("Inject errors at step 6–7", value=True)
run_button = st.sidebar.button("Run / Re-run Simulation", type="primary")

if "engine" not in st.session_state or run_button:
    engine = DecisionEngine(seed=int(seed))
    state: dict = {}
    obs: dict = {
        "tool_ok": True,
        "progress_delta": 0.0,
        "has_user_msg": False,
        "error_count_delta": 0,
    }

    history: list[dict] = []
    for step_idx in range(1, max_steps + 1):
        decision = engine.step(obs)
        outcome = simulate_coding_harness(decision.action, state)

        # Inject errors if enabled
        if inject_errors:
            if step_idx == 6 and not outcome.get("task_completed"):
                outcome["tool_ok"] = False
                outcome["error_count_delta"] = 1
                outcome["progress_delta"] = -0.05
                state["errors"] = state.get("errors", 0) + 1
            if step_idx == 7 and not outcome.get("task_completed"):
                outcome["tool_ok"] = False
                outcome["error_count_delta"] = 1
                outcome["progress_delta"] = -0.03
                state["errors"] = state.get("errors", 0) + 1

        diag = decision.layer_diagnostics
        hawkes_lam = diag["hawkes_intensities"]

        history.append({
            "step": step_idx,
            "action": decision.action,
            "confidence": decision.confidence,
            "P_H": decision.belief["healthy"],
            "P_D": decision.belief["degraded"],
            "P_B": decision.belief["broken"],
            "drift": decision.drift,
            "anomaly": decision.anomaly,
            "lam_success": hawkes_lam[0],
            "lam_error": hawkes_lam[1],
            "lam_user": hawkes_lam[2],
            "lam_tool": hawkes_lam[3],
            "progress": outcome.get("progress", 0),
            "alarms_total": diag["cusum_alarm_count"],
        })

        obs = outcome
        if outcome.get("task_completed"):
            break

    st.session_state.engine = engine
    st.session_state.history = history
    st.session_state.df = pd.DataFrame(history)

# Guard: first load — nothing to show yet
if "df" not in st.session_state:
    st.info("Click **Run / Re-run Simulation** in the sidebar to start.")
    st.stop()

df: pd.DataFrame = st.session_state.df

# ---- Row 1: Belief + Drift ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("HMM Belief Evolution (Layer 2)")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df["step"], df["P_H"], label="P(Healthy)", linewidth=2, color="#2ecc71")
    ax.plot(df["step"], df["P_D"], label="P(Degraded)", linewidth=2, color="#f39c12")
    ax.plot(df["step"], df["P_B"], label="P(Broken)", linewidth=2, color="#e74c3c")
    # Threshold references
    ax.axhline(0.45, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1, label="θ_B=0.45")
    ax.axhline(0.35, color="#f39c12", linestyle="--", alpha=0.5, linewidth=1, label="θ_D=0.35")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("CUSUM Drift + Alarms (Layer 1)")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df["step"], df["drift"], color="#3498db", linewidth=2, label="Drift S_t")
    ax.axhline(4.0, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1.5, label="Threshold h=4.0")
    # Mark alarm points
    alarm_mask = df["anomaly"] == True
    if alarm_mask.any():
        ax.scatter(
            df.loc[alarm_mask, "step"],
            df.loc[alarm_mask, "drift"],
            color="#e74c3c", s=100, marker="x", linewidths=2.5, zorder=5,
            label=f"Alarms ({alarm_mask.sum()})",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative drift S")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ---- Row 2: Hawkes + Actions ----
col3, col4 = st.columns(2)

with col3:
    st.subheader("Hawkes Per-Type Intensities")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["step"], df["lam_tool"], label="λ_tool", linewidth=1.5, color="#9b59b6")
    ax.plot(df["step"], df["lam_error"], label="λ_error", linewidth=1.5, color="#e74c3c")
    ax.plot(df["step"], df["lam_success"], label="λ_success", linewidth=1.5, color="#2ecc71")
    ax.plot(df["step"], df["lam_user"], label="λ_user", linewidth=1.5, color="#3498db")
    ax.set_xlabel("Step")
    ax.set_ylabel("Intensity λ_d(t)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col4:
    st.subheader("Decision Trace (Layer 3)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    action_colors = {
        "continue": "#2ecc71",
        "correct": "#f39c12",
        "escalate": "#e74c3c",
        "gather": "#3498db",
    }
    for act in df["action"].unique():
        mask = df["action"] == act
        ax.scatter(
            df.loc[mask, "step"],
            [1.0] * mask.sum(),
            s=120,
            label=act,
            c=action_colors.get(act, "gray"),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
            marker="s",
        )
    ax.set_xlabel("Step")
    ax.set_yticks([])
    ax.set_ylim(0.5, 1.5)
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig)

# ---- Table ----
st.subheader("Detailed Trace")
st.dataframe(
    df[[
        "step", "action", "confidence", "P_H", "P_D", "P_B",
        "drift", "anomaly", "progress",
    ]].style.format({
        "confidence": "{:.3f}",
        "P_H": "{:.3f}",
        "P_D": "{:.3f}",
        "P_B": "{:.3f}",
        "drift": "{:.3f}",
        "progress": "{:.2f}",
    }),
    width="stretch",
    hide_index=True,
)

# Footer
st.markdown("---")
st.caption(
    "3-layer architecture: CUSUM (Page 1954) + HMM Forward (Rabiner 1989) + "
    "Hawkes baseline (Hawkes 1971). Each layer has a citable mathematical "
    "foundation — no heuristic-only decision making."
)
