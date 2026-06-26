"""
Streamlit dashboard for MathHarness Judgment Engine.

Shows live belief evolution, Hawkes trigger intensity, EVOI scores and decisions.

Run:
    cd math_harness_judgment
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

from core.judgment_engine import JudgmentEngine
from examples.coding_agent_demo import simulate_coding_harness

st.set_page_config(page_title="MathHarness Judgment Engine", layout="wide")
st.title("MathHarness Judgment Engine — Live Dashboard")
st.markdown("**Mathematical Decision Core for Production Agent Harnesses**  |  Poisson/Hawkes · Bayesian · EVOI · PID Control")

# Sidebar controls
st.sidebar.header("Simulation Controls")
max_steps = st.sidebar.slider("Max steps", 6, 30, 16, 1)
seed = st.sidebar.number_input("Random seed", 1, 99999, 137)
run_button = st.sidebar.button("▶ Run / Re-run Simulation", type="primary")

if "engine" not in st.session_state or run_button:
    engine = JudgmentEngine(seed=int(seed))
    state = {}
    obs = {"progress_delta": 0.0, "tool_success": True, "error_count_delta": 0, "steps_taken": 0}

    history = []
    for step in range(1, max_steps + 1):
        decision = engine.decide(obs, {"task": "demo"})
        outcome = simulate_coding_harness(decision.action, state)
        obs = outcome
        history.append({
            "step": step,
            "action": decision.action,
            "confidence": decision.confidence,
            "trigger": decision.trigger_intensity,
            "evoi": decision.evoi,
            "task_success": decision.belief["task_success"],
            "error_risk": decision.belief["error_risk"],
            "stuck": decision.belief["stuck"],
            "progress": outcome.get("progress", 0),
        })
        if outcome.get("task_completed"):
            break

    st.session_state.engine = engine
    st.session_state.history = history
    st.session_state.df = pd.DataFrame(history)

df = st.session_state.df

col1, col2 = st.columns(2)

with col1:
    st.subheader("Belief Evolution (Bayesian)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["step"], df["task_success"], label="P(task_success)", linewidth=2)
    ax.plot(df["step"], df["error_risk"], label="P(error_risk)", linewidth=2)
    ax.plot(df["step"], df["stuck"], label="P(stuck)", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("Trigger Intensity (Hawkes Process)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["step"], df["trigger"], color="#e74c3c", linewidth=2.5, label="Trigger intensity")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="baseline")
    ax.fill_between(df["step"], 0, df["trigger"], alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Step")
    ax.set_ylabel("λ(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.subheader("Actions & Value Metrics")

fig, ax = plt.subplots(figsize=(10, 3.8))
colors = {"think": "#3498db", "read_file": "#9b59b6", "edit_code": "#2ecc71",
          "run_tests": "#e67e22", "verify": "#1abc9c", "escalate_to_user": "#e74c3c"}
for act in df["action"].unique():
    mask = df["action"] == act
    ax.scatter(df.loc[mask, "step"], df.loc[mask, "evoi"],
               s=90, label=act, c=colors.get(act, "gray"), alpha=0.85, edgecolors="black", linewidths=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("EVOI of chosen action")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Decision table
st.subheader("Decision Trace")
st.dataframe(
    df[["step", "action", "confidence", "trigger", "evoi", "task_success", "error_risk", "progress"]].style.format({
        "confidence": "{:.2f}",
        "trigger": "{:.2f}",
        "evoi": "{:.2f}",
        "task_success": "{:.2f}",
        "error_risk": "{:.2f}",
        "progress": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.caption("This dashboard visualizes how stochastic processes + Bayesian updating + information-theoretic action selection + control theory produce stable, non-heuristic agent decisions.")
st.caption("Target use case: plug the JudgmentEngine into DeepSeek-style or other production Harnesses.")
