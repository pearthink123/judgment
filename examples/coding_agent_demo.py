#!/usr/bin/env python3
"""
coding_agent_demo.py

Self-contained demo of the DecisionEngine controlling a simulated
coding Agent harness loop.

Task: implement + test a small Python utility (LRU cache decorator).

Architecture:
  Layer 1 — CUSUM anomaly detection (Hawkes-corrected surprisal)
  Layer 2 — 3-state HMM latent-state inference (Healthy/Degraded/Broken)
  Layer 3 — Threshold-gate decision (Continue/Correct/Escalate/Gather)

Run:
    python examples/coding_agent_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.engine import DecisionEngine, ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER


# ---------------------------------------------------------------------------
# Simulated harness — maps Engine actions to environment outcomes
# ---------------------------------------------------------------------------
def simulate_coding_harness(action: str, state: dict) -> dict:
    """
    Fake execution environment.  The Engine chooses an action;
    this function simulates what would happen in a real harness.
    """
    state.setdefault("code_written", False)
    state.setdefault("tests_run", 0)
    state.setdefault("errors", 0)
    state.setdefault("progress", 0.0)
    state.setdefault("verified", False)

    obs: dict = {}

    if action == ACTION_CONTINUE:
        # Execute the next planned tool call (simulated as "edit_code")
        if not state["code_written"]:
            state["code_written"] = True
            obs["tool_ok"] = True
            obs["progress_delta"] = 0.42
            state["progress"] = max(state["progress"], 0.42)
        else:
            obs["tool_ok"] = True
            obs["progress_delta"] = 0.12
            state["progress"] = min(0.93, state["progress"] + 0.12)

        obs["error_count_delta"] = 0

    elif action == ACTION_CORRECT:
        # Verify current state, run checks, fix issues
        if state["code_written"] and state["tests_run"] >= 1:
            success = np.random.rand() > 0.25
            obs["tool_ok"] = success
            if success:
                obs["progress_delta"] = 0.18
                state["progress"] = min(0.95, state["progress"] + 0.18)
                obs["error_count_delta"] = 0
            else:
                obs["progress_delta"] = -0.04
                obs["error_count_delta"] = 1
                state["errors"] += 1
        else:
            # Correct before anything is built — mild info gathering
            obs["tool_ok"] = True
            obs["progress_delta"] = 0.05
            obs["error_count_delta"] = 0

    elif action == ACTION_GATHER:
        # Read file, check status — low-cost info
        obs["tool_ok"] = True
        delta = 0.08 if not state["code_written"] else 0.04
        obs["progress_delta"] = delta
        state["progress"] = min(0.95, state["progress"] + delta)
        obs["error_count_delta"] = 0

    elif action == ACTION_ESCALATE:
        # Ask user for help
        obs["tool_ok"] = True
        obs["has_user_msg"] = True
        obs["progress_delta"] = 0.06
        state["progress"] = min(0.90, state["progress"] + 0.06)
        obs["error_count_delta"] = 0

    else:
        obs["tool_ok"] = True
        obs["progress_delta"] = 0.02
        obs["error_count_delta"] = 0

    state["steps"] = state.get("steps", 0) + 1
    obs["progress"] = round(state["progress"], 3)
    obs["errors_so_far"] = state["errors"]
    obs["tests_run"] = state["tests_run"]

    # Task completion check
    if state["code_written"] and state["progress"] >= 0.90:
        obs["task_completed"] = True

    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DecisionEngine — 3-Layer Math-Driven Coding Agent Demo")
    print("  Layer 1: CUSUM + Hawkes (anomaly detection)")
    print("  Layer 2: 3-State HMM (Healthy / Degraded / Broken)")
    print("  Layer 3: Threshold Gate (Continue / Correct / Escalate / Gather)")
    print("=" * 70)
    print()

    engine = DecisionEngine(seed=42)
    state: dict = {}

    # Initial observation — all clean
    obs: dict = {
        "tool_ok": True,
        "progress_delta": 0.0,
        "has_user_msg": False,
        "error_count_delta": 0,
    }

    max_steps = 20
    for step in range(1, max_steps + 1):
        decision = engine.step(obs)

        belief = decision.belief
        diag = decision.layer_diagnostics

        print(f"\n[Step {step:02d}]")
        print(
            f"  Belief: H={belief['healthy']:.3f}  "
            f"D={belief['degraded']:.3f}  B={belief['broken']:.3f}"
        )
        print(
            f"  Drift: S={decision.drift:.3f}  "
            f"anomaly={decision.anomaly}  "
            f"CUSUM alarms={diag['cusum_alarm_count']}"
        )
        print(
            f"  Hawkes λ: succ={diag['hawkes_intensities'][0]:.3f}  "
            f"err={diag['hawkes_intensities'][1]:.3f}  "
            f"user={diag['hawkes_intensities'][2]:.3f}  "
            f"tool={diag['hawkes_intensities'][3]:.3f}"
        )
        print(
            f"  → ACTION: {decision.action.upper()}  "
            f"(confidence={decision.confidence:.3f})"
        )
        print(f"    Rationale: {decision.rationale}")

        # Execute in simulated harness
        outcome = simulate_coding_harness(decision.action, state)

        # Inject occasional errors to test detection
        if step == 6 and not outcome.get("task_completed"):
            outcome["tool_ok"] = False
            outcome["error_count_delta"] = 1
            outcome["progress_delta"] = -0.05
            state["errors"] = state.get("errors", 0) + 1
            print("  [!] INJECTED ERROR at step 6")

        if step == 7 and not outcome.get("task_completed"):
            outcome["tool_ok"] = False
            outcome["error_count_delta"] = 1
            outcome["progress_delta"] = -0.03
            state["errors"] = state.get("errors", 0) + 1
            print("  [!] INJECTED ERROR at step 7")

        obs = outcome

        print(
            f"  Outcome: progress={outcome.get('progress', 0):.2f}  "
            f"tool_ok={outcome.get('tool_ok')}  "
            f"errors_so_far={outcome.get('errors_so_far', 0)}"
        )

        if outcome.get("task_completed"):
            print("\n" + "=" * 70)
            print("TASK COMPLETED")
            print(f"Total steps: {step}")
            print(f"Final belief: {belief}")
            break

        if decision.action == ACTION_ESCALATE and decision.confidence > 0.50:
            print("\n  Escalated to user.")
            break
    else:
        print("\nReached max steps.")

    # Summary
    print("\n" + "=" * 70)
    print("Decision Log Summary (last 8):")
    df = engine.get_diagnostics_dataframe()
    if not df.empty:
        print(df.tail(8).to_string(index=False))

    print("\nRun the Streamlit dashboard:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
