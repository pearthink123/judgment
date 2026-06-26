#!/usr/bin/env python3
"""
coding_agent_demo.py

A self-contained demo of the Math Judgment Engine controlling a simulated
coding Agent Harness loop.

Task: Implement + test a small Python utility (e.g. LRU cache decorator + usage).

At each step the JudgmentEngine decides the next action using:
- Hawkes process (proactive trigger)
- Bayesian belief over task success / error risk / stuck
- EVOI (which action gives the most value right now)
- PID + stochastic control for regulation

Run:
    python examples/coding_agent_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.judgment_engine import JudgmentEngine


def simulate_coding_harness(action: str, state: dict) -> dict:
    """
    Fake execution environment (improved so the task can actually succeed).
    The math engine now has a chance to drive a successful run.
    """
    state.setdefault("code_written", False)
    state.setdefault("tests_run", 0)
    state.setdefault("errors", 0)
    state.setdefault("progress", 0.0)
    state.setdefault("verified", False)

    obs = {"steps_taken": state.get("steps", 0)}

    if action == "think":
        obs["progress_delta"] = 0.07
        obs["tool_success"] = True
        state["progress"] = min(0.95, state["progress"] + 0.07)

    elif action == "read_file":
        obs["tool_success"] = True
        delta = 0.09 if not state["code_written"] else 0.06
        obs["progress_delta"] = delta
        state["progress"] = min(0.95, state["progress"] + delta)

    elif action == "edit_code":
        if not state["code_written"]:
            state["code_written"] = True
            obs["tool_success"] = True
            obs["progress_delta"] = 0.42
            state["progress"] = max(state["progress"], 0.42)
        else:
            obs["tool_success"] = True
            obs["progress_delta"] = 0.15
            state["progress"] = min(0.93, state["progress"] + 0.15)

    elif action == "run_tests":
        state["tests_run"] += 1
        if state["code_written"]:
            # After code is written, tests have a reasonable chance
            success = np.random.rand() > (0.28 if state["errors"] < 3 else 0.45)
            obs["tool_success"] = success
            if success:
                obs["progress_delta"] = 0.18
                state["progress"] = max(state["progress"], 0.72)
                obs["error_count_delta"] = 0
            else:
                obs["error_count_delta"] = 1
                state["errors"] += 1
                state["progress"] = max(0.25, state["progress"] - 0.04)
        else:
            # Testing before writing code → bad idea
            obs["tool_success"] = False
            obs["error_count_delta"] = 1
            state["errors"] += 1

    elif action == "verify":
        if state["tests_run"] >= 1 and state["progress"] >= 0.65:
            obs["tool_success"] = True
            obs["progress_delta"] = 0.15
            state["verified"] = True
            state["progress"] = min(0.98, state["progress"] + 0.15)
        else:
            obs["tool_success"] = False
            obs["error_count_delta"] = 1

    elif action == "escalate_to_user":
        obs["user_response"] = "positive" if state["errors"] >= 2 else "neutral"
        obs["tool_success"] = True
        obs["progress_delta"] = 0.08
        state["progress"] = min(0.9, state["progress"] + 0.08)

    else:
        obs["tool_success"] = True
        obs["progress_delta"] = 0.02

    state["steps"] = state.get("steps", 0) + 1
    obs["progress"] = round(state["progress"], 3)
    obs["errors_so_far"] = state["errors"]
    obs["tests_run"] = state["tests_run"]

    if state["verified"] or (state["code_written"] and state["progress"] >= 0.91 and state["tests_run"] >= 1):
        obs["task_completed"] = True

    return obs


def main():
    print("=" * 70)
    print("MathHarness Judgment Engine — Coding Agent Demo")
    print("Task: Build and verify a small Python utility using math-driven decisions")
    print("=" * 70)
    print()

    engine = JudgmentEngine(seed=137)
    state = {}
    obs = {
        "progress_delta": 0.0,
        "tool_success": True,
        "error_count_delta": 0,
        "steps_taken": 0,
    }

    max_steps = 18
    for step in range(1, max_steps + 1):
        decision = engine.decide(obs, {"task": "implement_lru_cache_utility"})

        print(f"\n[Step {step:02d}]")
        print(f"  Belief: task_success={decision.belief['task_success']:.3f}  "
              f"error_risk={decision.belief['error_risk']:.3f}  stuck={decision.belief['stuck']:.3f}")
        print(f"  Trigger (Hawkes): {decision.trigger_intensity:.3f}")
        print(f"  EVOI chosen: {decision.evoi:.3f}   |  confidence={decision.confidence:.3f}")
        print(f"  Control: agg={decision.control.aggressiveness:.2f}  corr={decision.control.correction_gain:.2f}  throttle={decision.control.throttle:.2f}")
        print(f"  → DECISION: {decision.action}")
        print(f"    Rationale: {decision.rationale}")

        # Execute
        outcome = simulate_coding_harness(decision.action, state)
        obs = outcome

        print(f"  Outcome: progress={outcome.get('progress',0):.2f}  "
              f"success={outcome.get('tool_success')}  errors={outcome.get('errors_so_far',0)}")

        if outcome.get("task_completed"):
            print("\n" + "=" * 70)
            print("✅ TASK COMPLETED SUCCESSFULLY")
            print(f"Total steps: {step}")
            print(f"Final belief: {decision.belief}")
            break

        if decision.belief.get("stuck", 0) > 0.78 and decision.action == "escalate_to_user":
            print("\n⚠️  Escalated to user (high stuck risk)")
            break
    else:
        print("\nReached max steps.")

    print("\n" + "=" * 70)
    print("Decision log summary (last 6):")
    df = engine.get_diagnostics_dataframe()
    if not df.empty:
        print(df.tail(6).to_string(index=False))

    print("\nDone. You can also run the Streamlit dashboard:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
