#!/usr/bin/env python3
"""
LangGraph agent with judgment oversight — complete runnable example.

This is a minimal but realistic ReAct-style agent that:
  1. Receives a coding task
  2. Plans, calls tools, observes results
  3. After each tool execution, the judgment engine checks health
  4. On CORRECT → corrective advice injected into agent messages
  5. On ESCALATE → stops and prints a health summary

Requires:
    pip install -e ".[dashboard]"    # for base + Streamlit deps

Optional (for real LLM):
    pip install -e ".[llm]"          # adds openai
    export DEEPSEEK_API_KEY=...

If langgraph is not installed, this script runs in a simulated
dict-based loop that follows the same pattern.

Run:
    python examples/langgraph_agent.py
    python examples/langgraph_agent.py --task "Write a function to merge two sorted lists"
    python examples/langgraph_agent.py --real-llm --model deepseek-chat
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import DecisionEngine
from core.pomdp import RewardConfig

# ---------------------------------------------------------------------------
# Simulated agent loop (no langgraph dependency)
# ---------------------------------------------------------------------------
def simulated_react_loop(engine: DecisionEngine, task: str, max_steps: int = 20):
    """
    A minimal ReAct loop that simulates an LLM agent.

    The loop:
      agent → tool → judgment → agent → ...

    This mirrors exactly what a LangGraph graph would do, but in plain
    Python so you can run it without installing langgraph.
    """
    print("=" * 65)
    print(f"Task: {task}")
    print(f"Engine: POMDP ({'general' if engine._policy else 'threshold'} preset)")
    print("=" * 65)

    # Simulated state — in real code this would be a TypedDict or Pydantic model
    import numpy as np
    rng = np.random.default_rng(42)

    messages: list[dict] = [{"role": "user", "content": task}]
    completed = False
    error_streak = 0

    for step_n in range(1, max_steps + 1):
        # ---- Agent step (simulated) ----
        # In a real harness the LLM decides what tool to call.
        # Here we simulate: mostly success, occasionally failure.
        if error_streak >= 3:
            # Recovering — simulate a corrective action
            tool_ok = rng.random() > 0.2
            progress_delta = 0.10 if tool_ok else -0.02
            tool_name = "verify"
        elif step_n <= 2:
            tool_ok = True
            progress_delta = 0.20
            tool_name = "read_file"
        elif error_streak > 0:
            tool_ok = rng.random() > 0.35
            progress_delta = 0.08 if tool_ok else -0.03
            tool_name = "edit_code"
        else:
            tool_ok = rng.random() > 0.12  # ~12% failure rate
            progress_delta = 0.15 if tool_ok else -0.04
            tool_name = "edit_code"

        if not tool_ok:
            error_streak += 1
        else:
            error_streak = max(0, error_streak - 1)

        error_count_delta = 0 if tool_ok else 1

        messages.append({
            "role": "assistant",
            "content": f"[{tool_name}] {'OK' if tool_ok else 'FAIL'} progress={progress_delta:+.2f}",
        })

        # ---- Build state ----
        state = {
            "messages": list(messages),
            "tool_ok": tool_ok,
            "progress_delta": progress_delta,
            "has_user_msg": False,
            "error_count_delta": error_count_delta,
            "step": step_n,
        }

        # ---- Judgment step ----
        decision = engine.step({
            "tool_ok": tool_ok,
            "progress_delta": progress_delta,
            "has_user_msg": False,
            "error_count_delta": error_count_delta,
        })

        belief = decision.belief
        marker = "C" if decision.action == "continue" else (
            "!" if decision.action == "correct" else "X"
        )
        print(
            f"  [{step_n:02d}] [{marker}] {decision.action:9s}  "
            f"H={belief['healthy']:.3f} D={belief['degraded']:.3f} "
            f"B={belief['broken']:.3f}  "
            f"drift={decision.drift:.3f}  "
            f"{'[!]' if decision.anomaly else '   '}"
        )

        # ---- Inject corrective advice ----
        if decision.action == "correct" and decision.corrective_advice:
            advice = decision.corrective_advice
            msg = (
                f"[JUDGMENT · {advice.action_type.upper()}] {advice.summary}"
            )
            print(f"      → {msg}")
            messages.append({"role": "system", "content": msg})

        if decision.action == "gather":
            messages.append({
                "role": "system",
                "content": "[JUDGMENT · GATHERING] Collect more info before acting.",
            })

        # ---- Escalate ----
        if decision.action == "escalate":
            print("\n  === ESCALATED ===")
            print(f"  Reason: {decision.rationale}")
            print(f"  Belief: {belief}")
            return {
                "status": "escalated",
                "steps": step_n,
                "final_belief": belief,
                "rationale": decision.rationale,
            }

        # ---- Check completion (simple heuristic) ----
        if step_n >= 8 and decision.action == "continue" and error_streak == 0:
            completed = True
            print(f"\n  [OK] Task completed in {step_n} steps.")
            return {
                "status": "success",
                "steps": step_n,
                "final_belief": belief,
            }

    print(f"\n  [!] Max steps ({max_steps}) reached.")
    return {
        "status": "max_steps",
        "steps": max_steps,
        "final_belief": engine.decision_log[-1].belief if engine.decision_log else {},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LangGraph-style agent with judgment oversight"
    )
    parser.add_argument(
        "--task", default="Implement an LRU cache decorator in Python",
        help="Task description for the agent",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--preset", default="general",
        choices=["general", "conservative", "permissive"],
        help="Reward function preset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = DecisionEngine(
        reward=RewardConfig.preset(args.preset),
        use_pomdp=True,
        use_corrective=True,
        seed=args.seed,
    )

    result = simulated_react_loop(engine, args.task, args.max_steps)

    print()
    print(f"Final: {result['status']} in {result['steps']} steps")
    belief = result["final_belief"]
    print(f"Belief: H={belief.get('healthy', 0):.3f} "
          f"D={belief.get('degraded', 0):.3f} "
          f"B={belief.get('broken', 0):.3f}")


if __name__ == "__main__":
    main()
