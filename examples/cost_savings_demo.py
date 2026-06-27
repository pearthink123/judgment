#!/usr/bin/env python3
"""
Production cost savings demo — quantifies how much judgment saves.

Simulates a realistic coding agent loop (200 steps, 4 tasks) with:
  - Task 1: completes normally (25 steps)
  - Task 2: silent degradation from step 15 (50 steps, many wasted)
  - Task 3: catastrophic failure at step 8 (40 steps wasted without judgment)
  - Task 4: loop trap from step 20 (stuck in repetitive tool calls)

Compares an agent WITHOUT judgment (blindly runs max steps on failure)
vs an agent WITH judgment (escalates early on detected failures).

Outputs: steps saved, tokens saved, cost saved (in $).
Assumes real API costs: $3/M input tokens, $15/M output tokens (GPT-4 class).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import DecisionEngine
import numpy as np

# ---------------------------------------------------------------------------
# Realistic cost model
# ---------------------------------------------------------------------------
INPUT_TOKENS_PER_STEP = 2000   # system + history + tool output
OUTPUT_TOKENS_PER_STEP = 500    # LLM's response + tool call
COST_PER_1M_INPUT = 3.0         # GPT-4 / Claude Opus pricing
COST_PER_1M_OUTPUT = 15.0

STEP_INPUT_COST = INPUT_TOKENS_PER_STEP * COST_PER_1M_INPUT / 1_000_000
STEP_OUTPUT_COST = OUTPUT_TOKENS_PER_STEP * COST_PER_1M_OUTPUT / 1_000_000
COST_PER_STEP = STEP_INPUT_COST + STEP_OUTPUT_COST

# ---------------------------------------------------------------------------
# Simulated agent tasks — one obs dict per step
# ---------------------------------------------------------------------------
def task_normal(steps=25):
    """Healthy task — completes smoothly."""
    for i in range(steps):
        yield {"tool_ok": True, "progress_delta": 1.0/steps + 0.02, "error_count_delta": 0}

def task_silent_degradation(steps=50):
    """Context drift — tool calls are 'successful' but progress decays."""
    for i in range(steps):
        if i < 15:
            yield {"tool_ok": True, "progress_delta": 0.06, "error_count_delta": 0}
        else:
            decay = 0.02 * (i - 14)
            prog = max(0.0, 0.05 - decay)
            yield {"tool_ok": True, "progress_delta": prog, "error_count_delta": 0}

def task_catastrophic(steps=50):
    """Catastrophic failure at step 8."""
    for i in range(steps):
        if i < 8:
            yield {"tool_ok": True, "progress_delta": 0.10, "error_count_delta": 0}
        elif i == 8:
            yield {"tool_ok": False, "progress_delta": -0.30, "error_count_delta": 3}
        else:
            yield {"tool_ok": False, "progress_delta": -0.02, "error_count_delta": 1}

def task_loop_trap(steps=50):
    """Loop trap — stuck from step 20."""
    for i in range(steps):
        if i < 20:
            yield {"tool_ok": True, "progress_delta": 0.04, "error_count_delta": 0}
        else:
            # Stuck — same tool, zero progress
            yield {"tool_ok": True, "progress_delta": 0.0, "error_count_delta": 0}


TASKS = [
    ("Normal completion", task_normal(25)),
    ("Silent degradation", task_silent_degradation(50)),
    ("Catastrophic failure", task_catastrophic(50)),
    ("Loop trap", task_loop_trap(50)),
]

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
def run_without_judgment(task_obs, max_steps):
    """Baseline: agent runs every step blindly."""
    steps_run = 0
    progress = 0.0
    for obs in task_obs:
        steps_run += 1
        progress += obs["progress_delta"]
        if progress >= 0.90:
            break
        if steps_run >= max_steps:
            break
    completed = progress >= 0.90
    return steps_run, completed, progress

def run_with_judgment(task_obs, max_steps, engine):
    """Agent with judgment: escalates on detection."""
    engine.reset()
    steps_run = 0
    progress = 0.0
    escd = False
    for obs in task_obs:
        steps_run += 1
        progress += obs["progress_delta"]
        d = engine.step(obs)
        if d.action == "escalate" and not escd:
            escd = True
            break
        if progress >= 0.90:
            break
        if steps_run >= max_steps:
            break
    completed = progress >= 0.90
    return steps_run, completed, progress, escd

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    engine = DecisionEngine(seed=42)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("COST SAVINGS ESTIMATE: Agent With vs Without Judgment")
    print(f"Cost model: ${COST_PER_STEP:.4f}/step "
          f"({INPUT_TOKENS_PER_STEP} in + {OUTPUT_TOKENS_PER_STEP} out tokens)")
    print("=" * 70)

    total_baseline_steps = 0
    total_baseline_cost = 0.0
    total_judgment_steps = 0
    total_judgment_cost = 0.0
    total_baseline_wasted = 0
    total_judgment_wasted = 0

    for task_name, task_obs in TASKS:
        obs_list = list(task_obs)
        max_s = len(obs_list)

        # WITHOUT judgment
        b_steps, b_ok, b_prog = run_without_judgment(obs_list, max_s)
        b_wasted = 0 if b_ok else b_steps
        b_cost = b_steps * COST_PER_STEP

        # WITH judgment
        j_steps, j_ok, j_prog, j_escd = run_with_judgment(obs_list, max_s, engine)
        j_wasted = 0 if j_ok else j_steps
        j_cost = j_steps * COST_PER_STEP

        total_baseline_steps += b_steps
        total_baseline_cost += b_cost
        total_baseline_wasted += b_wasted
        total_judgment_steps += j_steps
        total_judgment_cost += j_cost
        total_judgment_wasted += j_wasted

        print(f"\n  {task_name}:")
        print(f"    No judgment:   {b_steps:3d} steps ({b_wasted} wasted), ${b_cost:.2f} {'OK' if b_ok else 'FAIL'}")
        print(f"    With judgment: {j_steps:3d} steps ({j_wasted} wasted), ${j_cost:.2f} {'OK' if j_ok else 'FAIL'}")
        if j_escd:
            print(f"                   escalated early — saved {b_steps - j_steps} steps (${(b_steps - j_steps)*COST_PER_STEP:.2f})")
        if b_wasted > j_wasted:
            print(f"                   waste saved: {b_wasted - j_wasted} steps (${(b_wasted - j_wasted)*COST_PER_STEP:.2f})")

    print()
    print("=" * 70)
    print(f"  TOTAL without judgment: {total_baseline_steps} steps, ${total_baseline_cost:.2f}, "
          f"{total_baseline_wasted} wasted")
    print(f"  TOTAL with judgment:    {total_judgment_steps} steps, ${total_judgment_cost:.2f}, "
          f"{total_judgment_wasted} wasted")
    steps_saved = total_baseline_steps - total_judgment_steps
    cost_saved = total_baseline_cost - total_judgment_cost
    waste_reduction = (total_baseline_wasted - total_judgment_wasted) / max(total_baseline_wasted, 1) * 100
    print(f"  SAVED: {steps_saved} steps, ${cost_saved:.2f}, "
          f"{waste_reduction:.0f}% waste reduction")

    # Extrapolate to 1000 tasks/month
    print()
    print(f"  If running 1000 similar tasks/month:")
    monthly_savings = cost_saved / len(TASKS) * 1000
    print(f"    ~${monthly_savings:.0f}/month saved on token costs alone")
    print(f"    ~{steps_saved / len(TASKS) * 1000:.0f} unnecessary steps avoided")


if __name__ == "__main__":
    main()
