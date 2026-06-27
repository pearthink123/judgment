#!/usr/bin/env python3
"""
Evaluation Runner — head-to-head comparison: baseline vs judgment.

Runs identical synthetic trajectories through two Agent loops:
  1. BASELINE  — pure ReAct, no oversight (always continues)
  2. JUDGMENT  — ReAct + DecisionEngine (can escalate/correct)

Metrics:
  - Task success rate
  - Wasted steps (failure trajectories only)
  - Detection precision / recall
  - Mean detection delay
  - False escalation rate

Usage:
    python scripts/eval_runner.py
    python scripts/eval_runner.py --trajectories-per-model 30 --max-steps 40
    python scripts/eval_runner.py --output eval_report.json
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import (
    DecisionEngine,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)

from scripts.fault_models import FAULT_MODELS


# ---------------------------------------------------------------------------
# Run outcome
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    """What happened in one trajectory run."""
    model_name: str
    seed: int
    max_steps: int

    # Completion
    steps_run: int
    progress: float
    completed: bool         # progress >= 0.90
    escalated: bool
    escalated_step: Optional[int]

    # Anomaly detection (judgment only)
    alarms_fired: int
    first_alarm_step: Optional[int]
    faulty: bool            # this trajectory had injected faults
    fault_injected_step: Optional[int]

    # Actions
    actions: List[str]
    action_counts: Dict[str, int]


# ---------------------------------------------------------------------------
# Run a single trajectory through the JUDGMENT loop
# ---------------------------------------------------------------------------
def run_with_judgment(
    engine: DecisionEngine,
    observations: List[Dict[str, Any]],
    max_steps: int,
    faulty: bool,
    fault_step: Optional[int],
) -> RunOutcome:
    engine.reset()

    progress_total = 0.0
    actions: List[str] = []
    alarms = 0
    first_alarm: Optional[int] = None
    escalated = False
    escalated_step: Optional[int] = None
    completed = False
    steps_run = 0

    for step_i, obs in enumerate(observations):
        decision = engine.step(obs)
        actions.append(decision.action)

        if decision.anomaly:
            alarms += 1
            if first_alarm is None:
                first_alarm = step_i + 1

        if decision.action == ACTION_ESCALATE and not escalated:
            escalated = True
            escalated_step = step_i + 1

        progress_total += obs.get("progress_delta", 0.0)

        if progress_total >= 0.90:
            completed = True
            steps_run = step_i + 1
            break

        if escalated:
            steps_run = step_i + 1
            break
    else:
        steps_run = max_steps

    action_counts: Dict[str, int] = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    return RunOutcome(
        model_name="judgment",
        seed=0,
        max_steps=max_steps,
        steps_run=steps_run,
        progress=round(progress_total, 3),
        completed=completed,
        escalated=escalated,
        escalated_step=escalated_step,
        alarms_fired=alarms,
        first_alarm_step=first_alarm,
        faulty=faulty,
        fault_injected_step=fault_step,
        actions=actions,
        action_counts=action_counts,
    )


# ---------------------------------------------------------------------------
# Run a single trajectory through the BASELINE loop (no judgment)
# ---------------------------------------------------------------------------
def run_baseline(
    observations: List[Dict[str, Any]],
    max_steps: int,
    faulty: bool,
    fault_step: Optional[int],
) -> RunOutcome:
    progress_total = 0.0
    completed = False
    steps_run = 0

    for step_i, obs in enumerate(observations):
        steps_run = step_i + 1
        progress_total += obs.get("progress_delta", 0.0)

        if progress_total >= 0.90:
            completed = True
            break

    return RunOutcome(
        model_name="baseline",
        seed=0,
        max_steps=max_steps,
        steps_run=steps_run,
        progress=round(progress_total, 3),
        completed=completed,
        escalated=False,
        escalated_step=None,
        alarms_fired=0,
        first_alarm_step=None,
        faulty=faulty,
        fault_injected_step=fault_step,
        actions=["continue"] * steps_run,
        action_counts={"continue": steps_run},
    )


# ---------------------------------------------------------------------------
# Generate a trajectory from a fault model
# ---------------------------------------------------------------------------
def generate_trajectory(
    fault_model: str,
    max_steps: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Generate observations and return (observations, fault_injection_step)."""
    generator = FAULT_MODELS[fault_model]

    obs_list: List[Dict[str, Any]] = []
    fault_step: Optional[int] = None
    progress = 0.0

    for step_i in range(max_steps):
        obs = generator(step_i + 1, rng)
        obs_list.append(obs)
        progress += obs.get("progress_delta", 0.0)

        # Detect when fault is first injected
        if fault_model != "healthy" and fault_step is None:
            # Heuristic: first step where tool_ok=False or progress_delta <= 0
            if not obs["tool_ok"] or obs["progress_delta"] <= 0.0:
                fault_step = step_i + 1

        if progress >= 0.90:
            break

    return obs_list, fault_step


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
@dataclass
class EvalReport:
    model_name: str         # "baseline" or "judgment"
    n_trajectories: int

    success_rate: float
    mean_steps_success: float     # steps to complete (successful only)
    mean_steps_failure: float     # steps wasted (failed only)
    waste_ratio: float            # mean(steps/max_steps) for failed trajectories

    # Detection (judgment only — baseline has no detection)
    detection_precision: float    # true_alarms / total_alarms
    detection_recall: float       # % faulty trajectories with >=1 alarm
    mean_detection_delay: float   # steps from fault to first alarm

    false_escalation_rate: float  # % healthy trajectories escalated
    mean_progress: float          # mean final progress across all trajectories

    action_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "n_trajectories": self.n_trajectories,
            "success_rate": round(self.success_rate, 4),
            "mean_steps_success": round(self.mean_steps_success, 1),
            "mean_steps_failure": round(self.mean_steps_failure, 1),
            "waste_ratio": round(self.waste_ratio, 4),
            "detection_precision": round(self.detection_precision, 4),
            "detection_recall": round(self.detection_recall, 4),
            "mean_detection_delay": round(self.mean_detection_delay, 1),
            "false_escalation_rate": round(self.false_escalation_rate, 4),
            "mean_progress": round(self.mean_progress, 4),
            "action_distribution": self.action_distribution,
        }


def compute_report(outcomes: List[RunOutcome], model_name: str) -> EvalReport:
    n = len(outcomes)

    # Success
    successful = [o for o in outcomes if o.completed]
    failed = [o for o in outcomes if not o.completed]
    success_rate = len(successful) / n

    mean_steps_success = (
        np.mean([o.steps_run for o in successful]) if successful else 0.0
    )
    mean_steps_failure = (
        np.mean([o.steps_run for o in failed]) if failed else 0.0
    )

    # Waste ratio: for failed trajectories, steps_run / max_steps
    waste_ratios = [
        o.steps_run / o.max_steps for o in failed
    ]
    mean_waste = np.mean(waste_ratios) if waste_ratios else 0.0

    # Detection
    faulty = [o for o in outcomes if o.faulty]
    healthy = [o for o in outcomes if not o.faulty]

    total_alarms = sum(o.alarms_fired for o in outcomes)
    true_alarms = sum(o.alarms_fired for o in faulty)
    precision = true_alarms / max(total_alarms, 1)

    detected = sum(1 for o in faulty if o.first_alarm_step is not None)
    recall = detected / max(len(faulty), 1)

    delays = []
    for o in faulty:
        if o.first_alarm_step is not None and o.fault_injected_step is not None:
            delay = o.first_alarm_step - o.fault_injected_step
            delays.append(max(0, delay))
    mean_delay = np.mean(delays) if delays else float("nan")

    # False escalation
    false_esc = sum(1 for o in healthy if o.escalated)
    false_esc_rate = false_esc / max(len(healthy), 1)

    # Progress
    mean_progress = np.mean([o.progress for o in outcomes])

    # Action distribution
    action_dist: Dict[str, int] = {}
    for o in outcomes:
        for a, c in o.action_counts.items():
            action_dist[a] = action_dist.get(a, 0) + c

    return EvalReport(
        model_name=model_name,
        n_trajectories=n,
        success_rate=success_rate,
        mean_steps_success=mean_steps_success,
        mean_steps_failure=mean_steps_failure,
        waste_ratio=mean_waste,
        detection_precision=precision,
        detection_recall=recall,
        mean_detection_delay=mean_delay,
        false_escalation_rate=false_esc_rate,
        mean_progress=mean_progress,
        action_distribution=action_dist,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: baseline vs judgment."
    )
    parser.add_argument(
        "--trajectories-per-model", "-n", type=int, default=25,
        help="Trajectories per fault model (default 25)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per trajectory (default 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--pomcp", action="store_true",
        help="Use POMCP solver (default: grid POMDP)",
    )
    args = parser.parse_args()

    models = ["healthy", "context_drift", "tool_degradation", "loop_trap", "catastrophic_cascade"]
    n_per = args.trajectories_per_model
    max_steps = args.max_steps
    total = n_per * len(models)

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / "eval_results.json"
    )

    print("=" * 70)
    print("HEAD-TO-HEAD EVALUATION: Baseline vs Judgment")
    print(f"  {len(models)} fault models × {n_per} trajectories = {total} each")
    print(f"  Total: {total * 2} runs")
    print("=" * 70)

    # ---- Shared engine (for judgment) ----
    engine = DecisionEngine(
        use_pomdp=not args.pomcp,
        use_pomcp=args.pomcp,
        pomcp_n_simulations=500,
        seed=args.seed,
    )

    base_rng = np.random.default_rng(args.seed)

    baseline_outcomes: List[RunOutcome] = []
    judgment_outcomes: List[RunOutcome] = []

    t_start = time.time()

    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")

        model_baseline: List[RunOutcome] = []
        model_judgment: List[RunOutcome] = []

        for traj_i in range(n_per):
            seed_i = args.seed * 1000 + traj_i
            rng = np.random.default_rng(seed_i)
            obs_list, fault_step = generate_trajectory(model_name, max_steps, rng)

            faulty = model_name != "healthy"
            actual_steps = len(obs_list)

            # Baseline
            b_outcome = run_baseline(obs_list, actual_steps, faulty, fault_step)
            # Judgment
            j_outcome = run_with_judgment(engine, obs_list, actual_steps, faulty, fault_step)

            model_baseline.append(b_outcome)
            model_judgment.append(j_outcome)

        # Per-model summary
        b_rep = compute_report(model_baseline, "baseline")
        j_rep = compute_report(model_judgment, "judgment")

        baseline_outcomes.extend(model_baseline)
        judgment_outcomes.extend(model_judgment)

        waste_delta = b_rep.waste_ratio - j_rep.waste_ratio
        print(
            f"  Baseline: sr={b_rep.success_rate:.2f}  "
            f"waste={b_rep.waste_ratio:.2f}  "
            f"prog={b_rep.mean_progress:.2f}"
        )
        print(
            f"  Judgment: sr={j_rep.success_rate:.2f}  "
            f"waste={j_rep.waste_ratio:.2f}  "
            f"prog={j_rep.mean_progress:.2f}  "
            f"recall={j_rep.detection_recall:.2f}  "
            f"delay={j_rep.mean_detection_delay:.1f}s"
        )
        print(
            f"  Delta:    waste={waste_delta:+.2f}  "
            f"(judgment wastes {'LESS' if waste_delta > 0 else 'MORE'} steps on failure)"
        )

    duration = time.time() - t_start

    # ---- Aggregate reports ----
    baseline_report = compute_report(baseline_outcomes, "baseline")
    judgment_report = compute_report(judgment_outcomes, "judgment")

    # ---- Final output ----
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>14} {'Judgment':>14} {'Delta':>10}")
    print("-" * 73)
    print(f"{'Success rate':<35} {baseline_report.success_rate:>14.4f} {judgment_report.success_rate:>14.4f} {judgment_report.success_rate - baseline_report.success_rate:>+10.4f}")
    print(f"{'Mean progress':<35} {baseline_report.mean_progress:>14.4f} {judgment_report.mean_progress:>14.4f} {judgment_report.mean_progress - baseline_report.mean_progress:>+10.4f}")
    print(f"{'Waste ratio (failure)':<35} {baseline_report.waste_ratio:>14.4f} {judgment_report.waste_ratio:>14.4f} {baseline_report.waste_ratio - judgment_report.waste_ratio:>+10.4f}")
    print(f"{'Mean steps (success)':<35} {baseline_report.mean_steps_success:>14.1f} {judgment_report.mean_steps_success:>14.1f} {'—':>10}")
    print(f"{'Detection precision':<35} {'—':>14} {judgment_report.detection_precision:>14.4f} {'—':>10}")
    print(f"{'Detection recall':<35} {'—':>14} {judgment_report.detection_recall:>14.4f} {'—':>10}")
    print(f"{'Mean detection delay':<35} {'—':>14} {judgment_report.mean_detection_delay:>14.1f}s {'—':>10}")
    print(f"{'False escalation rate':<35} {'—':>14} {judgment_report.false_escalation_rate:>14.4f} {'—':>10}")
    print(f"{'Duration':<35} {duration:>14.2f}s {'—':>14} {'—':>10}")
    print()
    print(f"Judgment actions: {judgment_report.action_distribution}")
    print(f"Baseline actions:  {baseline_report.action_distribution}")

    # Save
    output_path.write_text(json.dumps({
        "baseline": baseline_report.to_dict(),
        "judgment": judgment_report.to_dict(),
        "config": {
            "trajectories_per_model": n_per,
            "max_steps": max_steps,
            "seed": args.seed,
            "solver": "pomcp" if args.pomcp else "grid",
            "duration_seconds": round(duration, 2),
        },
    }, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
