#!/usr/bin/env python3
"""
Benchmark Suite — quantitative evaluation of the DecisionEngine.

Runs N synthetic trajectories with known fault-injection points and
measures:

  - Detection delay     : steps from fault injection to first alarm
  - Detection rate      : % of faulty trajectories with ≥1 alarm
  - False-positive rate : % of healthy steps misclassified as anomaly
  - Escalation accuracy : % of broken trajectories that trigger escalate

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --trajectories 200 --max-steps 30 --seed 0

Output:
    console summary + scripts/benchmark_results.json
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
from core.pomcp import POMCPPlanner


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------
def generate_trajectory(
    rng: np.random.Generator,
    max_steps: int = 30,
    fault_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Generate one synthetic agent trajectory.

    Parameters
    ----------
    rng : seeded RNG
    max_steps : int
    fault_mode : str or None
        None        → healthy trajectory (no injected faults)
        "early"     → faults start at step 3–5
        "mid"       → faults start at step 8–12
        "late"      → faults start at step 15–20
        "recovery"  → faults at steps 6–10, then recovery
        "intermittent" → random faults ~30% of steps

    Returns
    -------
    (observations, fault_injection_step)
      fault_injection_step is None for healthy trajectories.
    """
    obs_list: List[Dict[str, Any]] = []
    fault_step: Optional[int] = None
    error_streak = 0

    # Determine fault window
    if fault_mode == "early":
        fault_start = int(rng.integers(3, 6))
    elif fault_mode == "mid":
        fault_start = int(rng.integers(8, 13))
    elif fault_mode == "late":
        fault_start = int(rng.integers(15, 21))
    elif fault_mode == "recovery":
        fault_start = 6
    elif fault_mode == "intermittent":
        fault_start = -1  # probabilistic per step
    else:
        fault_start = -1  # healthy

    faulty = fault_start > 0 or fault_mode == "intermittent"

    for step_i in range(max_steps):
        if fault_mode == "recovery" and 6 <= step_i <= 10:
            tool_ok = False
            progress_delta = -0.04
            error_count_delta = 1
            if fault_step is None:
                fault_step = step_i
        elif fault_mode == "intermittent" and rng.random() < 0.30:
            tool_ok = False
            progress_delta = -0.03
            error_count_delta = 1
            if fault_step is None:
                fault_step = step_i
        elif faulty and step_i >= fault_start:
            tool_ok = rng.random() > 0.80  # ~80% failure rate
            progress_delta = -0.05 if not tool_ok else 0.02
            error_count_delta = 0 if tool_ok else 1
            if fault_step is None:
                fault_step = step_i
        else:
            tool_ok = rng.random() > 0.10  # ~10% base failure rate
            progress_delta = 0.12 + 0.05 * rng.random() if tool_ok else -0.02
            error_count_delta = 0 if tool_ok else 1

        if not tool_ok:
            error_streak += 1
        else:
            error_streak = max(0, error_streak - 1)

        obs_list.append({
            "tool_ok": tool_ok,
            "progress_delta": progress_delta,
            "has_user_msg": False,
            "error_count_delta": error_count_delta,
        })

    return obs_list, fault_step


# ---------------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryResult:
    """Per-trajectory benchmark metrics."""

    fault_mode: str
    fault_step: Optional[int]         # None = healthy
    steps: int
    first_alarm_step: Optional[int]   # first CUSUM alarm
    escalated: bool
    escalated_step: Optional[int]
    actions: List[str]
    alarms_total: int
    engine: str                       # "grid" or "pomcp"


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    engine_name: str
    n_trajectories: int
    duration_seconds: float

    # Detection metrics
    healthy_trajectories: int
    faulty_trajectories: int
    false_positive_rate: float        # % healthy steps with alarm
    detection_rate: float             # % faulty trajectories with ≥1 alarm
    median_detection_delay: float     # steps from fault to first alarm
    mean_detection_delay: float

    # Escalation metrics
    escalation_rate: float            # % faulty trajectories escalated
    healthy_escalation_rate: float    # % healthy trajectories incorrectly escalated

    # Action distribution
    action_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "n_trajectories": self.n_trajectories,
            "duration_seconds": self.duration_seconds,
            "healthy_trajectories": self.healthy_trajectories,
            "faulty_trajectories": self.faulty_trajectories,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "detection_rate": round(self.detection_rate, 4),
            "median_detection_delay": round(self.median_detection_delay, 1),
            "mean_detection_delay": round(self.mean_detection_delay, 1),
            "escalation_rate": round(self.escalation_rate, 4),
            "healthy_escalation_rate": round(self.healthy_escalation_rate, 4),
            "action_distribution": self.action_counts,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    engine: DecisionEngine,
    n_trajectories: int = 100,
    max_steps: int = 30,
    seed: int = 0,
) -> BenchmarkReport:
    """
    Run N synthetic trajectories and compute detection metrics.
    """
    rng = np.random.default_rng(seed)
    results: List[TrajectoryResult] = []

    # Fault modality distribution: 30% healthy, 70% faulty (across 5 modes)
    fault_modes = [None] * 30 + ["early"] * 14 + ["mid"] * 14 + \
                  ["late"] * 14 + ["recovery"] * 14 + ["intermittent"] * 14
    rng.shuffle(fault_modes)

    t_start = time.time()

    for traj_idx in range(n_trajectories):
        fault_mode = fault_modes[traj_idx]
        obs_list, fault_step = generate_trajectory(rng, max_steps, fault_mode)

        engine.reset()

        actions: List[str] = []
        first_alarm: Optional[int] = None
        alarms_total = 0
        escalated = False
        escalated_step: Optional[int] = None

        for step_i, obs in enumerate(obs_list):
            decision = engine.step(obs)
            actions.append(decision.action)

            if decision.anomaly and first_alarm is None:
                first_alarm = step_i + 1

            if decision.anomaly:
                alarms_total += 1

            if decision.action == ACTION_ESCALATE and not escalated:
                escalated = True
                escalated_step = step_i + 1

        results.append(TrajectoryResult(
            fault_mode=fault_mode or "healthy",
            fault_step=fault_step,
            steps=max_steps,
            first_alarm_step=first_alarm,
            escalated=escalated,
            escalated_step=escalated_step,
            actions=actions,
            alarms_total=alarms_total,
            engine="pomcp" if engine._pomcp else "grid",
        ))

    duration = time.time() - t_start

    # --- Aggregate ---
    healthy_results = [r for r in results if r.fault_mode == "healthy"]
    faulty_results = [r for r in results if r.fault_mode != "healthy"]

    # False-positive rate: % of healthy steps marked as anomaly
    healthy_steps_total = sum(r.steps for r in healthy_results)
    healthy_alarms_total = sum(r.alarms_total for r in healthy_results)
    fpr = healthy_alarms_total / max(healthy_steps_total, 1)

    # Detection rate: % of faulty trajectories with ≥1 alarm
    detected = sum(1 for r in faulty_results if r.first_alarm_step is not None)
    detection_rate = detected / max(len(faulty_results), 1)

    # Detection delay: steps from fault injection to first alarm
    delays = []
    for r in faulty_results:
        if r.first_alarm_step is not None and r.fault_step is not None:
            delay = r.first_alarm_step - r.fault_step
            delays.append(max(0, delay))

    median_delay = float(np.median(delays)) if delays else float("nan")
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    # Escalation rate
    escalated_faulty = sum(1 for r in faulty_results if r.escalated)
    escalation_rate = escalated_faulty / max(len(faulty_results), 1)
    escalated_healthy = sum(1 for r in healthy_results if r.escalated)
    healthy_esc_rate = escalated_healthy / max(len(healthy_results), 1)

    # Action distribution
    action_counts: Dict[str, int] = {}
    for r in results:
        for a in r.actions:
            action_counts[a] = action_counts.get(a, 0) + 1

    return BenchmarkReport(
        engine_name="pomcp" if engine._pomcp else "grid",
        n_trajectories=n_trajectories,
        duration_seconds=round(duration, 2),
        healthy_trajectories=len(healthy_results),
        faulty_trajectories=len(faulty_results),
        false_positive_rate=fpr,
        detection_rate=detection_rate,
        median_detection_delay=median_delay,
        mean_detection_delay=mean_delay,
        escalation_rate=escalation_rate,
        healthy_escalation_rate=healthy_esc_rate,
        action_counts=action_counts,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the DecisionEngine on synthetic agent trajectories."
    )
    parser.add_argument("--trajectories", "-n", type=int, default=100,
                        help="Number of trajectories (default 100)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Steps per trajectory (default 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--pomcp", action="store_true",
                        help="Use POMCP solver instead of grid POMDP")
    parser.add_argument("--pomcp-sims", type=int, default=500,
                        help="POMCP simulations per decision (default 500)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON path (default: scripts/benchmark_results.json)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both grid and POMCP for comparison")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / "benchmark_results.json"
    )

    reports: List[BenchmarkReport] = []

    # --- Grid POMDP ---
    if not args.pomcp or args.compare:
        print("=" * 65)
        print("Benchmark: Grid POMDP (231-point exact value iteration)")
        print(f"  {args.trajectories} trajectories × {args.max_steps} steps")
        print("=" * 65)

        engine_grid = DecisionEngine(use_pomdp=True, use_pomcp=False, seed=args.seed)
        report_grid = run_benchmark(
            engine_grid,
            n_trajectories=args.trajectories,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        reports.append(report_grid)

        print(f"\n{'Metric':<35} {'Value':>15}")
        print("-" * 51)
        print(f"{'Healthy trajectories':<35} {report_grid.healthy_trajectories:>15}")
        print(f"{'Faulty trajectories':<35} {report_grid.faulty_trajectories:>15}")
        print(f"{'False-positive rate':<35} {report_grid.false_positive_rate:>15.4f}")
        print(f"{'Detection rate':<35} {report_grid.detection_rate:>15.4f}")
        print(f"{'Median detection delay':<35} {report_grid.median_detection_delay:>15.1f} steps")
        print(f"{'Mean detection delay':<35} {report_grid.mean_detection_delay:>15.1f} steps")
        print(f"{'Escalation rate (faulty)':<35} {report_grid.escalation_rate:>15.4f}")
        print(f"{'Escalation rate (healthy)':<35} {report_grid.healthy_escalation_rate:>15.4f}")
        print(f"{'Duration':<35} {report_grid.duration_seconds:>15.2f}s")
        print(f"\nAction distribution: {report_grid.action_counts}")

    # --- POMCP ---
    if args.pomcp or args.compare:
        print()
        print("=" * 65)
        print(f"Benchmark: POMCP ({args.pomcp_sims} sims, online MCTS)")
        print(f"  {args.trajectories} trajectories × {args.max_steps} steps")
        print("=" * 65)

        engine_pomcp = DecisionEngine(
            use_pomdp=False, use_pomcp=True,
            pomcp_n_simulations=args.pomcp_sims,
            seed=args.seed,
        )
        report_pomcp = run_benchmark(
            engine_pomcp,
            n_trajectories=args.trajectories,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        reports.append(report_pomcp)

        print(f"\n{'Metric':<35} {'Value':>15}")
        print("-" * 51)
        print(f"{'Healthy trajectories':<35} {report_pomcp.healthy_trajectories:>15}")
        print(f"{'Faulty trajectories':<35} {report_pomcp.faulty_trajectories:>15}")
        print(f"{'False-positive rate':<35} {report_pomcp.false_positive_rate:>15.4f}")
        print(f"{'Detection rate':<35} {report_pomcp.detection_rate:>15.4f}")
        print(f"{'Median detection delay':<35} {report_pomcp.median_detection_delay:>15.1f} steps")
        print(f"{'Mean detection delay':<35} {report_pomcp.mean_detection_delay:>15.1f} steps")
        print(f"{'Escalation rate (faulty)':<35} {report_pomcp.escalation_rate:>15.4f}")
        print(f"{'Escalation rate (healthy)':<35} {report_pomcp.healthy_escalation_rate:>15.4f}")
        print(f"{'Duration':<35} {report_pomcp.duration_seconds:>15.2f}s")
        print(f"\nAction distribution: {report_pomcp.action_counts}")

    # --- Comparison ---
    if args.compare and len(reports) == 2:
        g = reports[0]
        p = reports[1]
        print()
        print("=" * 65)
        print("COMPARISON: Grid vs POMCP")
        print("=" * 65)
        print(f"{'Metric':<30} {'Grid':>15} {'POMCP':>15}")
        print("-" * 60)
        print(f"{'Detection rate':<30} {g.detection_rate:>15.4f} {p.detection_rate:>15.4f}")
        print(f"{'False-positive rate':<30} {g.false_positive_rate:>15.4f} {p.false_positive_rate:>15.4f}")
        print(f"{'Median delay (steps)':<30} {g.median_detection_delay:>15.1f} {p.median_detection_delay:>15.1f}")
        print(f"{'Escalation rate':<30} {g.escalation_rate:>15.4f} {p.escalation_rate:>15.4f}")
        print(f"{'Duration (s)':<30} {g.duration_seconds:>15.2f} {p.duration_seconds:>15.2f}")

    # --- Save JSON ---
    output_path.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
