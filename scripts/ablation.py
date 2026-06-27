#!/usr/bin/env python3
"""
Ablation study — measures the marginal contribution of each component.

Runs the same synthetic trajectories through 5 configurations:

  Config A: HMM + threshold gate (no CUSUM, no POMDP) — simplest baseline
  Config B: HMM + CUSUM + threshold gate (anomaly nudges belief)
  Config C: HMM + CUSUM + POMDP grid (current default minus corrective)
  Config D: HMM + CUSUM + FastPOMCP (scalable online MCTS)
  Config E: Full stack (D + content signals + corrective router)

Metrics per config:
  - Detection recall (% faulty trajectories with alarm)
  - False-positive rate (% healthy steps with alarm)
  - Waste ratio (mean steps/max on failed trajectories)
  - Mean steps to complete (successful only)
  - Escalation accuracy (% escalations that hit faulty trajectories)

Usage:
    python scripts/ablation.py --trajectories-per-model 20 --max-steps 30
"""

import sys, json, time, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import (
    DecisionEngine,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)
from scripts.fault_models import FAULT_MODELS, healthy_generator


# ---------------------------------------------------------------------------
# Config builders — each returns a DecisionEngine with specific components
# ---------------------------------------------------------------------------
def build_config_a(seed: int) -> DecisionEngine:
    """HMM + threshold gate only (no CUSUM, no POMDP)."""
    e = DecisionEngine(use_pomdp=False, use_fast_pomcp=False, use_corrective=False, seed=seed)
    # Disable CUSUM by setting threshold absurdly high
    e.cusum.h = 1e9
    e.cusum.S = 0.0
    return e


def build_config_b(seed: int) -> DecisionEngine:
    """HMM + CUSUM + threshold gate."""
    e = DecisionEngine(use_pomdp=False, use_fast_pomcp=False, use_corrective=False, seed=seed)
    return e


def build_config_c(seed: int) -> DecisionEngine:
    """HMM + CUSUM + grid POMDP (no corrective)."""
    e = DecisionEngine(use_pomdp=True, use_fast_pomcp=False, use_corrective=False, seed=seed)
    return e


def build_config_d(seed: int) -> DecisionEngine:
    """HMM + CUSUM + FastPOMCP (no corrective)."""
    e = DecisionEngine(use_fast_pomcp=True, use_corrective=False, seed=seed)
    return e


def build_config_e(seed: int) -> DecisionEngine:
    """Full stack (FastPOMCP + corrective + content signals)."""
    return DecisionEngine(
        use_fast_pomcp=True, use_corrective=True,
        use_content_signals=True, seed=seed,
    )


CONFIGS = {
    "A: HMM+threshold": build_config_a,
    "B: +CUSUM": build_config_b,
    "C: +POMDP(grid)": build_config_c,
    "D: +FastPOMCP": build_config_d,
    "E: full stack": build_config_e,
}


# ---------------------------------------------------------------------------
# Single trajectory evaluation
# ---------------------------------------------------------------------------
@dataclass
class AblationResult:
    config_name: str
    n_trajectories: int
    success_rate: float
    waste_ratio: float                 # for failed trajectories
    detection_recall: float            # % faulty with >=1 alarm
    detection_precision: float         # true alarms / total alarms
    mean_detection_delay: float
    false_escalation_rate: float       # % healthy trajectories escalated
    mean_steps: float                  # across all trajectories
    action_distribution: Dict[str, int]


def evaluate_config(
    engine_builder,
    config_name: str,
    fault_modes: List[str],
    n_per_mode: int,
    max_steps: int,
    base_seed: int,
) -> AblationResult:
    rng = np.random.default_rng(base_seed)
    all_outcomes: List[Dict[str, Any]] = []

    for model_name in fault_modes:
        for traj_i in range(n_per_mode):
            seed_i = abs(hash(f"{config_name}_{model_name}_{traj_i}")) % (2**31 - 1)
            rng2 = np.random.default_rng(seed_i)

            generator = FAULT_MODELS[model_name]
            faulty = model_name != "healthy"

            engine = engine_builder(seed_i % 1000)

            alarms_fired = 0
            first_alarm: Optional[int] = None
            fault_step: Optional[int] = None
            escalated = False
            escalated_step: Optional[int] = None
            completed = False
            progress = 0.0
            steps_run = 0
            actions: List[str] = []

            for step_i in range(max_steps):
                obs = generator(step_i + 1, rng2)
                progress += obs.get("progress_delta", 0.0)

                # Track fault injection moment
                if faulty and fault_step is None:
                    if not obs["tool_ok"] or obs["progress_delta"] <= 0.0:
                        fault_step = step_i + 1

                # Inject llm_text for config E
                if "content_signals" in str(type(engine)):
                    obs.setdefault("llm_text", obs.get("llm_text", "Step complete."))

                decision = engine.step(obs)
                actions.append(decision.action)
                steps_run = step_i + 1

                if decision.anomaly:
                    alarms_fired += 1
                    if first_alarm is None:
                        first_alarm = step_i + 1

                if decision.action == ACTION_ESCALATE and not escalated:
                    escalated = True
                    escalated_step = step_i + 1

                if progress >= 0.90:
                    completed = True
                    break
                if escalated:
                    break

            all_outcomes.append({
                "faulty": faulty,
                "completed": completed,
                "steps_run": steps_run,
                "max_steps": max_steps,
                "alarms_fired": alarms_fired,
                "first_alarm_step": first_alarm,
                "fault_step": fault_step,
                "escalated": escalated,
                "escalated_step": escalated_step,
                "actions": actions,
            })

    # ---- Aggregate ----
    n = len(all_outcomes)
    completed_list = [o for o in all_outcomes if o["completed"]]
    failed_list = [o for o in all_outcomes if not o["completed"]]
    faulty_list = [o for o in all_outcomes if o["faulty"]]
    healthy_list = [o for o in all_outcomes if not o["faulty"]]

    success_rate = len(completed_list) / n

    waste_ratios = [o["steps_run"] / o["max_steps"] for o in failed_list]
    waste = np.mean(waste_ratios) if waste_ratios else 0.0

    detected = sum(1 for o in faulty_list if o["first_alarm_step"] is not None)
    recall = detected / max(len(faulty_list), 1)

    total_alarms = sum(o["alarms_fired"] for o in all_outcomes)
    true_alarms = sum(o["alarms_fired"] for o in faulty_list)
    precision = true_alarms / max(total_alarms, 1)

    delays = []
    for o in faulty_list:
        if o["first_alarm_step"] is not None and o["fault_step"] is not None:
            delays.append(max(0, o["first_alarm_step"] - o["fault_step"]))
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    false_esc = sum(1 for o in healthy_list if o["escalated"])
    false_esc_rate = false_esc / max(len(healthy_list), 1)

    mean_steps = float(np.mean([o["steps_run"] for o in all_outcomes]))

    action_dist: Dict[str, int] = {}
    for o in all_outcomes:
        for a in o["actions"]:
            action_dist[a] = action_dist.get(a, 0) + 1

    return AblationResult(
        config_name=config_name,
        n_trajectories=n,
        success_rate=success_rate,
        waste_ratio=waste,
        detection_recall=recall,
        detection_precision=precision,
        mean_detection_delay=mean_delay,
        false_escalation_rate=false_esc_rate,
        mean_steps=mean_steps,
        action_distribution=action_dist,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation study — marginal component value")
    parser.add_argument("--trajectories-per-model", "-n", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", "-r", type=int, default=1,
                        help="Repeat N times with different seeds for confidence intervals")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    fault_modes = ["healthy", "context_drift", "tool_degradation", "loop_trap", "catastrophic_cascade"]

    output_path = Path(args.output) if args.output else (Path(__file__).parent / "ablation_results.json")

    print("=" * 80)
    print(f"ABLATION STUDY: {len(CONFIGS)} configs × {len(fault_modes)} models × {args.trajectories_per_model} traj")
    print("=" * 80)

    t0 = time.time()
    all_seed_results: Dict[str, List[AblationResult]] = {name: [] for name in CONFIGS}

    for repeat_i in range(args.repeat):
        rep_seed = args.seed + repeat_i * 10000
        if args.repeat > 1:
            print(f"\n  -- repeat {repeat_i+1}/{args.repeat} (seed={rep_seed}) --")

        for config_name, builder in CONFIGS.items():
            if args.repeat > 1:
                print(f"    {config_name}...", end=" ", flush=True)
            else:
                print(f"\n  {config_name}...", end=" ", flush=True)
            tc = time.time()
            result = evaluate_config(
                builder, config_name, fault_modes,
                args.trajectories_per_model, args.max_steps, rep_seed,
            )
            elapsed_c = time.time() - tc
            all_seed_results[config_name].append(result)
            print(f"recall={result.detection_recall:.2f}  "
                  f"waste={result.waste_ratio:.2f}  "
                  f"({elapsed_c:.1f}s)")

    # Average across repeats
    results: List[AblationResult] = []
    for config_name in CONFIGS:
        runs = all_seed_results[config_name]
        # Average the key metrics
        avg = AblationResult(
            config_name=config_name,
            n_trajectories=runs[0].n_trajectories,
            success_rate=float(np.mean([r.success_rate for r in runs])),
            waste_ratio=float(np.mean([r.waste_ratio for r in runs])),
            detection_recall=float(np.mean([r.detection_recall for r in runs])),
            detection_precision=float(np.mean([r.detection_precision for r in runs])),
            mean_detection_delay=float(np.mean([r.mean_detection_delay for r in runs])),
            false_escalation_rate=float(np.mean([r.false_escalation_rate for r in runs])),
            mean_steps=float(np.mean([r.mean_steps for r in runs])),
            action_distribution=runs[0].action_distribution,
        )
        results.append(avg)

    total_time = time.time() - t0

    # ---- Table output ----
    print()
    print("=" * 80)
    print(f"{'Config':<25} {'Recall':>8} {'FPR':>8} {'Waste':>8} {'Delay':>8} {'FalseEsc':>10}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.config_name:<25} "
            f"{r.detection_recall:>8.4f} "
            f"{r.detection_precision:>8.4f} "
            f"{r.waste_ratio:>8.4f} "
            f"{r.mean_detection_delay:>8.1f} "
            f"{r.false_escalation_rate:>10.4f}"
        )

    print("-" * 80)
    # Show marginal improvements
    if len(results) >= 2:
        base = results[0]
        print()
        print("=== Marginal contribution (vs HMM+threshold baseline) ===")
        for r in results[1:]:
            delta_waste = base.waste_ratio - r.waste_ratio
            delta_recall = r.detection_recall - base.detection_recall
            delta_fpr = r.detection_precision - base.detection_precision
            print(
                f"  {r.config_name}: "
                f"recall {delta_recall:+.2f}  "
                f"waste {delta_waste:+.3f}  "
                f"precision {delta_fpr:+.2f}"
            )

    print(f"\nTotal time: {total_time:.1f}s")

    # Save
    output_path.write_text(json.dumps(
        [{k: round(v, 4) if isinstance(v, float) else v
          for k, v in r.__dict__.items()}
         for r in results],
        indent=2,
    ), encoding="utf-8")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
