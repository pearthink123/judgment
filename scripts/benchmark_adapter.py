#!/usr/bin/env python3
"""
Real-world benchmark adapter — plug judgment into any benchmark data stream.

Converts benchmark trajectories (SWE-bench, GAIA, WebArena, custom) into
engine-compatible observation dicts via a user-defined extractor.

Usage:
    python scripts/benchmark_adapter.py --data ./my_traces.jsonl

Input format (JSONL):
    {"step": 1, "tool_ok": true, "output": "..."}
    {"step": 2, "tool_ok": false, "output": "Error: ..."}
    ...

Output:
    Per-trajectory: detection stats, escalation timing, token waste estimate
    Aggregate: recall, FPR, mean delay, savings in tokens
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import (
    DecisionEngine,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)


# ---------------------------------------------------------------------------
# Default extractor — maps a benchmark step to engine observation
# ---------------------------------------------------------------------------
def default_extractor(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort extraction from common benchmark trace formats.

    Tries these key names (in order):
      - tool_ok / tool_success / exit_code == 0
      - progress_delta / progress / score_delta
      - user_msg / human_intervention / interrupt
      - error_count_delta / errors / failures
      - llm_text / output / response / content
    """
    # tool_ok
    tool_ok = record.get("tool_ok")
    if tool_ok is None:
        tool_ok = record.get("tool_success")
    if tool_ok is None:
        ec = record.get("exit_code")
        tool_ok = (ec == 0) if ec is not None else True
    if tool_ok is None:
        # Guess from output content
        text = str(record.get("output", record.get("content", ""))).lower()
        tool_ok = not any(kw in text for kw in ["error", "fail", "exception", "traceback"])

    # progress_delta — try direct value, then infer from score
    progress = record.get("progress_delta", record.get("progress"))
    if progress is None:
        score = record.get("score", record.get("score_delta"))
        if score is not None:
            progress = float(score) * 0.1  # heuristic scaling
        else:
            progress = 0.05 if bool(tool_ok) else -0.02

    # user_msg
    user_msg = record.get("has_user_msg", record.get("user_msg"))
    if user_msg is None:
        user_msg = record.get("human_intervention", record.get("interrupt", False))

    # error_count_delta
    errors = record.get("error_count_delta", record.get("errors"))
    if errors is None:
        errors = record.get("failures", 0)

    # llm_text
    llm_text = record.get("llm_text", record.get("output"))
    if llm_text is None:
        llm_text = record.get("response", record.get("content"))

    return {
        "tool_ok": bool(tool_ok),
        "progress_delta": float(progress or 0.0),
        "has_user_msg": bool(user_msg),
        "error_count_delta": int(errors or 0),
        "llm_text": str(llm_text) if llm_text else None,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    n_trajectories: int
    n_steps_total: int
    detected_rate: float           # % faulty trajectories detected
    detection_delay_mean: float    # mean steps from first fault to alarm
    false_alarm_rate: float        # % healthy steps with alarm
    escalations: int
    false_escalations: int
    est_tokens_saved: int          # rough estimate
    per_trajectory: List[Dict[str, Any]]


def run_benchmark(
    trajectories: List[List[Dict[str, Any]]],
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[Callable] = None,
    token_cost_per_step: int = 5000,
    verbose: bool = False,
) -> BenchmarkReport:
    """
    Run judgment against real benchmark traces.

    Parameters
    ----------
    trajectories : list of trajectories, each a list of step dicts
    engine : DecisionEngine (created if None)
    extractor : (record) → observation dict (uses default if None)
    token_cost_per_step : int — rough token estimate per step (for savings calc)
    verbose : bool

    Returns
    -------
    BenchmarkReport
    """
    eng = engine or DecisionEngine()
    ext = extractor or default_extractor

    per_traj: List[Dict[str, Any]] = []
    total_steps = 0
    total_detected = 0
    total_faulty = 0
    delays: List[float] = []
    all_healthy_alarms = 0
    all_healthy_steps = 0
    total_escalations = 0
    total_false_escalations = 0
    total_tokens_saved = 0

    for traj_idx, traj in enumerate(trajectories):
        eng.reset()
        alarms = 0
        first_alarm: Optional[int] = None
        first_fault: Optional[int] = None
        escalated = False
        healthy_steps_this = 0
        healthy_alarms_this = 0

        for step_i, record in enumerate(traj):
            obs = ext(record)
            decision = eng.step(obs)
            total_steps += 1

            is_faulty = not obs.get("tool_ok", True) or obs.get("error_count_delta", 0) > 0 or obs.get("progress_delta", 0.0) <= 0.0

            if is_faulty and first_fault is None:
                first_fault = step_i + 1

            if decision.anomaly:
                alarms += 1
                if first_alarm is None:
                    first_alarm = step_i + 1
                if not is_faulty:
                    healthy_alarms_this += 1
                    all_healthy_alarms += 1

            if not is_faulty:
                healthy_steps_this += 1
                all_healthy_steps += 1

            if decision.action == ACTION_ESCALATE and not escalated:
                escalated = True
                total_escalations += 1
                if not is_faulty:
                    total_false_escalations += 1

        is_traj_faulty = first_fault is not None
        if is_traj_faulty:
            total_faulty += 1
            if first_alarm is not None:
                total_detected += 1
                delays.append(max(0, first_alarm - (first_fault or 0)))
                # Token savings: steps after detection that we didn't waste
                tokens_saved = max(0, (len(traj) - (first_alarm or len(traj))) * token_cost_per_step)
                total_tokens_saved += tokens_saved

        per_traj.append({
            "trajectory": traj_idx,
            "faulty": is_traj_faulty,
            "steps": len(traj),
            "alarms": alarms,
            "first_alarm": first_alarm,
            "first_fault": first_fault,
            "escalated": escalated,
            "healthy_alarms": healthy_alarms_this,
            "healthy_steps": healthy_steps_this,
        })

        if verbose and (traj_idx + 1) % 10 == 0:
            print(f"  [{traj_idx+1}/{len(trajectories)}] processed")

    return BenchmarkReport(
        n_trajectories=len(trajectories),
        n_steps_total=total_steps,
        detected_rate=total_detected / max(total_faulty, 1),
        detection_delay_mean=float(np.mean(delays)) if delays else float("nan"),
        false_alarm_rate=all_healthy_alarms / max(all_healthy_steps, 1),
        escalations=total_escalations,
        false_escalations=total_false_escalations,
        est_tokens_saved=total_tokens_saved,
        per_trajectory=per_traj,
    )


# ---------------------------------------------------------------------------
# Synthetic data generator — mimics real benchmark trace format
# ---------------------------------------------------------------------------
def generate_synthetic_benchmark_data(
    n_trajectories: int = 50,
    max_steps: int = 30,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Generate benchmark-like trace data for testing.
    Each trajectory is a sequence of step dicts with realistic fields.
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for t in range(n_trajectories):
        # 50% healthy, 50% faulty
        faulty = rng.random() > 0.50
        fault_start = int(rng.integers(5, max_steps - 5)) if faulty else max_steps + 1
        traj_steps = int(rng.integers(max_steps // 2, max_steps))
        steps = []

        for s in range(traj_steps):
            if s >= fault_start:
                tool_ok = rng.random() > 0.75
                output = "Error: operation failed" if not tool_ok else "Partial success"
            else:
                tool_ok = rng.random() > 0.08
                output = f"Step {s} completed successfully."

            steps.append({
                "step": s,
                "tool_ok": tool_ok,
                "output": output,
                "exit_code": 0 if tool_ok else 1,
                "score": 0.1 if tool_ok else -0.05,
            })

        trajectories.append(steps)

    if output_path:
        Path(output_path).write_text(
            "\n".join(json.dumps(step) for traj in trajectories for step in traj),
            encoding="utf-8",
        )

    return trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run judgment against benchmark traces"
    )
    parser.add_argument("--data", default=None,
                        help="JSONL file with benchmark traces (one JSON object per line)")
    parser.add_argument("--generate", type=int, default=30,
                        help="Generate N synthetic benchmark trajectories (default 30)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.data:
        # Load real benchmark traces
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: {data_path} not found.")
            sys.exit(1)
        records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Group by trajectory (naive: look for step==1 or sequential step breaks)
        trajectories: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for rec in records:
            if rec.get("step") == 1 and current:
                trajectories.append(current)
                current = []
            current.append(rec)
        if current:
            trajectories.append(current)
        print(f"Loaded {len(trajectories)} trajectories ({len(records)} steps) from {data_path}")
    else:
        print(f"Generating {args.generate} synthetic benchmark trajectories...")
        trajectories = generate_synthetic_benchmark_data(
            n_trajectories=args.generate,
            output_path=args.output or str(Path(__file__).parent / "benchmark_synthetic.jsonl"),
        )

    print(f"Running judgment on {len(trajectories)} trajectories...")
    engine = DecisionEngine(use_fast_pomcp=True, seed=42)
    t0 = time.time()
    report = run_benchmark(trajectories, engine=engine, verbose=True)
    duration = time.time() - t0

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Trajectories:      {report.n_trajectories}")
    print(f"  Total steps:       {report.n_steps_total}")
    print(f"  Detection rate:    {report.detected_rate:.2%}")
    print(f"  Mean delay:        {report.detection_delay_mean:.1f} steps")
    print(f"  False alarm rate:  {report.false_alarm_rate:.2%}")
    print(f"  Escalations:       {report.escalations} ({report.false_escalations} false)")
    print(f"  Est. tokens saved: {report.est_tokens_saved:,}")
    print(f"  Duration:          {duration:.1f}s")

    if args.output:
        out_path = Path(args.output) if not args.output.endswith(".jsonl") else Path(str(args.output) + ".report.json")
        out_path.write_text(json.dumps({
            "detected_rate": report.detected_rate,
            "detection_delay_mean": report.detection_delay_mean,
            "false_alarm_rate": report.false_alarm_rate,
            "escalations": report.escalations,
            "false_escalations": report.false_escalations,
            "est_tokens_saved": report.est_tokens_saved,
            "n_trajectories": report.n_trajectories,
            "n_steps_total": report.n_steps_total,
            "duration_s": round(duration, 1),
        }, indent=2), encoding="utf-8")
        print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
