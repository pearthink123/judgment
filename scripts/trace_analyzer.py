#!/usr/bin/env python3
"""
Offline trace analyzer — validate judgment against real Agent run logs.

Accepts JSONL/JSON traces from LangSmith, Helicone, OpenTelemetry, or
any custom harness. Replays the engine against historical data and produces:

  1. Per-step diagnostic table (belief, drift, solver, anomaly)
  2. Aggregate stats (detection recall, FPR, escalation accuracy)
  3. Signal quality report (noise level, missing signals, distribution shift)

Usage:
    python scripts/trace_analyzer.py --input runs.jsonl
    python scripts/trace_analyzer.py --input runs.jsonl --output report.json
    python scripts/trace_analyzer.py --input runs.jsonl --verbose > analysis.log

Input format (JSONL, one JSON object per step):
    {"run_id": "abc", "step": 1, "tool_ok": true, "output": "...", ...}
    {"run_id": "abc", "step": 2, "tool_ok": false, "output": "Error: ...", ...}

Or JSON array of arrays (one trajectory = one array of step dicts).
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import (
    DecisionEngine, Decision,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)
from core.config import EngineConfig


# ---------------------------------------------------------------------------
# Signal quality metrics
# ---------------------------------------------------------------------------
@dataclass
class SignalQualityReport:
    """Measures how clean the input signal is."""
    n_steps: int
    tool_ok_ratio: float           # % steps where tool_ok=True
    progress_mean: float           # mean(progress_delta)
    progress_std: float            # std — high = noisy progress signal
    error_rate_mean: float         # mean(error_count_delta)
    zero_progress_ratio: float     # % steps with progress_delta ≈ 0
    missing_tool_ok_ratio: float   # % steps lacking tool_ok field
    missing_progress_ratio: float  # % steps lacking progress_delta
    signal_quality_score: float    # 0-1 composite (higher = cleaner)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "tool_ok_ratio": round(self.tool_ok_ratio, 4),
            "progress_mean": round(self.progress_mean, 4),
            "progress_std": round(self.progress_std, 4),
            "error_rate_mean": round(self.error_rate_mean, 4),
            "zero_progress_ratio": round(self.zero_progress_ratio, 4),
            "missing_tool_ok": round(self.missing_tool_ok_ratio, 4),
            "missing_progress": round(self.missing_progress_ratio, 4),
            "quality_score": round(self.signal_quality_score, 4),
        }


def compute_signal_quality(records: List[Dict[str, Any]]) -> SignalQualityReport:
    n = len(records)
    ok_count = sum(1 for r in records if r.get("tool_ok") is True)
    prog_vals = [r.get("progress_delta", r.get("progress", 0.0)) for r in records]
    prog_mean = float(np.mean(prog_vals)) if prog_vals else 0.0
    prog_std = float(np.std(prog_vals)) if len(prog_vals) > 1 else 0.0
    err_vals = [r.get("error_count_delta", r.get("errors", 0)) for r in records]
    err_mean = float(np.mean(err_vals)) if err_vals else 0.0
    zero_prog = sum(1 for v in prog_vals if abs(v) < 0.01) / max(n, 1)
    missing_ok = sum(1 for r in records if r.get("tool_ok") is None) / max(n, 1)
    missing_prog = sum(1 for r in records if r.get("progress_delta") is None) / max(n, 1)

    # Composite quality: higher is better
    # Penalise missing fields and extreme noise
    quality = (
        0.30 * (1.0 - missing_ok) +
        0.30 * (1.0 - missing_prog) +
        0.20 * (1.0 - min(prog_std / 0.3, 1.0)) +
        0.20 * (1.0 - min(err_mean / 3.0, 1.0))
    )
    quality = max(0.0, min(1.0, quality))

    return SignalQualityReport(
        n_steps=n,
        tool_ok_ratio=ok_count / max(n, 1),
        progress_mean=prog_mean,
        progress_std=prog_std,
        error_rate_mean=err_mean,
        zero_progress_ratio=zero_prog,
        missing_tool_ok_ratio=missing_ok,
        missing_progress_ratio=missing_prog,
        signal_quality_score=quality,
    )


# ---------------------------------------------------------------------------
# Trace replay engine
# ---------------------------------------------------------------------------
@dataclass
class TraceAnalysisResult:
    """Per-trajectory analysis output."""
    run_id: str
    n_steps: int
    actions: List[str]
    belief_trajectory: List[Dict[str, float]]
    drift_trajectory: List[float]
    anomalies: List[int]          # step numbers where alarm fired
    escalated: bool
    escalated_step: Optional[int]
    final_belief: Dict[str, float]
    # Detection: ground-truth labels (if provided in trace)
    had_known_fault: bool
    fault_detected: bool
    detection_delay: Optional[int]
    # Per-step diagnostics
    step_details: List[Dict[str, Any]]


@dataclass
class TraceAnalysisReport:
    config_name: str
    n_runs: int
    n_steps_total: int
    signal_quality: SignalQualityReport
    per_run: List[TraceAnalysisResult]

    # aggregate
    detection_recall: float
    false_positive_rate: float
    mean_delay: float
    escalation_rate: float
    false_escalation_rate: float


# ---------------------------------------------------------------------------
# Default extractor — same as benchmark adapter
# ---------------------------------------------------------------------------
from scripts.benchmark_adapter import default_extractor


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def analyze_traces(
    trajectories: List[List[Dict[str, Any]]],
    config: Optional[EngineConfig] = None,
    run_ids: Optional[List[str]] = None,
    verbose: bool = False,
) -> TraceAnalysisReport:
    cfg = config or EngineConfig.preset("default")
    engine = DecisionEngine.from_config(cfg)

    all_records: List[Dict[str, Any]] = []
    for traj in trajectories:
        all_records.extend(traj)

    signal_quality = compute_signal_quality(all_records)

    per_run: List[TraceAnalysisResult] = []

    for traj_idx, traj in enumerate(trajectories):
        engine.reset()
        run_id = run_ids[traj_idx] if run_ids else str(traj_idx)

        actions: List[str] = []
        beliefs: List[Dict[str, float]] = []
        drifts: List[float] = []
        anomaly_steps: List[int] = []
        escalated = False
        escalated_step: Optional[int] = None
        step_details: List[Dict[str, Any]] = []

        # Check if trace has ground-truth fault labels
        has_fault_labels = any(
            "fault" in r or "is_error" in r or "expected_failure" in r
            for r in traj
        )
        known_fault_steps = set()
        if has_fault_labels:
            for i, r in enumerate(traj):
                if r.get("fault") or r.get("is_error") or r.get("expected_failure"):
                    known_fault_steps.add(i + 1)

        had_known_fault = len(known_fault_steps) > 0
        fault_detected = False
        first_alarm: Optional[int] = None
        first_fault: Optional[int] = min(known_fault_steps) if known_fault_steps else None

        for step_i, record in enumerate(traj):
            obs = default_extractor(record)
            decision = engine.step(obs)

            actions.append(decision.action)
            beliefs.append(dict(decision.belief))
            drifts.append(decision.drift)

            if decision.anomaly:
                anomaly_steps.append(step_i + 1)
                if first_alarm is None:
                    first_alarm = step_i + 1
                if had_known_fault and (step_i + 1) in known_fault_steps:
                    fault_detected = True

            if decision.action == ACTION_ESCALATE and not escalated:
                escalated = True
                escalated_step = step_i + 1

            step_details.append({
                "step": step_i + 1,
                "action": decision.action,
                "belief": decision.belief,
                "drift": decision.drift,
                "anomaly": decision.anomaly,
                "solver": decision.layer_diagnostics.get("solver", "?"),
                "rationale": decision.rationale[:80],
            })

        detection_delay = None
        if first_alarm is not None and first_fault is not None:
            detection_delay = max(0, first_alarm - first_fault)

        per_run.append(TraceAnalysisResult(
            run_id=run_id,
            n_steps=len(traj),
            actions=actions,
            belief_trajectory=beliefs,
            drift_trajectory=drifts,
            anomalies=anomaly_steps,
            escalated=escalated,
            escalated_step=escalated_step,
            final_belief=beliefs[-1] if beliefs else {},
            had_known_fault=had_known_fault,
            fault_detected=fault_detected,
            detection_delay=detection_delay,
            step_details=step_details,
        ))

        if verbose and (traj_idx + 1) % 10 == 0:
            print(f"  [{traj_idx + 1}/{len(trajectories)}] {run_id}: "
                  f"{len(anomaly_steps)} alarms, escalated={escalated}")

    # ---- Aggregate ----
    runs_with_faults = [r for r in per_run if r.had_known_fault]
    healthy_runs = [r for r in per_run if not r.had_known_fault]

    recall = sum(1 for r in runs_with_faults if r.fault_detected) / max(len(runs_with_faults), 1)

    # FPR: % healthy steps with alarm (when using ground-truth labels)
    healthy_alarms = sum(len(r.anomalies) for r in healthy_runs)
    healthy_steps_total = sum(r.n_steps for r in healthy_runs)
    fpr = healthy_alarms / max(healthy_steps_total, 1)

    delays = [r.detection_delay for r in runs_with_faults if r.detection_delay is not None]
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    esc_rate = sum(1 for r in per_run if r.escalated) / max(len(per_run), 1)
    false_esc = sum(1 for r in healthy_runs if r.escalated) / max(len(healthy_runs), 1)

    return TraceAnalysisReport(
        config_name=cfg.preset.__name__ if hasattr(cfg, 'preset') else "custom",
        n_runs=len(trajectories),
        n_steps_total=sum(len(t) for t in trajectories),
        signal_quality=signal_quality,
        per_run=per_run,
        detection_recall=recall,
        false_positive_rate=fpr,
        mean_delay=mean_delay,
        escalation_rate=esc_rate,
        false_escalation_rate=false_esc,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_traces(path: str) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """Load trajectories from JSONL (one object per line) or JSON array."""
    raw = Path(path).read_text(encoding="utf-8").strip()

    # Try JSON array of arrays
    if raw.startswith("[["):
        data = json.loads(raw)
        return data, [f"traj_{i}" for i in range(len(data))]

    if raw.startswith("[{"):
        data = json.loads(raw)
        return [data], [f"traj_0"]

    # JSONL: one object per line, group by run_id
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not lines:
        return [], []

    # Group by run_id
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in lines:
        rid = rec.get("run_id", rec.get("trace_id", rec.get("id", "default")))
        groups.setdefault(str(rid), []).append(rec)

    return list(groups.values()), list(groups.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze real Agent traces with judgment engine"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to JSONL or JSON trace file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON report path")
    parser.add_argument("--preset", default="default",
                        choices=["default", "fast", "conservative", "permissive", "lightweight"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    trajectories, run_ids = load_traces(str(input_path))
    if not trajectories:
        print("Error: no valid trajectories found in input.")
        sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectories "
          f"({sum(len(t) for t in trajectories)} steps) from {input_path.name}")

    cfg = EngineConfig.preset(args.preset)

    t0 = time.time()
    report = analyze_traces(trajectories, config=cfg, run_ids=run_ids, verbose=args.verbose)
    duration = time.time() - t0

    # ---- Output ----
    print()
    print("=" * 65)
    print("TRACE ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Runs:             {report.n_runs}")
    print(f"  Total steps:      {report.n_steps_total}")
    print(f"  Duration:         {duration:.1f}s")
    print()
    print("--- Signal Quality ---")
    sq = report.signal_quality
    print(f"  Quality score:    {sq.signal_quality_score:.2f} / 1.0")
    print(f"  tool_ok ratio:    {sq.tool_ok_ratio:.2%}")
    print(f"  progress μ/σ:     {sq.progress_mean:.3f} / {sq.progress_std:.3f}")
    print(f"  error rate:       {sq.error_rate_mean:.3f}")
    print(f"  zero progress:    {sq.zero_progress_ratio:.2%}")
    print(f"  missing tool_ok:  {sq.missing_tool_ok_ratio:.2%}")
    print(f"  missing progress: {sq.missing_progress_ratio:.2%}")
    print()
    print("--- Detection ---")
    print(f"  Recall:           {report.detection_recall:.2%}")
    print(f"  False pos rate:   {report.false_positive_rate:.2%}")
    print(f"  Mean delay:       {report.mean_delay:.1f} steps")
    print(f"  Escalation rate:  {report.escalation_rate:.2%} "
          f"({report.false_escalation_rate:.2%} false)")

    if args.verbose:
        print()
        print("--- Per-Run Summary ---")
        for r in report.per_run:
            print(f"  {r.run_id:20s}  steps={r.n_steps:3d}  "
                  f"alarms={len(r.anomalies):2d}  "
                  f"esc={r.escalated}  "
                  f"H={r.final_belief.get('healthy', 0):.3f}  "
                  f"fault={'Y' if r.had_known_fault else 'N'}  "
                  f"detected={'Y' if r.fault_detected else '-'}")

    if args.output:
        out = {
            "n_runs": report.n_runs,
            "n_steps_total": report.n_steps_total,
            "duration_s": round(duration, 1),
            "signal_quality": sq.to_dict(),
            "detection_recall": round(report.detection_recall, 4),
            "false_positive_rate": round(report.false_positive_rate, 4),
            "mean_delay": report.mean_delay,
            "escalation_rate": round(report.escalation_rate, 4),
            "false_escalation_rate": round(report.false_escalation_rate, 4),
            "per_run": [
                {
                    "run_id": r.run_id,
                    "n_steps": r.n_steps,
                    "anomalies": r.anomalies,
                    "escalated": r.escalated,
                    "escalated_step": r.escalated_step,
                    "final_belief": r.final_belief,
                    "had_known_fault": r.had_known_fault,
                    "fault_detected": r.fault_detected,
                    "detection_delay": r.detection_delay,
                }
                for r in report.per_run
            ],
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
