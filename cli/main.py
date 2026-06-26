"""
judgment — CLI entry point.

Commands:
  judgment run TASK       Run a task with the JudgmentHarness.
  judgment train LOGDIR   Learn HMM parameters from logs.
  judgment dashboard      Launch Streamlit diagnostics.
"""

import sys
import argparse
import json
from pathlib import Path

# Ensure judgment package is importable from the CLI
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_run(args):
    """Run a task through the harness."""
    from harness.loop import JudgmentHarness
    from harness.executor import SimulatedExecutor, LLMExecutor
    from core.pomdp import RewardConfig

    # Choose executor
    if args.api_key or args.model:
        executor = LLMExecutor(
            model=args.model or "deepseek-chat",
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        print("[simulated mode — no API key provided]")
        executor = SimulatedExecutor(seed=args.seed)

    # Reward preset
    reward = RewardConfig.preset(args.preset) if args.preset != "general" else None

    harness = JudgmentHarness(
        executor=executor,
        reward=reward,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    task = " ".join(args.task)
    print(f"\nTask: {task}")
    print(f"Max steps: {args.max_steps} | Preset: {args.preset}")
    print("=" * 60)

    result = harness.run(task)

    print()
    print(f"Status:    {result.status}")
    print(f"Steps:     {result.steps}")
    print(f"Duration:  {result.duration_seconds}s")
    print(f"Belief:    H={result.final_belief.get('healthy',0):.3f} "
          f"D={result.final_belief.get('degraded',0):.3f} "
          f"B={result.final_belief.get('broken',0):.3f}")
    print(f"Summary:   {result.summary}")

    # Decision trace
    if args.verbose:
        print("\nDecision trace:")
        for d in result.decision_log:
            print(f"  step {result.decision_log.index(d)+1}: {d.action} "
                  f"(H={d.belief.get('healthy',0):.2f}, "
                  f"D={d.belief.get('degraded',0):.2f}, "
                  f"B={d.belief.get('broken',0):.2f}) "
                  f"anomaly={d.anomaly}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps({
            "status": result.status,
            "steps": result.steps,
            "final_belief": result.final_belief,
            "summary": result.summary,
            "duration_seconds": result.duration_seconds,
            "decision_log": [
                {
                    "step": i + 1,
                    "action": d.action,
                    "belief": d.belief,
                    "rationale": d.rationale,
                }
                for i, d in enumerate(result.decision_log)
            ],
        }, indent=2), encoding="utf-8")
        print(f"\nOutput written to {out_path}")


def cmd_train(args):
    """Train HMM parameters from agent run logs."""
    from core.training import train_hmm
    import glob

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: {logdir} does not exist.")
        sys.exit(1)

    json_files = sorted(logdir.glob("*.json")) + sorted(logdir.glob("*.jsonl"))
    if not json_files:
        print(f"No .json or .jsonl files found in {logdir}")
        sys.exit(1)

    logs = []
    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Assume each file is one trajectory
            logs.append(data)
        else:
            print(f"Warning: {f.name} is not a list — skipping.")

    if not logs:
        print("No valid trajectories loaded.")
        sys.exit(1)

    print(f"Loaded {len(logs)} trajectories from {len(json_files)} files.")
    print(f"Running Baum-Welch for {args.iterations} iterations...")

    prior, T, B, ll_history = train_hmm(
        logs, labels=None, n_iter=args.iterations, tol=args.tolerance,
    )

    print(f"Final log-likelihood: {ll_history[-1]:.2f}")
    print(f"Prior: {prior}")
    print(f"Transition:\n{T}")

    if args.output:
        import numpy as np
        out = {
            "prior": prior.tolist(),
            "T": T.tolist(),
            "B": {str(k): v.tolist() for k, v in B.items()},
            "log_lik_history": ll_history,
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Parameters saved to {args.output}")


def cmd_dashboard(_args):
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", str(_args.port),
    ])


def main():
    parser = argparse.ArgumentParser(
        prog="judgment",
        description="Math-driven Agent Harness — decision engine + execution loop.",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- run ----
    run_p = sub.add_parser("run", help="Run a task through the harness.")
    run_p.add_argument("task", nargs="+", help="Task description.")
    run_p.add_argument("--model", default=None, help="LLM model name.")
    run_p.add_argument("--api-key", default=None, help="API key (or set DEEPSEEK_API_KEY env).")
    run_p.add_argument("--base-url", default=None, help="API base URL override.")
    run_p.add_argument("--temperature", type=float, default=0.3)
    run_p.add_argument("--max-tokens", type=int, default=4096)
    run_p.add_argument("--max-steps", type=int, default=40)
    run_p.add_argument("--preset", default="general",
                       choices=["general", "conservative", "permissive"])
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--verbose", "-v", action="store_true")
    run_p.add_argument("--output", "-o", default=None, help="Save result to JSON file.")

    # ---- train ----
    train_p = sub.add_parser("train", help="Learn HMM parameters from logs.")
    train_p.add_argument("logdir", help="Directory containing .json/.jsonl trajectory files.")
    train_p.add_argument("--iterations", type=int, default=50)
    train_p.add_argument("--tolerance", type=float, default=1e-4)
    train_p.add_argument("--output", "-o", default=None, help="Save learned params to JSON.")

    # ---- dashboard ----
    dash_p = sub.add_parser("dashboard", help="Launch Streamlit diagnostic dashboard.")
    dash_p.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
