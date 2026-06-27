#!/usr/bin/env python3
"""
Minimal example — 30 lines, copy-paste, runs instantly.

    pip install judgment
    python examples/minimal.py

No API keys. No LangGraph. Just the engine watching a simulated agent loop.
"""

try:
    from judgment import DecisionEngine
except ImportError:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.engine import DecisionEngine

engine = DecisionEngine()

# Simulate 15 steps of an Agent — mostly success, 2 injected errors
observations = [
    {"tool_ok": True,  "progress_delta": 0.15, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.12, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.10, "error_count_delta": 0},
    {"tool_ok": False, "progress_delta": -0.05, "error_count_delta": 2},  # error
    {"tool_ok": False, "progress_delta": -0.08, "error_count_delta": 1},  # error
    {"tool_ok": True,  "progress_delta": 0.14, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.16, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.11, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.13, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.09, "error_count_delta": 0},
    {"tool_ok": False, "progress_delta": -0.03, "error_count_delta": 1},  # error
    {"tool_ok": True,  "progress_delta": 0.10, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.08, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.12, "error_count_delta": 0},
    {"tool_ok": True,  "progress_delta": 0.14, "error_count_delta": 0},
]

for i, obs in enumerate(observations):
    d = engine.step(obs)
    print(f"step {i+1:2d}  {d.action:10s}  "
          f"H={d.belief['healthy']:.3f}  "
          f"{'[!] anomaly' if d.anomaly else '           '}  "
          f"{d.rationale[:60]}")

print(f"\nFinal: H={engine.decision_log[-1].belief['healthy']:.3f}, "
      f"{len(engine.decision_log)} steps, "
      f"escalated: {any(d.action == 'escalate' for d in engine.decision_log)}")
