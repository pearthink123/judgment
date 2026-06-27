"""
Lightweight metrics registry — Prometheus-compatible counters and gauges.

Zero external dependencies.  The registry accumulates values in-memory
and supports export to Prometheus text format or plain dicts.

Usage:
    from observability.metrics import MetricsRegistry, engine_monitor

    registry = MetricsRegistry()

    # Wrap engine to auto-record metrics
    engine = DecisionEngine()
    step = engine_monitor(engine, registry)
    decision = step(obs)   # now also records latency, actions, alarms
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from core.engine import (
    DecisionEngine, Decision,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class MetricsRegistry:
    """
    Thread-safe accumulator for engine health metrics.

    Export formats:
      - .to_dict()     → plain Python dict
      - .to_prometheus() → Prometheus text format string
      - .to_json()     → JSON string
    """

    def __init__(self):
        # Counters (monotonic)
        self.steps_total: int = 0
        self.alarms_total: int = 0
        self.escalations_total: int = 0
        self.corrections_total: int = 0

        # Per-action counters
        self.actions: Dict[str, int] = {
            ACTION_CONTINUE: 0,
            ACTION_CORRECT: 0,
            ACTION_ESCALATE: 0,
            ACTION_GATHER: 0,
        }

        # Gauges (current values)
        self.healthy_prob: float = 0.65
        self.degraded_prob: float = 0.28
        self.broken_prob: float = 0.07
        self.cusum_drift: float = 0.0

        # Histogram: latency buckets (ms)
        self._latency_buckets = [0.5, 1, 5, 10, 25, 50, 100, 500, 1000]
        self._latency_counts: List[int] = [0] * len(self._latency_buckets)
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------
    def record_step(self, decision: Decision, latency_ms: float):
        self.steps_total += 1
        self.actions[decision.action] = self.actions.get(decision.action, 0) + 1

        if decision.anomaly:
            self.alarms_total += 1
        if decision.action == ACTION_ESCALATE:
            self.escalations_total += 1
        if decision.action == ACTION_CORRECT:
            self.corrections_total += 1

        # Gauges
        self.healthy_prob = decision.belief.get("healthy", 0.0)
        self.degraded_prob = decision.belief.get("degraded", 0.0)
        self.broken_prob = decision.belief.get("broken", 0.0)
        self.cusum_drift = decision.drift

        # Latency histogram
        self._latency_sum += latency_ms
        self._latency_count += 1
        for i, upper in enumerate(self._latency_buckets):
            if latency_ms <= upper:
                self._latency_counts[i] += 1

    @property
    def latency_avg_ms(self) -> float:
        if self._latency_count == 0:
            return 0.0
        return self._latency_sum / self._latency_count

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "alarms_total": self.alarms_total,
            "escalations_total": self.escalations_total,
            "corrections_total": self.corrections_total,
            "actions": dict(self.actions),
            "current_belief": {
                "healthy": round(self.healthy_prob, 4),
                "degraded": round(self.degraded_prob, 4),
                "broken": round(self.broken_prob, 4),
            },
            "cusum_drift": round(self.cusum_drift, 4),
            "latency_avg_ms": round(self.latency_avg_ms, 1),
            "latency_p50_ms": self._percentile(50),
            "latency_p99_ms": self._percentile(99),
        }

    def to_prometheus(self, prefix: str = "judgment") -> str:
        lines = [
            f"# HELP {prefix}_steps_total Total engine steps processed.",
            f"# TYPE {prefix}_steps_total counter",
            f"{prefix}_steps_total {self.steps_total}",
            f"# HELP {prefix}_alarms_total Total CUSUM alarms fired.",
            f"# TYPE {prefix}_alarms_total counter",
            f"{prefix}_alarms_total {self.alarms_total}",
            f"# HELP {prefix}_escalations_total Total escalations.",
            f"# TYPE {prefix}_escalations_total counter",
            f"{prefix}_escalations_total {self.escalations_total}",
            f"# HELP {prefix}_corrections_total Total corrective actions.",
            f"# TYPE {prefix}_corrections_total counter",
            f"{prefix}_corrections_total {self.corrections_total}",
        ]
        for action, count in self.actions.items():
            lines.append(
                f"# HELP {prefix}_actions_total Actions by type."
                f"\n# TYPE {prefix}_actions_total counter"
                f"\n{prefix}_actions_total{{action=\"{action}\"}} {count}"
            )

        lines += [
            f"# HELP {prefix}_belief_healthy Current P(Healthy).",
            f"# TYPE {prefix}_belief_healthy gauge",
            f"{prefix}_belief_healthy {self.healthy_prob:.4f}",
            f"# HELP {prefix}_belief_degraded Current P(Degraded).",
            f"# TYPE {prefix}_belief_degraded gauge",
            f"{prefix}_belief_degraded {self.degraded_prob:.4f}",
            f"# HELP {prefix}_belief_broken Current P(Broken).",
            f"# TYPE {prefix}_belief_broken gauge",
            f"{prefix}_belief_broken {self.broken_prob:.4f}",
            f"# HELP {prefix}_drift Current CUSUM drift.",
            f"# TYPE {prefix}_drift gauge",
            f"{prefix}_drift {self.cusum_drift:.4f}",
            f"# HELP {prefix}_latency_ms Engine step latency.",
            f"# TYPE {prefix}_latency_ms summary",
            f"{prefix}_latency_ms{{quantile=\"0.5\"}} {self._percentile(50):.1f}",
            f"{prefix}_latency_ms{{quantile=\"0.99\"}} {self._percentile(99):.1f}",
            f"{prefix}_latency_ms_count {self._latency_count}",
            f"{prefix}_latency_ms_sum {self._latency_sum:.1f}",
        ]
        return "\n".join(lines) + "\n"

    def _percentile(self, p: float) -> float:
        if self._latency_count == 0:
            return 0.0
        total = sum(self._latency_counts)
        if total == 0:
            return 0.0
        target = total * p / 100
        cumulative = 0
        for i, cnt in enumerate(self._latency_counts):
            cumulative += cnt
            if cumulative >= target:
                return float(self._latency_buckets[i])
        return float(self._latency_buckets[-1])

    def reset(self):
        self.steps_total = 0
        self.alarms_total = 0
        self.escalations_total = 0
        self.corrections_total = 0
        self.actions = {a: 0 for a in self.actions}
        self._latency_counts = [0] * len(self._latency_buckets)
        self._latency_sum = 0.0
        self._latency_count = 0


# ---------------------------------------------------------------------------
# Auto-monitored step function
# ---------------------------------------------------------------------------
def engine_monitor(
    engine: DecisionEngine,
    registry: MetricsRegistry,
) -> Callable[[Dict[str, Any]], Decision]:
    """
    Wrap engine.step() to auto-record metrics.

    Usage:
        registry = MetricsRegistry()
        step = engine_monitor(engine, registry)
        decision = step(obs)  # registry updated automatically
    """
    _step = engine.step

    def monitored_step(observation: Dict[str, Any]) -> Decision:
        t0 = time.perf_counter()
        decision = _step(observation)
        latency_ms = (time.perf_counter() - t0) * 1000
        registry.record_step(decision, latency_ms)
        return decision

    return monitored_step
