"""Tests for observability/metrics.py and observability/logging.py."""

import sys, json, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from observability.metrics import MetricsRegistry, engine_monitor
from observability.logging import JudgmentLogger
from core.engine import DecisionEngine, Decision


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------
class TestMetricsRegistry:
    def test_init_zeros(self):
        r = MetricsRegistry()
        assert r.steps_total == 0
        assert r.alarms_total == 0
        assert r.escalations_total == 0

    def test_record_step_updates_counters(self):
        r = MetricsRegistry()
        d = Decision(action="continue", belief={"healthy": 0.9, "degraded": 0.08, "broken": 0.02},
                     confidence=0.9, anomaly=False, drift=0.0)
        r.record_step(d, 2.5)
        assert r.steps_total == 1
        assert r.actions["continue"] == 1
        assert r.healthy_prob == 0.9

    def test_record_anomaly(self):
        r = MetricsRegistry()
        d = Decision(action="correct", belief={"healthy": 0.3, "degraded": 0.5, "broken": 0.2},
                     confidence=0.5, anomaly=True, drift=3.2)
        r.record_step(d, 10.0)
        assert r.alarms_total == 1
        assert r.corrections_total == 1

    def test_record_escalate(self):
        r = MetricsRegistry()
        d = Decision(action="escalate", belief={"healthy": 0.01, "degraded": 0.1, "broken": 0.89},
                     confidence=0.89, anomaly=True, drift=5.0)
        r.record_step(d, 30.0)
        assert r.escalations_total == 1

    def test_latency_histogram(self):
        r = MetricsRegistry()
        for _ in range(10):
            d = Decision(action="continue", belief={"healthy": 0.9, "degraded": 0.08, "broken": 0.02},
                         confidence=0.9, anomaly=False, drift=0.0)
            r.record_step(d, 3.0)
        assert 2.0 < r.latency_avg_ms < 5.0

    def test_to_dict(self):
        r = MetricsRegistry()
        d = Decision(action="continue", belief={"healthy": 0.95, "degraded": 0.04, "broken": 0.01},
                     confidence=0.95, anomaly=False, drift=0.0)
        r.record_step(d, 1.5)
        out = r.to_dict()
        assert out["steps_total"] == 1
        assert out["current_belief"]["healthy"] == 0.95

    def test_to_prometheus(self):
        r = MetricsRegistry()
        d = Decision(action="continue", belief={"healthy": 0.9, "degraded": 0.08, "broken": 0.02},
                     confidence=0.9, anomaly=False, drift=0.0)
        r.record_step(d, 5.0)
        text = r.to_prometheus()
        assert 'judgment_steps_total 1' in text
        assert 'judgment_belief_healthy' in text
        assert 'action="continue"' in text

    def test_reset(self):
        r = MetricsRegistry()
        d = Decision(action="continue", belief={"healthy": 0.9, "degraded": 0.08, "broken": 0.02},
                     confidence=0.9, anomaly=False, drift=0.0)
        r.record_step(d, 5.0)
        r.reset()
        assert r.steps_total == 0
        assert r.alarms_total == 0


# ---------------------------------------------------------------------------
# engine_monitor
# ---------------------------------------------------------------------------
class TestEngineMonitor:
    def test_monitored_step_returns_decision(self):
        engine = DecisionEngine(seed=1)
        registry = MetricsRegistry()
        step = engine_monitor(engine, registry)

        decision = step({"tool_ok": True, "progress_delta": 0.15, "has_user_msg": False, "error_count_delta": 0})
        assert decision.action in {"continue", "correct", "escalate", "gather"}

    def test_monitored_step_records_metrics(self):
        engine = DecisionEngine(seed=1)
        registry = MetricsRegistry()
        step = engine_monitor(engine, registry)

        for _ in range(5):
            step({"tool_ok": True, "progress_delta": 0.12, "has_user_msg": False, "error_count_delta": 0})
        assert registry.steps_total == 5
        assert registry.actions["continue"] >= 3

    def test_monitored_step_tracks_escalations(self):
        engine = DecisionEngine(seed=3)
        registry = MetricsRegistry()
        step = engine_monitor(engine, registry)

        for _ in range(5):
            step({"tool_ok": True, "progress_delta": 0.12, "has_user_msg": False, "error_count_delta": 0})
        for _ in range(8):
            step({"tool_ok": False, "progress_delta": -0.05, "has_user_msg": False, "error_count_delta": 2})

        assert registry.steps_total == 13
        # Should have at least one escalation after that many errors
        assert registry.escalations_total >= 1 or registry.alarms_total >= 1

    def test_different_engines_independent_metrics(self):
        e1 = DecisionEngine(seed=1)
        e2 = DecisionEngine(seed=2)
        r1 = MetricsRegistry()
        r2 = MetricsRegistry()
        s1 = engine_monitor(e1, r1)
        s2 = engine_monitor(e2, r2)

        s1({"tool_ok": True, "progress_delta": 0.1, "has_user_msg": False, "error_count_delta": 0})
        assert r1.steps_total == 1
        assert r2.steps_total == 0


# ---------------------------------------------------------------------------
# JudgmentLogger
# ---------------------------------------------------------------------------
class TestJudgmentLogger:
    def test_logger_step_does_not_crash(self, capsys):
        logger = JudgmentLogger()
        d = Decision(action="continue", belief={"healthy": 0.9, "degraded": 0.08, "broken": 0.02},
                     confidence=0.9, anomaly=False, drift=0.0)
        logger.step(1, d, latency_ms=3.2)
        captured = capsys.readouterr()
        assert '"step"' in captured.out
        assert '"continue"' in captured.out

    def test_logger_escalate(self, capsys):
        logger = JudgmentLogger()
        d = Decision(action="escalate", belief={"healthy": 0.01, "degraded": 0.1, "broken": 0.89},
                     confidence=0.89, anomaly=True, drift=5.0)
        logger.escalate(10, d)
        captured = capsys.readouterr()
        assert '"escalate"' in captured.out

    def test_logger_train(self, capsys):
        logger = JudgmentLogger()
        logger.train_complete(n_trajectories=10, log_lik=-42.5, duration_s=3.2)
        captured = capsys.readouterr()
        assert '"train_complete"' in captured.out
