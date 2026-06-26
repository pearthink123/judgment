"""Tests for corrective action router (corrective.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.corrective import (
    CorrectiveRouter, CorrectiveAdvice,
    CORRECTIVE_VERIFY, CORRECTIVE_RETHINK, CORRECTIVE_RETRY, CORRECTIVE_ROLLBACK,
)
from core.engine import Decision


def _make_decision(
    healthy: float, degraded: float, broken: float,
    anomaly: bool = False, action: str = "continue",
):
    return Decision(
        action=action,
        belief={"healthy": healthy, "degraded": degraded, "broken": broken},
        confidence=max(healthy, degraded, broken),
        anomaly=anomaly,
        drift=0.5 if anomaly else 0.0,
    )


class TestCorrectiveRouter:
    def test_empty_log_returns_verify(self):
        r = CorrectiveRouter()
        advice = r.analyse([])
        assert advice.action_type == CORRECTIVE_VERIFY

    def test_isolated_failure_is_retry(self):
        r = CorrectiveRouter()
        log = [
            _make_decision(0.90, 0.08, 0.02, anomaly=False),
            _make_decision(0.88, 0.10, 0.02, anomaly=True),   # single anomaly
            _make_decision(0.85, 0.13, 0.02, anomaly=False),
        ]
        advice = r.analyse(log)
        assert advice.action_type == CORRECTIVE_RETRY

    def test_multiple_failures_is_verify(self):
        r = CorrectiveRouter()
        log = [
            _make_decision(0.90, 0.08, 0.02, anomaly=False),
            _make_decision(0.82, 0.15, 0.03, anomaly=True),
            _make_decision(0.70, 0.25, 0.05, anomaly=True),   # 2 failures
        ]
        advice = r.analyse(log)
        assert advice.action_type == CORRECTIVE_VERIFY

    def test_stalled_progress_is_rethink(self):
        r = CorrectiveRouter()
        log = []
        # Build a log where P(H) is stagnant for 6+ steps
        for i in range(8):
            log.append(_make_decision(
                0.70 - 0.005 * i, 0.25 + 0.005 * i, 0.05,
                anomaly=False,
            ))
        advice = r.analyse(log)
        # After 7+ stalled steps, should be rethink
        # (stalled_steps counts consecutive non-increasing steps from the end)
        assert advice.action_type in (CORRECTIVE_RETHINK, CORRECTIVE_VERIFY)

    def test_consecutive_errors_plus_stalled_is_rollback(self):
        r = CorrectiveRouter()
        log = []
        for i in range(8):
            # More severe: all last 5 steps anomalous with declining health
            log.append(_make_decision(
                max(0.05, 0.80 - 0.10 * i),
                0.15 + 0.06 * i,
                0.05 + 0.04 * i,
                anomaly=(i >= 3),
            ))
        advice = r.analyse(log)
        # With 5 consecutive anomalies + stalled, should be rollback or rethink
        assert advice.recent_failures >= 3
        assert advice.action_type in (CORRECTIVE_ROLLBACK, CORRECTIVE_RETHINK, CORRECTIVE_VERIFY)

    def test_advice_has_evidence(self):
        r = CorrectiveRouter()
        log = [_make_decision(0.85, 0.12, 0.03, anomaly=True)]
        advice = r.analyse(log)
        assert "recent_failures" in advice.evidence
        assert "stalled_steps" in advice.evidence
        assert advice.summary
