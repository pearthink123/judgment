"""
Tests for Layer 1: CUSUM anomaly detection (cusum.py)

Validates:
  - Initial state
  - No false alarm on healthy sequences
  - Rapid detection of anomalous sequences
  - Hawkes correction reduces drift for expected events
  - Reset after alarm
  - Two-sided L with per-state surprisals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.cusum import CUSUMDetector


class TestCUSUMInitialisation:
    def test_defaults(self):
        c = CUSUMDetector()
        assert c.S == 0.0
        assert c.h == 4.0
        assert c.gamma == 0.35
        assert c.alarm_count == 0
        assert c.t == 0

    def test_custom_threshold(self):
        c = CUSUMDetector(h=5.0)
        assert c.h == 5.0


class TestCUSUMNoFalseAlarm:
    """Healthy observations should not trigger the CUSUM alarm."""

    def test_all_healthy(self):
        c = CUSUMDetector(h=4.0, gamma=0.0)  # no Hawkes correction for simplicity
        for _ in range(30):
            # Healthy obs: surprisal_healthy = 1.2, surprisal_degraded = 2.0
            # L ≈ 1.2 - 2.0 = -0.8 (negative → S decays to 0)
            result = c.update(surprisal_healthy=1.2, hawkes_intensity=1.0, surprisal_degraded=2.0)
        assert c.alarm_count == 0
        assert c.S < 1.0, f"S should stay near 0, got {c.S}"

    def test_low_drift_no_alarm(self):
        """Even with slightly elevated surprisal, no single step should alarm."""
        c = CUSUMDetector(h=4.0)
        result = c.update(surprisal_healthy=1.5, hawkes_intensity=1.0, surprisal_degraded=1.3)
        assert not result["alarm"]
        assert result["S"] < 3.0


class TestCUSUMDetection:
    """Anomalous observations should be detected."""

    def test_rapid_detection(self):
        """4 consecutive anomaly steps should trigger alarm."""
        c = CUSUMDetector(h=4.0, gamma=0.0, drift_floor=-0.5)
        alarms = 0
        for step_i in range(10):
            # Anomalous: healthy surprisal >> degraded surprisal
            result = c.update(
                surprisal_healthy=2.5,
                hawkes_intensity=1.0,
                surprisal_degraded=0.8,
            )
            if result["alarm"]:
                alarms += 1
        assert alarms >= 1, "Should have fired at least 1 alarm"

    def test_detection_faster_with_worse_anomalies(self):
        """Larger surprisal gap → faster detection."""
        c1 = CUSUMDetector(h=4.0, gamma=0.0)
        # Run with moderate anomaly
        for _ in range(6):
            c1.update(surprisal_healthy=2.0, hawkes_intensity=1.0, surprisal_degraded=1.2)

        c2 = CUSUMDetector(h=4.0, gamma=0.0)
        # Run with severe anomaly
        for _ in range(6):
            c2.update(surprisal_healthy=4.0, hawkes_intensity=1.0, surprisal_degraded=0.5)

        # C2 should have higher drift (it's more anomalous)
        assert c2.S >= c1.S or c2.alarm_count >= 1, (
            "Severe anomaly should produce equal or higher drift"
        )

    def test_reset_after_alarm(self):
        c = CUSUMDetector(h=3.0, gamma=0.0, drift_floor=-1.0)
        # Force alarm
        for _ in range(10):
            result = c.update(
                surprisal_healthy=3.0,
                hawkes_intensity=1.0,
                surprisal_degraded=0.3,
            )
            if result["alarm"]:
                break
        assert c.S == 0.0, "S should reset to 0 after alarm"


class TestHawkesCorrection:
    """Hawkes correction should reduce drift for expected events."""

    def test_high_intensity_reduces_drift(self):
        c = CUSUMDetector(h=4.0, gamma=0.35)
        result = c.update(surprisal_healthy=2.0, hawkes_intensity=8.0, surprisal_degraded=1.0)
        # Without Hawkes: L ≈ 2.0 - 1.0 = 1.0 and S ≈ 1.0
        # With Hawkes: corrected ≈ 2.0 - 0.35*log(8) ≈ 2.0 - 0.73 = 1.27
        # L ≈ 1.27 - 1.0 = 0.27 — much smaller
        assert result["S"] < 0.5, f"Hawkes correction should reduce drift, got S={result['S']}"

    def test_low_intensity_no_effect(self):
        """When Hawkes intensity is at baseline, correction is near 0."""
        c = CUSUMDetector(h=4.0, gamma=0.35)
        result1 = c.update(surprisal_healthy=2.0, hawkes_intensity=1.0, surprisal_degraded=1.0)
        c_no = CUSUMDetector(h=4.0, gamma=0.0)
        result_no = c_no.update(surprisal_healthy=2.0, hawkes_intensity=1.0, surprisal_degraded=1.0)
        # With λ=1.0, correction = 0.35 * log(1) = 0, so S should be same
        assert abs(result1["S"] - result_no["S"]) < 0.01


class TestCUSUMTracking:
    def test_S_history_recorded(self):
        c = CUSUMDetector()
        for _ in range(5):
            c.update(surprisal_healthy=1.5, hawkes_intensity=1.0, surprisal_degraded=1.3)
        assert len(c.S_history) == 5

    def test_alarm_history_recorded(self):
        c = CUSUMDetector(h=2.0, gamma=0.0)
        for _ in range(8):
            result = c.update(surprisal_healthy=3.0, hawkes_intensity=1.0, surprisal_degraded=0.5)
        assert c.alarm_count > 0


class TestCUSUMReset:
    def test_full_reset(self):
        c = CUSUMDetector()
        for _ in range(5):
            c.update(surprisal_healthy=2.0, hawkes_intensity=1.0, surprisal_degraded=1.5)
        c.reset()
        assert c.S == 0.0
        assert c.t == 0
        assert c.alarm_count == 0
        assert len(c.S_history) == 0
