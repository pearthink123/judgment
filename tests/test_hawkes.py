"""
Tests for the multivariate marked Hawkes process (hawkes.py).

Validates:
  - Intensity monotonicity after events
  - Exponential decay
  - Stationarity check
  - Mark sampling bounds
  - Multi-type event recording
  - Surprisal computation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.hawkes import (
    HawkesProcess,
    HawkesEvent,
    EVENT_SUCCESS,
    EVENT_ERROR,
    EVENT_USER,
    EVENT_TOOL,
    N_TYPES,
    MARK_CONFIG,
)


class TestHawkesInit:
    def test_default_init(self):
        hp = HawkesProcess()
        assert hp.mu.shape == (N_TYPES,)
        assert hp.alpha.shape == (N_TYPES, N_TYPES)
        assert hp.beta == 1.0
        assert len(hp.events) == 0

    def test_stationarity(self):
        hp = HawkesProcess()
        assert hp.check_stationarity(), (
            f"Default params should be stationary, ρ={hp.spectral_radius}"
        )

    def test_custom_params(self):
        mu = np.array([0.5, 0.2, 0.1, 0.6])
        alpha = 0.3 * np.ones((4, 4))
        hp = HawkesProcess(mu=mu, alpha=alpha, beta=1.0)
        np.testing.assert_array_equal(hp.mu, mu)
        np.testing.assert_array_equal(hp.alpha, alpha)


class TestIntensity:
    def test_baseline_intensity(self):
        hp = HawkesProcess()
        lam = hp.intensity(0.0)
        np.testing.assert_array_almost_equal(lam, hp.mu)

    def test_intensity_increases_after_event(self):
        hp = HawkesProcess()
        baseline = hp.intensity(0.0)
        hp.add_event(0.5, EVENT_TOOL, mark=1.0)
        after = hp.intensity(1.0)
        assert after[EVENT_TOOL] > baseline[EVENT_TOOL]

    def test_error_self_excites_more_than_success(self):
        """α_{err,err} > α_{err,succ} → error→error should be strongest cross-excitation."""
        alpha = hp_default = HawkesProcess().alpha
        assert alpha[EVENT_ERROR, EVENT_ERROR] > alpha[EVENT_ERROR, EVENT_SUCCESS], (
            "Error self-excitation should exceed error-from-success"
        )

    def test_zero_excitation_success_from_error(self):
        """α_{succ,err} = 0 by design — success probability doesn't rise after errors."""
        hp = HawkesProcess()
        assert hp.alpha[EVENT_SUCCESS, EVENT_ERROR] == 0.0


class TestDecay:
    def test_intensity_decays_over_time(self):
        hp = HawkesProcess()
        hp.add_event(0.0, EVENT_ERROR, mark=1.5)
        lam_early = hp.intensity(0.5)
        lam_late = hp.intensity(3.0)
        # After 3 steps, intensity should be closer to baseline
        diff_early = np.linalg.norm(lam_early - hp.mu)
        diff_late = np.linalg.norm(lam_late - hp.mu)
        assert diff_late < diff_early, "Intensity should decay toward baseline"


class TestMarkSampling:
    def test_static_sample_mark_in_bounds(self):
        rng = np.random.default_rng(42)
        for etype, cfg in MARK_CONFIG.items():
            for _ in range(50):
                m = HawkesProcess.sample_mark(etype, rng)
                assert cfg["lo"] - 0.01 <= m <= cfg["hi"] + 0.01, (
                    f"Type {etype}: mark {m} outside [{cfg['lo']}, {cfg['hi']}]"
                )

    def test_auto_sampled_mark(self):
        hp = HawkesProcess()
        hp.add_event(0.0, EVENT_ERROR)  # no mark → auto-sample
        assert len(hp.events) == 1
        m = hp.events[0].mark
        cfg = MARK_CONFIG[EVENT_ERROR]
        assert cfg["lo"] - 0.01 <= m <= cfg["hi"] + 0.01


class TestAddObservation:
    def test_success_observation(self):
        hp = HawkesProcess()
        hp.add_observation(1.0, tool_ok=True, has_user_msg=False, progress_delta=0.20, error_count_delta=0)
        # Should emit: success + tool (2 events)
        types = [e.event_type for e in hp.events]
        assert EVENT_SUCCESS in types
        assert EVENT_TOOL in types
        assert EVENT_ERROR not in types
        assert EVENT_USER not in types

    def test_error_observation(self):
        hp = HawkesProcess()
        hp.add_observation(1.0, tool_ok=False, has_user_msg=False, progress_delta=-0.05, error_count_delta=2)
        types = [e.event_type for e in hp.events]
        assert EVENT_ERROR in types
        assert EVENT_TOOL in types

    def test_user_msg_observation(self):
        hp = HawkesProcess()
        hp.add_observation(1.0, tool_ok=True, has_user_msg=True, progress_delta=0.05, error_count_delta=0)
        types = [e.event_type for e in hp.events]
        assert EVENT_USER in types


class TestSurprisal:
    def test_surprisal_positive(self):
        hp = HawkesProcess()
        s = hp.surprisal(EVENT_TOOL, t=0.0)
        assert s > 0.0

    def test_lower_surprisal_after_events(self):
        """More events → higher λ → lower surprisal."""
        hp = HawkesProcess()
        s_before = hp.surprisal(EVENT_TOOL, t=0.0)
        for i in range(5):
            hp.add_event(float(i), EVENT_TOOL, mark=1.0)
        s_after = hp.surprisal(EVENT_TOOL, t=5.0)
        assert s_after < s_before, (
            f"Surprisal should decrease after clustering: {s_before:.3f} → {s_after:.3f}"
        )


class TestDiagnostics:
    def test_intensity_trajectory(self):
        hp = HawkesProcess()
        hp.add_event(0.0, EVENT_TOOL, mark=1.0)
        traj = hp.get_intensity_trajectory(window=5.0, steps=20)
        assert traj.shape == (20, 4)
