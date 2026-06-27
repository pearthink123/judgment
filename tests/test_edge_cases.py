"""Edge case + robustness tests — NaN, extreme obs, numerical stability, divergence."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.engine import DecisionEngine, Decision
from core.hmm import HiddenMarkovModel, encode_observation
from core.hawkes import HawkesProcess
from core.cusum import CUSUMDetector
from core.hmm import N_STATES


# ---------------------------------------------------------------------------
# NaN / Inf in observation
# ---------------------------------------------------------------------------
class TestNaNHandling:
    def test_nan_progress_delta(self):
        """NaN progress should not crash the engine."""
        engine = DecisionEngine(seed=1)
        d = engine.step({
            "tool_ok": True,
            "progress_delta": float("nan"),
            "has_user_msg": False,
            "error_count_delta": 0,
        })
        assert d.action in {"continue", "correct", "escalate", "gather"}
        # Should not propagate NaN into belief
        for v in d.belief.values():
            assert not np.isnan(v)

    def test_inf_error_delta(self):
        engine = DecisionEngine(seed=1)
        d = engine.step({
            "tool_ok": True,
            "progress_delta": 0.1,
            "has_user_msg": False,
            "error_count_delta": 999999,
        })
        assert d.action in {"continue", "correct", "escalate", "gather"}

    def test_all_nan_observations(self):
        """Repeated NaN should not produce NaN belief."""
        engine = DecisionEngine(seed=1)
        for _ in range(10):
            d = engine.step({
                "tool_ok": True,
                "progress_delta": float("nan"),
                "has_user_msg": False,
                "error_count_delta": 0,
            })
        for v in d.belief.values():
            assert 0 <= v <= 1, f"Belief out of range: {v}"

    def test_negative_error_count(self):
        """Negative errors (impossible, but defensive)."""
        engine = DecisionEngine(seed=1)
        d = engine.step({
            "tool_ok": True,
            "progress_delta": 0.1,
            "has_user_msg": False,
            "error_count_delta": -5,
        })
        # Should not crash — engine handles via int() cast
        assert d.action in {"continue", "correct", "escalate", "gather"}


# ---------------------------------------------------------------------------
# Extreme belief states
# ---------------------------------------------------------------------------
class TestExtremeBelief:
    def test_100pct_healthy_consistency(self):
        """If agent is 100% healthy for N steps, belief should converge."""
        engine = DecisionEngine(seed=1)
        beliefs = []
        for _ in range(15):
            d = engine.step({
                "tool_ok": True,
                "progress_delta": 0.20,
                "has_user_msg": False,
                "error_count_delta": 0,
            })
            beliefs.append(d.belief["healthy"])
        # Should converge above 0.95
        assert beliefs[-1] > 0.95

    def test_100pct_broken_convergence(self):
        """If agent is 100% failing for N steps, broken should dominate."""
        engine = DecisionEngine(seed=1)
        beliefs = []
        for _ in range(15):
            d = engine.step({
                "tool_ok": False,
                "progress_delta": -0.10,
                "has_user_msg": False,
                "error_count_delta": 3,
            })
            beliefs.append(d.belief["broken"])
        assert beliefs[-1] > 0.50

    def test_oscillation_handling(self):
        """Alternating success/failure every step is extreme — engine may flip
        frequently, which is correct because each observation genuinely changes
        the belief. The key assertion is that it doesn't crash or diverge."""
        engine = DecisionEngine(seed=1)
        actions = []
        for i in range(20):
            ok = i % 2 == 0
            d = engine.step({
                "tool_ok": ok,
                "progress_delta": 0.15 if ok else -0.05,
                "has_user_msg": False,
                "error_count_delta": 0 if ok else 1,
            })
            actions.append(d.action)
        # With genuine alternation, flipping is correct behavior.
        # Hysteresis prevents single-step jitter, but true alternation
        # should produce action changes.
        assert all(a in {"continue", "correct", "escalate", "gather"} for a in actions)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
class TestNumericalStability:
    def test_hmm_log_space_no_underflow(self):
        """Forward algorithm should handle very unbalanced observations."""
        hmm = HiddenMarkovModel()
        # Extreme observation: fail + neg progress + user message + high errors
        obs = encode_observation(False, -0.50, True, 100)
        b = hmm.forward_step(obs)
        # Should be valid probabilities
        assert abs(b.sum() - 1.0) < 1e-6
        assert np.all(b >= 0)
        assert not np.any(np.isnan(b))

    def test_hawkes_many_events(self):
        """Hawkes with many events should not overflow/underflow."""
        hp = HawkesProcess()
        for i in range(500):
            hp.add_event(float(i), i % 4, mark=1.0 if i % 2 == 0 else 1.5)
        lam = hp.intensity(500.0)
        assert lam.shape == (4,)
        assert np.all(lam >= 0)
        assert np.all(np.isfinite(lam))

    def test_hawkes_very_old_events(self):
        """Very old events should contribute almost nothing to intensity."""
        hp = HawkesProcess()
        hp.add_event(0.0, 0, mark=2.0)  # event at t=0
        lam_recent = hp.intensity(1.0)
        lam_old = hp.intensity(100.0)
        # With beta=1.0, exp(-100) ≈ 0. The old event should be negligible
        assert abs(lam_old[0] - hp.mu[0]) < 0.01

    def test_cusum_extreme_surprisal(self):
        """CUSUM with extreme surprisal values should not overflow."""
        c = CUSUMDetector(h=10.0)
        for s_healthy in [1000.0, -1000.0, 0.0, 1e-10, 1e10]:
            result = c.update(
                surprisal_healthy=s_healthy,
                hawkes_intensity=1.0,
                surprisal_degraded=1.0,
            )
            assert np.isfinite(result["S"])


# ---------------------------------------------------------------------------
# Rapid state changes
# ---------------------------------------------------------------------------
class TestRapidStateChanges:
    def test_instant_degradation(self):
        """3 perfect steps → 1 catastrophic step → engine should react."""
        engine = DecisionEngine(seed=1)
        # 3 perfect steps
        for _ in range(3):
            engine.step({
                "tool_ok": True, "progress_delta": 0.20,
                "has_user_msg": False, "error_count_delta": 0,
            })
        # 1 catastrophic step
        d = engine.step({
            "tool_ok": False, "progress_delta": -0.50,
            "has_user_msg": False, "error_count_delta": 5,
        })
        # Should at minimum detect anomaly — not necessarily escalate in 1 step
        # but the belief should shift dramatically
        assert d.belief["healthy"] < 0.20 or d.anomaly

    def test_recovery_after_cleanup(self):
        """After escalate → human fixes → engine should recover."""
        engine = DecisionEngine(seed=1)
        # Break it
        for _ in range(8):
            engine.step({
                "tool_ok": False, "progress_delta": -0.05,
                "has_user_msg": False, "error_count_delta": 2,
            })
        # Fix it
        healthy_beliefs = []
        for _ in range(10):
            d = engine.step({
                "tool_ok": True, "progress_delta": 0.20,
                "has_user_msg": True, "error_count_delta": 0,
            })
            healthy_beliefs.append(d.belief["healthy"])
        # Should eventually recover
        assert healthy_beliefs[-1] > healthy_beliefs[0]


# ---------------------------------------------------------------------------
# Content signal edge cases
# ---------------------------------------------------------------------------
class TestContentSignalEdgeCases:
    def test_empty_llm_text(self):
        from core.content_signals import ContentSignalExtractor
        e = ContentSignalExtractor()
        # Empty string
        result = e.extract("")
        assert 4 in result and 5 in result and 6 in result

    def test_very_long_llm_text(self):
        from core.content_signals import ContentSignalExtractor
        e = ContentSignalExtractor()
        text = "the quick brown fox " * 2000  # ~8000 tokens
        result = e.extract(text)
        assert all(k in result for k in [4, 5, 6])

    def test_unicode_llm_text(self):
        from core.content_signals import ContentSignalExtractor
        e = ContentSignalExtractor()
        text = "你好世界 " * 50 + "error wrong incorrect sorry"
        result = e.extract(text)
        assert all(k in result for k in [4, 5, 6])


# ---------------------------------------------------------------------------
# Engine reset and reproducibility
# ---------------------------------------------------------------------------
class TestEngineResetReproducibility:
    def test_reset_restores_clean_state(self):
        engine = DecisionEngine(seed=1)
        for _ in range(10):
            engine.step({
                "tool_ok": True, "progress_delta": 0.1,
                "has_user_msg": False, "error_count_delta": 0,
            })
        assert engine.step_count > 0
        engine.reset()
        assert engine.step_count == 0
        assert len(engine.decision_log) == 0
        assert engine.prev_action is None

    def test_save_load_preserves_belief(self):
        """After save/load, next step should produce same belief."""
        engine = DecisionEngine(seed=42)
        # 5 healthy + 3 error steps
        for _ in range(5):
            engine.step({"tool_ok": True, "progress_delta": 0.12, "error_count_delta": 0})
        for _ in range(3):
            engine.step({"tool_ok": False, "progress_delta": -0.05, "error_count_delta": 2})

        snap = engine.save_state()
        engine_restored = DecisionEngine(seed=42)
        engine_restored.load_state(snap)

        obs = {"tool_ok": True, "progress_delta": 0.1, "error_count_delta": 0}
        d_orig = engine.step(obs)
        d_restored = engine_restored.step(obs)
        # Belief must be identical — that's what the HMM state captures
        for k in ["healthy", "degraded", "broken"]:
            assert abs(d_orig.belief[k] - d_restored.belief[k]) < 0.01, (
                f"{k}: {d_orig.belief[k]} vs {d_restored.belief[k]}"
            )

    def test_save_load_preserves_prev_action(self):
        engine = DecisionEngine(seed=42)
        for _ in range(10):
            engine.step({"tool_ok": True, "progress_delta": 0.12, "error_count_delta": 0})
        snap = engine.save_state()
        engine2 = DecisionEngine(seed=42)
        engine2.load_state(snap)
        assert engine2.prev_action == engine.prev_action
        assert engine2.step_count == engine.step_count

    def test_same_seed_reproduces(self):
        e1 = DecisionEngine(seed=42)
        e2 = DecisionEngine(seed=42)
        obs = [
            {"tool_ok": True, "progress_delta": 0.15, "has_user_msg": False, "error_count_delta": 0},
            {"tool_ok": False, "progress_delta": -0.05, "has_user_msg": False, "error_count_delta": 1},
            {"tool_ok": True, "progress_delta": 0.10, "has_user_msg": False, "error_count_delta": 0},
        ]
        for o in obs:
            d1 = e1.step(o)
            d2 = e2.step(o)
        # Final beliefs should be identical
        assert d1.belief == d2.belief
        assert d1.action == d2.action
