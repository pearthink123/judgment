"""Tests for content quality signal extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.content_signals import (
    ContentSignalExtractor,
    tokenize,
    NEGATION_WORDS,
    LENGTH_BINS,
    NOVELTY_BINS,
    NEGATION_BINS,
)


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello world") == ["hello", "world"]

    def test_numbers(self):
        assert tokenize("x = 42 + 1") == ["x", "42", "1"]

    def test_empty(self):
        assert tokenize("") == []

    def test_punctuation_ignored(self):
        assert tokenize("hello, world!") == ["hello", "world"]


class TestNegationWords:
    def test_known_words(self):
        assert "wrong" in NEGATION_WORDS
        assert "incorrect" in NEGATION_WORDS
        assert "however" in NEGATION_WORDS
        assert "not" in NEGATION_WORDS


class TestContentSignalExtractor:
    def test_empty_text_returns_normal(self):
        e = ContentSignalExtractor()
        result = e.extract(None)
        assert result[e.DIM_LENGTH] == LENGTH_BINS["normal"]
        assert result[e.DIM_NOVELTY] == NOVELTY_BINS["normal"]
        assert result[e.DIM_NEGATION] == NEGATION_BINS["normal"]

    def test_short_text_no_length_anomaly_without_history(self):
        e = ContentSignalExtractor()
        # First call — no history, so defaults to normal
        result = e.extract("hello")
        assert result[e.DIM_LENGTH] == LENGTH_BINS["normal"]

    def test_very_short_after_long_history(self):
        e = ContentSignalExtractor()
        # Build history of long texts
        for _ in range(8):
            e.extract("x " * 200)  # 200 tokens
        # Now a very short text
        result = e.extract("hi")
        assert result[e.DIM_LENGTH] == LENGTH_BINS["low"]

    def test_repetitive_text(self):
        e = ContentSignalExtractor()
        result = e.extract("a a a a a a a a a a")  # 10 tokens, 1 unique
        assert result[e.DIM_NOVELTY] == NOVELTY_BINS["repetitive"]

    def test_fresh_text(self):
        e = ContentSignalExtractor()
        unique_words = " ".join(str(i) for i in range(30))
        result = e.extract(unique_words)
        assert result[e.DIM_NOVELTY] == NOVELTY_BINS["fresh"]

    def test_normal_text(self):
        e = ContentSignalExtractor()
        text = "The function processes the input data and returns the processed output result."
        result = e.extract(text)
        assert result[e.DIM_NOVELTY] == NOVELTY_BINS["normal"]

    def test_negation_normal(self):
        e = ContentSignalExtractor()
        result = e.extract("The test passes successfully.")
        assert result[e.DIM_NEGATION] == NEGATION_BINS["normal"]

    def test_negation_elevated(self):
        e = ContentSignalExtractor()
        # Many negation/self-correction words
        result = e.extract(
            "Sorry, this is wrong. The correct approach is not what I said. "
            "Actually, the error was in the previous step. However, this is "
            "incorrect too. I apologize for the mistake."
        )
        assert result[e.DIM_NEGATION] == NEGATION_BINS["elevated"]

    def test_return_keys(self):
        e = ContentSignalExtractor()
        result = e.extract("test text here")
        assert e.DIM_LENGTH in result
        assert e.DIM_NOVELTY in result
        assert e.DIM_NEGATION in result

    def test_reset(self):
        e = ContentSignalExtractor()
        for _ in range(5):
            e.extract("x " * 100)
        assert e.recent_mean_length > 50
        e.reset()
        assert e.recent_mean_length == 0.0

    def test_content_signals_integrate_with_hmm(self):
        """Signals from extractor can be passed to encode_observation."""
        from core.hmm import encode_observation

        e = ContentSignalExtractor()
        content = e.extract("some normal text here for testing purposes")
        obs = encode_observation(True, 0.15, False, 0, content_signals=content)

        # Should include both structural and content dims
        assert 0 in obs  # tool_ok
        assert 1 in obs  # progress
        assert 4 in obs  # length
        assert 5 in obs  # novelty
        assert 6 in obs  # negation

    def test_engine_with_content_signals(self):
        """Engine with content signals enabled runs without error."""
        from core.engine import DecisionEngine

        engine = DecisionEngine(use_content_signals=True, seed=42)
        obs = {
            "tool_ok": True,
            "progress_delta": 0.15,
            "has_user_msg": False,
            "error_count_delta": 0,
            "llm_text": "The agent processes the data and returns a valid result.",
        }
        decision = engine.step(obs)
        assert decision.action in {"continue", "correct", "escalate", "gather"}
        # Content signals should be present
        assert decision.content_signals is not None
        assert "length_z_cat" in decision.content_signals

    def test_engine_with_pomcp(self):
        """Engine with POMCP solver runs without error."""
        from core.engine import DecisionEngine

        engine = DecisionEngine(use_pomcp=True, use_fast_pomcp=False, seed=42)
        obs = {
            "tool_ok": True,
            "progress_delta": 0.15,
            "has_user_msg": False,
            "error_count_delta": 0,
        }
        decision = engine.step(obs)
        assert decision.action in {"continue", "correct", "escalate", "gather"}
        assert decision.layer_diagnostics["solver"] == "pomcp"

    def test_engine_dual_mode(self):
        """Engine with both POMCP and content signals."""
        from core.engine import DecisionEngine

        engine = DecisionEngine(
            use_pomcp=True, use_fast_pomcp=False, use_content_signals=True,
            pomcp_n_simulations=300, seed=42,
        )
        solvers_seen = set()
        for _ in range(5):
            obs = {
                "tool_ok": True,
                "progress_delta": 0.12,
                "has_user_msg": False,
                "error_count_delta": 0,
                "llm_text": "Step executed successfully with correct output.",
            }
            decision = engine.step(obs)
            assert decision.content_signals is not None
            solvers_seen.add(decision.layer_diagnostics["solver"])
        # First step uses POMCP, subsequent steps use threshold fast-path
        assert "pomcp" in solvers_seen, f"Expected at least one pomcp step, got {solvers_seen}"
