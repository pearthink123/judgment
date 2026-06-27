"""Tests for custom adapter (integration/custom.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from integration.custom import (
    wrap_step, judgment_guard, quick_check,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)
from core.engine import DecisionEngine


class TestWrapStep:
    def test_wraps_and_returns_decision(self):
        engine = DecisionEngine(seed=1)

        @wrap_step(engine)
        def my_step(state):
            return {
                "tool_ok": True,
                "progress_delta": 0.15,
                "has_user_msg": False,
                "error_count_delta": 0,
            }

        result, decision = my_step({"task": "test"})
        assert result["tool_ok"] is True
        assert decision.action in {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}
        assert decision.belief["healthy"] > 0.5

    def test_preserves_function_args(self):
        engine = DecisionEngine(seed=1)

        @wrap_step(engine)
        def add(a, b):
            return {"tool_ok": True, "progress_delta": 0.1, "error_count_delta": 0}

        result, decision = add(3, 4)
        assert result["tool_ok"] is True

    def test_with_extractor(self):
        engine = DecisionEngine(seed=1)

        def my_extractor(result):
            return {
                "tool_ok": result["success"],
                "progress_delta": result["delta"],
                "has_user_msg": False,
                "error_count_delta": 0,
            }

        @wrap_step(engine, extractor=my_extractor)
        def my_step(state):
            return {"success": False, "delta": -0.05}

        result, decision = my_step({})
        assert decision.belief["healthy"] < 0.95  # should register the failure

    def test_error_streak_detected(self):
        engine = DecisionEngine(seed=2)

        @wrap_step(engine)
        def my_step():
            return {"tool_ok": False, "progress_delta": -0.05, "error_count_delta": 1}

        actions = []
        for _ in range(10):
            _, decision = my_step()
            actions.append(decision.action)

        # Should escalate or correct after sustained errors
        assert "escalate" in actions or "correct" in actions


class TestJudgmentGuard:
    def test_context_manager_yields_checker(self):
        engine = DecisionEngine(seed=1)

        with judgment_guard(engine) as check:
            action = check({
                "tool_ok": True,
                "progress_delta": 0.15,
                "has_user_msg": False,
                "error_count_delta": 0,
            })
            assert action == ACTION_CONTINUE
            assert check.last_decision is not None
            assert check.last_decision.belief["healthy"] > 0.5

    def test_advice_on_error(self):
        engine = DecisionEngine(seed=2)

        with judgment_guard(engine) as check:
            # Build healthy context then inject sustained errors
            for _ in range(3):
                check({"tool_ok": True, "progress_delta": 0.1, "has_user_msg": False, "error_count_delta": 0})
            # Sustained errors — should trigger correction or escalation
            actions = []
            for _ in range(5):
                a = check({"tool_ok": False, "progress_delta": -0.06, "has_user_msg": False, "error_count_delta": 2})
                actions.append(a)
            assert any(a in {ACTION_CORRECT, ACTION_ESCALATE} for a in actions)

    def test_default_engine_created(self):
        with judgment_guard() as check:
            action = check({
                "tool_ok": True, "progress_delta": 0.1,
                "has_user_msg": False, "error_count_delta": 0,
            })
            assert action in {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}


class TestQuickCheck:
    def test_healthy_continue(self):
        engine = DecisionEngine(seed=1)
        action = quick_check(engine, tool_ok=True, progress_delta=0.15)
        assert action == ACTION_CONTINUE

    def test_repeated_errors_detected(self):
        engine = DecisionEngine(seed=3)
        actions = []
        for _ in range(10):
            a = quick_check(engine, tool_ok=False, progress_delta=-0.05, error_count_delta=2)
            actions.append(a)
        assert "escalate" in actions or "correct" in actions
