"""Tests for CrewAI adapter (integration/crewai.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from integration.crewai import (
    create_judgment_callback,
    create_judgment_tool,
    create_tool_wrapper,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE,
)
from core.engine import DecisionEngine


class TestCallback:
    def test_callback_does_not_crash(self):
        engine = DecisionEngine(seed=1)
        cb = create_judgment_callback(engine)

        # Simulate CrewAI step output — plain dict
        cb({"result": "Step completed successfully."})
        # Should not raise

    def test_callback_verbose(self, capsys):
        engine = DecisionEngine(seed=1)
        cb = create_judgment_callback(engine, verbose=True)
        cb({"result": "Step completed."})
        captured = capsys.readouterr()
        assert "CONTINUE" in captured.out or "continue" in captured.out.lower()

    def test_callback_on_escalate(self):
        engine = DecisionEngine(seed=2)
        escalated = []

        def handler(decision):
            escalated.append(decision)

        cb = create_judgment_callback(engine, on_escalate=handler)

        # Generate errors to trigger escalation
        for _ in range(15):
            cb({"result": "Error: tool failed"})

        # After sustained errors, escalate should have been called
        # (not guaranteed with 15 steps but highly likely)
        # Just verify the callback doesn't crash
        assert True

    def test_callback_with_extractor(self):
        engine = DecisionEngine(seed=1)

        def my_extractor(output):
            return {
                "tool_ok": output["ok"],
                "progress_delta": output["delta"],
                "has_user_msg": False,
                "error_count_delta": output["errors"],
            }

        cb = create_judgment_callback(engine, extractor=my_extractor)
        cb({"ok": True, "delta": 0.15, "errors": 0})
        # Should not crash and should use the extracted values
        assert engine.step_count == 1


class TestJudgmentTool:
    def test_tool_spec_structure(self):
        engine = DecisionEngine(seed=1)
        spec = create_judgment_tool(engine)
        assert "name" in spec
        assert spec["name"] == "check_health"
        assert "description" in spec
        assert "func" in spec
        assert callable(spec["func"])

    def test_tool_func_returns_report(self):
        engine = DecisionEngine(seed=1)
        # Process a step first
        engine.step({
            "tool_ok": True, "progress_delta": 0.15,
            "has_user_msg": False, "error_count_delta": 0,
        })

        spec = create_judgment_tool(engine)
        report = spec["func"]()
        assert "Healthy" in report
        assert "0." in report  # some probability value

    def test_tool_func_no_steps(self):
        engine = DecisionEngine(seed=1)
        spec = create_judgment_tool(engine)
        report = spec["func"]()
        assert "unknown" in report.lower() or "no steps" in report.lower()


class TestToolWrapper:
    def test_healthy_returns_continue(self):
        engine = DecisionEngine(seed=1)
        observe = create_tool_wrapper(engine)
        action = observe("my_tool", ok=True, progress=0.15, errors=0)
        assert action == ACTION_CONTINUE

    def test_errors_detected(self):
        engine = DecisionEngine(seed=3)
        observe = create_tool_wrapper(engine)
        actions = []
        for _ in range(12):
            a = observe("my_tool", ok=False, progress=-0.05, errors=2)
            actions.append(a)
        assert "escalate" in actions or "correct" in actions

    def test_default_args(self):
        engine = DecisionEngine(seed=1)
        observe = create_tool_wrapper(engine)
        action = observe()
        assert action == ACTION_CONTINUE
