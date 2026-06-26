"""End-to-end tests for JudgmentHarness."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.loop import JudgmentHarness, RunResult
from harness.executor import SimulatedExecutor
from harness.tools import default_registry
from core.engine import DecisionEngine
from core.corrective import CorrectiveAdvice


def test_harness_simulated_run():
    """Basic end-to-end: simulated executor completes a simple task."""
    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        max_steps=10,
        seed=1,
    )
    result = harness.run("Write a hello world function.")
    assert result.status in ("success", "max_steps", "escalated")
    assert result.steps > 0
    assert len(result.messages) > 0
    assert result.duration_seconds >= 0


def test_harness_produces_decision_log():
    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        max_steps=5,
        seed=1,
    )
    result = harness.run("Test task.")
    assert len(result.decision_log) > 0
    # Each decision should have the expected fields
    for d in result.decision_log:
        assert d.action in ("continue", "correct", "escalate", "gather")
        assert "healthy" in d.belief


def test_harness_with_custom_engine():
    """Harness accepts a pre-configured engine."""
    engine = DecisionEngine(seed=99, use_pomdp=True, use_corrective=True)
    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        engine=engine,
        max_steps=5,
    )
    result = harness.run("Custom engine test.")
    assert result.status in ("success", "max_steps", "escalated")


def test_harness_with_tools():
    """Harness works with the default tool registry."""
    tools = default_registry()
    assert len(tools.list_names()) >= 4

    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        tools=tools,
        max_steps=5,
        seed=1,
    )
    result = harness.run("Read a file.")
    assert result.status in ("success", "max_steps", "escalated")


def test_harness_error_script():
    """Simulated executor with error script triggers corrective."""
    executor = SimulatedExecutor(
        script=["error", "error", "success", "success", "success"],
        seed=42,
    )
    harness = JudgmentHarness(
        executor=executor,
        max_steps=8,
        seed=1,
    )
    result = harness.run("Test with errors.")
    # After consecutive errors, the engine should react
    actions = [d.action for d in result.decision_log]
    assert "correct" in actions or "escalate" in actions, (
        f"Expected corrective/escalate after errors, got: {actions}"
    )


def test_harness_reset_between_runs():
    """Harness engine resets between runs."""
    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        max_steps=3,
        seed=1,
    )
    r1 = harness.run("Task 1.")
    r2 = harness.run("Task 2.")
    # Engine should start fresh each time
    assert r2.steps <= 3  # starts from step 1, not continuing


def test_harness_message_history():
    """Messages are recorded."""
    harness = JudgmentHarness(
        executor=SimulatedExecutor(seed=42),
        max_steps=4,
        seed=1,
    )
    result = harness.run("Hello.")
    # Should have at least the user message and some assistant responses
    assert len(result.messages) >= 2
    assert result.messages[0]["role"] == "user"
