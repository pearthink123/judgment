"""Tests for harness integration adapters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from integration.base import (
    BaseAdapter, default_extractor, RouteTargets,
    KEY_ACTION, KEY_HEALTH, KEY_RATIONALE, KEY_ADVICE,
)
from integration.langgraph import (
    create_judgment_node, create_judgment_router,
)
from core.engine import DecisionEngine, ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
def _healthy_state() -> dict:
    return {
        "messages": [{"role": "user", "content": "test"}],
        "tool_ok": True,
        "progress_delta": 0.15,
        "has_user_msg": False,
        "error_count_delta": 0,
    }


def _error_state() -> dict:
    return {
        "messages": [{"role": "user", "content": "test"}],
        "tool_ok": False,
        "progress_delta": -0.05,
        "has_user_msg": False,
        "error_count_delta": 2,
    }


# ---------------------------------------------------------------------------
# BaseAdapter
# ---------------------------------------------------------------------------
class TestBaseAdapter:
    def test_extract_default(self):
        adapter = BaseAdapter()
        obs = adapter.extract(_healthy_state())
        assert obs["tool_ok"] is True
        assert obs["progress_delta"] == 0.15

    def test_extract_custom(self):
        def my_extractor(state):
            return {
                "tool_ok": state.get("custom_ok", False),
                "progress_delta": state.get("custom_progress", 0.0),
                "has_user_msg": False,
                "error_count_delta": 0,
            }
        adapter = BaseAdapter(extractor=my_extractor)
        obs = adapter.extract({"custom_ok": True, "custom_progress": 0.5})
        assert obs["tool_ok"] is True
        assert obs["progress_delta"] == 0.5

    def test_step_returns_decision(self):
        adapter = BaseAdapter(engine=DecisionEngine(seed=1))
        decision = adapter.step(_healthy_state())
        assert decision.action in {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}
        assert "healthy" in decision.belief

    def test_inject_annotates_keys(self):
        adapter = BaseAdapter(engine=DecisionEngine(seed=1))
        decision = adapter.step(_healthy_state())
        state = adapter.inject(_healthy_state(), decision)
        assert KEY_ACTION in state
        assert KEY_HEALTH in state
        assert KEY_RATIONALE in state

    def test_inject_preserves_messages(self):
        adapter = BaseAdapter(engine=DecisionEngine(seed=1))
        decision = adapter.step(_healthy_state())
        state = adapter.inject(_healthy_state(), decision)
        assert len(state["messages"]) >= 1

    def test_route_maps_actions(self):
        adapter = BaseAdapter(routes=RouteTargets(
            continue_target="agent_node",
            correct_target="agent_node",
            escalate_target="human_node",
            gather_target="tools_node",
        ))
        decision = type("D", (), {"action": ACTION_ESCALATE})()
        assert adapter.route(decision) == "human_node"

    def test_process_full_pipeline(self):
        adapter = BaseAdapter(engine=DecisionEngine(seed=1))
        new_state, target = adapter.process(_healthy_state())
        assert target in {"agent", "human", "tools"}
        assert KEY_ACTION in new_state


class TestDefaultExtractor:
    def test_standard_keys(self):
        obs = default_extractor({
            "tool_ok": False,
            "progress_delta": 0.25,
            "has_user_msg": True,
            "error_count_delta": 1,
        })
        assert obs["tool_ok"] is False
        assert obs["progress_delta"] == 0.25
        assert obs["has_user_msg"] is True
        assert obs["error_count_delta"] == 1

    def test_fallback_keys(self):
        """Should fall back to common alternative key names."""
        obs = default_extractor({
            "tool_success": False,
            "progress": 0.30,
            "user_message": True,
            "errors_this_step": 2,
        })
        assert obs["tool_ok"] is False
        assert obs["progress_delta"] == 0.30
        assert obs["has_user_msg"] is True
        assert obs["error_count_delta"] == 2


# ---------------------------------------------------------------------------
# LangGraph node / router
# ---------------------------------------------------------------------------
class TestLangGraphNode:
    def test_node_returns_dict(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        result = node(_healthy_state())
        assert isinstance(result, dict)
        assert KEY_ACTION in result
        assert result[KEY_ACTION] in {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}

    def test_node_preserves_original_keys(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        state = _healthy_state()
        state["custom_field"] = 42
        result = node(state)
        assert result["custom_field"] == 42

    def test_node_without_annotation(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine, annotate_state=False)
        result = node(_healthy_state())
        assert KEY_ACTION not in result

    def test_node_custom_extractor(self):
        engine = DecisionEngine(seed=1)
        def my_ext(state):
            return {"tool_ok": True, "progress_delta": 0.5, "has_user_msg": False, "error_count_delta": 0}
        node = create_judgment_node(engine, extractor=my_ext)
        result = node(_healthy_state())
        assert KEY_ACTION in result


class TestLangGraphRouter:
    def test_router_returns_string(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state = node(_healthy_state())
        target = router(state)
        assert isinstance(target, str)
        assert target in {"agent", "human", "tools"}

    def test_router_continues_on_healthy(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        # Multiple healthy steps should all route to agent
        state = _healthy_state()
        for _ in range(5):
            state = node(state)
            target = router(state)
            assert target in {"agent", "tools"}  # not human

    def test_router_custom_map(self):
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine, route_map={
            "continue": "my_agent",
            "correct": "my_agent",
            "escalate": "my_human",
            "gather": "my_tools",
        })
        state = node(_healthy_state())
        target = router(state)
        assert target in {"my_agent", "my_human", "my_tools"}

    def test_router_fallback_without_node(self):
        """Router should work even if judgment node didn't run."""
        engine = DecisionEngine(seed=1)
        router = create_judgment_router(engine)
        # State without __judgment_action — router falls back to inline engine
        target = router(_healthy_state())
        assert isinstance(target, str)


# ---------------------------------------------------------------------------
# Escalation path
# ---------------------------------------------------------------------------
class TestEscalationPath:
    def test_error_cascade_triggers_escalate(self):
        """Simulate a full LangGraph-style loop where errors trigger escalation."""
        engine = DecisionEngine(seed=2)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state = _healthy_state()
        escalated = False

        for step_i in range(15):
            if step_i < 3:
                # Normal steps
                state["tool_ok"] = True
                state["progress_delta"] = 0.12
                state["error_count_delta"] = 0
            elif step_i < 6:
                # Mild errors
                state["tool_ok"] = False
                state["progress_delta"] = 0.0
                state["error_count_delta"] = 0
            else:
                # Severe errors
                state["tool_ok"] = False
                state["progress_delta"] = -0.06
                state["error_count_delta"] = 2

            state = node(state)
            target = router(state)
            if target in {"human", "my_human"}:
                escalated = True
                break

        # With persistent errors, escalation should happen
        assert escalated, "Engine should escalate after persistent errors"
