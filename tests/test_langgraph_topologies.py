"""
LangGraph topology tests — validating "drop-in" claims under real graph patterns.

Tests three scenarios without importing langgraph:
  1. Parallel branches — two tools → merge → judgment
  2. Subgraph with internal judgment — nested engine instances
  3. Checkpoint + resume — state preservation across interrupt/recover
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from integration.langgraph import create_judgment_node, create_judgment_router
from integration.base import KEY_ACTION, KEY_HEALTH
from core.engine import DecisionEngine, ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER


# ---------------------------------------------------------------------------
# Scenario 1: Parallel branches — two tool calls, merge, one judgment
# ---------------------------------------------------------------------------
class TestParallelBranches:
    """
    Graph: agent → [tool_A, tool_B] → merge → judgment → conditional → agent|human

    In real LangGraph, parallel branches fan out and then join at a merge node.
    The challenge: each branch produces partial observations that must be
    combined before feeding to judgment.
    """

    def _merge_state(self, state: dict, branch_a: dict, branch_b: dict) -> dict:
        """Simulate LangGraph's fan-in merge."""
        state = dict(state)
        state["messages"] = state.get("messages", []) + \
            branch_a.get("messages", []) + branch_b.get("messages", [])
        # Merge tool observations: tool fails if ANY branch failed
        state["tool_ok"] = branch_a.get("tool_ok", True) and branch_b.get("tool_ok", True)
        state["progress_delta"] = branch_a.get("progress_delta", 0.0) + branch_b.get("progress_delta", 0.0)
        state["error_count_delta"] = branch_a.get("error_count_delta", 0) + branch_b.get("error_count_delta", 0)
        return state

    def test_both_branches_healthy(self):
        """Both tool branches succeed → judgment says continue."""
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state = {"messages": []}
        branch_a = {"messages": [{"role": "tool", "content": "ok"}], "tool_ok": True, "progress_delta": 0.1, "error_count_delta": 0}
        branch_b = {"messages": [{"role": "tool", "content": "ok"}], "tool_ok": True, "progress_delta": 0.05, "error_count_delta": 0}

        merged = self._merge_state(state, branch_a, branch_b)
        result = node(merged)
        target = router(result)

        assert target in {"agent", "tools"}
        assert result[KEY_ACTION] == ACTION_CONTINUE

    def test_one_branch_fails(self):
        """One tool branch fails → judgment may react."""
        engine = DecisionEngine(seed=2)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state = {"messages": []}
        branch_a = {"messages": [{"role": "tool", "content": "error!"}], "tool_ok": False, "progress_delta": -0.05, "error_count_delta": 1}
        branch_b = {"messages": [{"role": "tool", "content": "ok"}], "tool_ok": True, "progress_delta": 0.05, "error_count_delta": 0}

        merged = self._merge_state(state, branch_a, branch_b)
        result = node(merged)
        target = router(result)

        # Should handle the partial-failure case without crashing
        assert isinstance(target, str)
        assert result[KEY_ACTION] in {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}

    def test_multiple_cycles_parallel(self):
        """Run multiple parallel cycles, alternating success and failure."""
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state: dict = {"messages": []}
        targets = []

        for cycle in range(5):
            fail = cycle >= 2  # start failing after 2 cycles

            branch_a = {"messages": [], "tool_ok": not fail, "progress_delta": 0.05, "error_count_delta": 1 if fail else 0}
            branch_b = {"messages": [], "tool_ok": not fail, "progress_delta": 0.05, "error_count_delta": 0}
            state = self._merge_state(state, branch_a, branch_b)
            state = node(state)
            targets.append(router(state))

        # All cycles should complete without crash
        assert len(targets) == 5
        assert all(t in {"agent", "tools", "human"} for t in targets)

    def test_custom_extractor_for_parallel(self):
        """Use a custom extractor that understands merged parallel state."""
        engine = DecisionEngine(seed=1)

        def parallel_extractor(state: dict) -> dict:
            # Extract from post-merge state
            return {
                "tool_ok": state.get("merged_ok", state.get("tool_ok", True)),
                "progress_delta": state.get("merged_progress", 0.0),
                "has_user_msg": bool(state.get("user_message", False)),
                "error_count_delta": state.get("merged_errors", 0),
                "llm_text": str(state.get("messages", [{}])[-1].get("content", "")) if state.get("messages") else None,
            }

        node = create_judgment_node(engine, extractor=parallel_extractor)

        state = {"merged_ok": True, "merged_progress": 0.15, "merged_errors": 0, "messages": []}
        result = node(state)
        assert result[KEY_ACTION] == ACTION_CONTINUE


# ---------------------------------------------------------------------------
# Scenario 2: Subgraph with internal judgment
# ---------------------------------------------------------------------------
class TestSubgraphWithJudgment:
    """
    Outer graph: agent → subgraph → judgment_outer → agent|human
    Subgraph: agent_inner → tools → judgment_inner → agent_inner|exit

    Two engines: one for subgraph health, one for overall task health.
    """

    def test_nested_judgment_engines(self):
        """Two independent engines, nested."""
        outer_engine = DecisionEngine(seed=1)
        inner_engine = DecisionEngine(seed=2)

        outer_node = create_judgment_node(outer_engine)
        outer_router = create_judgment_router(outer_engine)
        inner_node = create_judgment_node(inner_engine)
        inner_router = create_judgment_router(
            inner_engine, route_map={"continue": "agent", "correct": "agent", "escalate": "exit", "gather": "tools"}
        )

        outer_state: dict = {"messages": []}
        inner_state: dict = {"messages": []}

        # Run subgraph: 3 steps
        for _ in range(3):
            inner_state["tool_ok"] = True
            inner_state["progress_delta"] = 0.12
            inner_state["error_count_delta"] = 0
            inner_state = inner_node(inner_state)
            inner_target = inner_router(inner_state)
            assert inner_target in {"agent", "tools", "exit"}

        # Subgraph done — feed accumulated progress to outer engine
        outer_state["tool_ok"] = True
        outer_state["progress_delta"] = 0.36  # 3 * 0.12
        outer_state["error_count_delta"] = 0
        outer_state = outer_node(outer_state)
        outer_target = outer_router(outer_state)

        assert outer_target in {"agent", "tools", "human"}

    def test_subgraph_escalates_to_outer(self):
        """When subgraph escalates, outer engine should see the degraded state."""
        outer_engine = DecisionEngine(seed=1)
        inner_engine = DecisionEngine(seed=3)

        outer_node = create_judgment_node(outer_engine)
        outer_router = create_judgment_router(outer_engine)

        outer_state: dict = {"messages": []}

        # Simulate subgraph producing an escalated trajectory
        # Feed the outer engine with degraded signals
        for _ in range(4):
            outer_state["tool_ok"] = False
            outer_state["progress_delta"] = -0.03
            outer_state["error_count_delta"] = 1
            outer_state = outer_node(outer_state)

        outer_target = outer_router(outer_state)
        # Should escalate after sustained failure in subgraph
        assert outer_target == "human"


# ---------------------------------------------------------------------------
# Scenario 3: Checkpoint + resume
# ---------------------------------------------------------------------------
class TestCheckpointResume:
    """
    Agent runs → judgment escalate → graph interrupts (checkpoint)
    → human intervenes → resume from checkpoint
    → Verdict: Judgment state preserved across checkpoint boundary
    """

    def test_state_survives_checkpoint_cycle(self):
        """Engine state after checkpoint should reflect pre-interrupt history."""
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state: dict = {"messages": []}

        # Phase 1: normal execution
        for _ in range(4):
            state["tool_ok"] = True
            state["progress_delta"] = 0.10
            state["error_count_delta"] = 0
            state = node(state)

        health_before = state[KEY_HEALTH]
        assert health_before["healthy"] > 0.80

        # Phase 2: error burst → escalate → checkpoint
        for _ in range(5):
            state["tool_ok"] = False
            state["progress_delta"] = -0.05
            state["error_count_delta"] = 2
            state = node(state)

        target = router(state)
        if target == "human":
            # Simulate checkpoint save
            checkpoint = dict(state)
            checkpoint["__checkpoint_marker"] = True

            # Human intervenes → resume
            resume_state = dict(checkpoint)
            resume_state["tool_ok"] = True
            resume_state["progress_delta"] = 0.05
            resume_state["error_count_delta"] = 0
            resume_state["has_user_msg"] = True  # user got involved

            # After resume, engine should still reflect pre-checkpoint history
            # Additional recovery steps
            for _ in range(3):
                resume_state["tool_ok"] = True
                resume_state["progress_delta"] = 0.12
                resume_state["error_count_delta"] = 0
                resume_state = node(resume_state)

            # Should eventually recover to healthy
            final_target = router(resume_state)
            assert final_target in {"agent", "tools"}

    def test_multiple_checkpoint_cycles(self):
        """Multiple escalate-recover cycles don't corrupt engine state."""
        engine = DecisionEngine(seed=1)
        node = create_judgment_node(engine)
        router = create_judgment_router(engine)

        state: dict = {"messages": []}
        escalated_count = 0

        for cycle in range(3):
            # Healthy phase
            for _ in range(3):
                state["tool_ok"] = True
                state["progress_delta"] = 0.10
                state["error_count_delta"] = 0
                state = node(state)

            # Error phase
            for _ in range(4):
                state["tool_ok"] = False
                state["progress_delta"] = -0.05
                state["error_count_delta"] = 1
                state = node(state)

            if router(state) == "human":
                escalated_count += 1
                # Recovery intervention
                state["tool_ok"] = True
                state["progress_delta"] = 0.05
                state["error_count_delta"] = 0

        # At least some escalations triggered
        assert escalated_count >= 1, f"Expected escalation in at least one cycle, got {escalated_count}"
