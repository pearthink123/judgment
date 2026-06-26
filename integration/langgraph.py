"""
LangGraph adapter — drop-in health monitoring for LangGraph agents.

Two integration patterns:

  1. NODE  — create_judgment_node(engine) → a LangGraph-compatible node
  2. ROUTER — create_judgment_router(engine) → conditional-edge function

Usage:

    from judgment.integration.langgraph import (
        create_judgment_node, create_judgment_router,
    )
    from judgment import DecisionEngine

    engine = DecisionEngine()

    graph = StateGraph(MyState)
    graph.add_node("agent", my_agent_node)
    graph.add_node("tools", my_tool_node)
    graph.add_node("human", human_intervention_node)
    graph.add_node("judgment", create_judgment_node(engine))

    graph.add_edge("tools", "judgment")

    graph.add_conditional_edges(
        "judgment",
        create_judgment_router(engine),
        {"agent": "agent", "tools": "tools", "human": "human"},
    )

No LangGraph imports at module level — works with any dict-based state.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import (
    BaseAdapter,
    ObservationExtractor,
    RouteTargets,
    default_extractor,
    KEY_ACTION,
    KEY_HEALTH,
    KEY_RATIONALE,
    KEY_ADVICE,
)
from core.engine import (
    DecisionEngine,
    ACTION_CONTINUE,
    ACTION_CORRECT,
    ACTION_ESCALATE,
    ACTION_GATHER,
)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
def create_judgment_node(
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[ObservationExtractor] = None,
    routes: Optional[RouteTargets] = None,
    annotate_state: bool = True,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a LangGraph-compatible judgment node.

    This node:
      1. Extracts an observation from the current state
      2. Feeds it to the DecisionEngine
      3. Annotates the state with __judgment_* keys
      4. Injects corrective / gather messages into state["messages"]
      5. Returns the modified state dict

    Parameters
    ----------
    engine : DecisionEngine
    extractor : callable — maps state dict → observation dict
    routes : RouteTargets — node names for each engine action
    annotate_state : bool — add __judgment_* keys (default True)

    Returns
    -------
    node_fn : callable — (state: dict) → dict
    """
    adapter = BaseAdapter(engine=engine, extractor=extractor, routes=routes)

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        decision = adapter.step(state)
        new_state = adapter.inject(dict(state), decision)
        if not annotate_state:
            for k in (KEY_ACTION, KEY_HEALTH, KEY_RATIONALE, KEY_ADVICE):
                new_state.pop(k, None)
        return new_state

    node.__name__ = "judgment_node"
    return node


# ---------------------------------------------------------------------------
# Conditional-edge router factory
# ---------------------------------------------------------------------------
def create_judgment_router(
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[ObservationExtractor] = None,
    route_map: Optional[Dict[str, str]] = None,
) -> Callable[[Dict[str, Any]], str]:
    """
    Create a conditional-edge routing function.

    Reads state[__judgment_action] to decide the next node.
    Falls back to running the engine inline if the key is absent.

    Parameters
    ----------
    engine : DecisionEngine — fallback if __judgment_action not in state
    extractor : callable
    route_map : dict — override default node names, e.g.:
        {"continue": "agent", "correct": "agent",
         "escalate": "human", "gather": "tools"}

    Returns
    -------
    router_fn : callable — (state: dict) → str
    """
    adapter = BaseAdapter(engine=engine, extractor=extractor)
    if route_map:
        adapter.routes = RouteTargets(
            continue_target=route_map.get("continue", "agent"),
            correct_target=route_map.get("correct", "agent"),
            escalate_target=route_map.get("escalate", "human"),
            gather_target=route_map.get("gather", "agent"),
        )

    def router(state: Dict[str, Any]) -> str:
        action = state.get(KEY_ACTION)
        if action is None and adapter.engine is not None:
            decision = adapter.step(state)
            action = decision.action
            # also inject so subsequent nodes can see the decision
            state = dict(state)
            state[KEY_ACTION] = action
            state[KEY_HEALTH] = decision.belief
            state[KEY_RATIONALE] = decision.rationale

        _map = {
            ACTION_CONTINUE: adapter.routes.continue_target,
            ACTION_CORRECT: adapter.routes.correct_target,
            ACTION_ESCALATE: adapter.routes.escalate_target,
            ACTION_GATHER: adapter.routes.gather_target,
        }
        return _map.get(action, adapter.routes.continue_target)

    router.__name__ = "judgment_router"
    return router


# ---------------------------------------------------------------------------
# One-shot: augment a LangGraph StateGraph builder
# ---------------------------------------------------------------------------
def with_judgment(
    graph_builder,
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[ObservationExtractor] = None,
    tool_node_name: str = "tools",
    agent_node_name: str = "agent",
    human_node_name: str = "human",
    human_node_fn: Optional[Callable] = None,
):
    """
    Augment an existing LangGraph StateGraph builder with judgment oversight.

    Adds:
      - "judgment" node (DecisionEngine)
      - human_node_name node (default: interrupt signal)
      - judgment → {agent, tools, human} conditional edges

    The caller is still responsible for wiring tool → judgment.

    Parameters
    ----------
    graph_builder : langgraph.graph.StateGraph (uncompiled)
    engine : DecisionEngine
    extractor : callable — (state) → observation dict
    tool_node_name, agent_node_name, human_node_name : str
    human_node_fn : callable or None — custom human-intervention node

    Returns
    -------
    graph_builder — augmented builder
    """
    # Judgment node
    graph_builder.add_node(
        "judgment",
        create_judgment_node(engine=engine, extractor=extractor),
    )

    # Human escalation node
    if human_node_fn is None:
        def _default_human(state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(state)
            messages = list(state.get("messages", []))
            health = state.get(KEY_HEALTH, {})
            messages.append({
                "role": "system",
                "content": (
                    "[JUDGMENT · ESCALATED] "
                    f"H={health.get('healthy', 0):.2f} "
                    f"D={health.get('degraded', 0):.2f} "
                    f"B={health.get('broken', 0):.2f}. "
                    "Waiting for human intervention."
                ),
            })
            state["messages"] = messages
            state["__judgment_escalated"] = True
            return state
        human_node_fn = _default_human

    graph_builder.add_node(human_node_name, human_node_fn)

    # Conditional routing from judgment
    graph_builder.add_conditional_edges(
        "judgment",
        create_judgment_router(engine=engine, extractor=extractor),
        {
            agent_node_name: agent_node_name,
            tool_node_name: tool_node_name,
            human_node_name: human_node_name,
        },
    )

    return graph_builder
