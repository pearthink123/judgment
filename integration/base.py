"""
Abstract adapter protocol for harness integration.

Any harness adapter (LangGraph, CrewAI, custom loop) implements
this interface.  The core contract is:

  adapter.extract(state)    → observation dict
  adapter.inject(state, d)  → modified state
  adapter.route(state)      → next node / action name
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from core.engine import Decision, DecisionEngine, ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER


# ---------------------------------------------------------------------------
# Observation extractor — maps harness state → engine observation
# ---------------------------------------------------------------------------
ObservationExtractor = Callable[[Dict[str, Any]], Dict[str, Any]]


def default_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract engine-compatible observation from a generic state dict.

    Override this with a harness-specific function if your state keys differ.
    """
    # Try common key names from LangGraph, CrewAI, and custom loops
    return {
        "tool_ok": state.get("tool_ok", state.get("tool_success", True)),
        "progress_delta": float(state.get("progress_delta", state.get("progress", 0.0))),
        "has_user_msg": bool(state.get("has_user_msg", state.get("user_message", False))),
        "error_count_delta": int(state.get("error_count_delta", state.get("errors_this_step", 0))),
    }


# ---------------------------------------------------------------------------
# Routing target names (configurable per harness)
# ---------------------------------------------------------------------------
@dataclass
class RouteTargets:
    """Node names the adapter will route to."""
    continue_target: str = "agent"       # back to normal agent loop
    correct_target: str = "agent"        # back to agent with corrective message
    escalate_target: str = "human"       # human-in-the-loop node
    gather_target: str = "agent"         # back to agent with gather prompt


# ---------------------------------------------------------------------------
# State keys used by the adapter (prefixed to avoid collisions)
# ---------------------------------------------------------------------------
KEY_ACTION = "__judgment_action"
KEY_HEALTH = "__judgment_health"
KEY_RATIONALE = "__judgment_rationale"
KEY_ADVICE = "__judgment_advice"


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------
class BaseAdapter:
    """
    Abstract adapter.  Subclass and override extract / inject / route.

    Parameters
    ----------
    engine : DecisionEngine
    extractor : callable  (state) → observation dict
    routes : RouteTargets
    """

    def __init__(
        self,
        engine: Optional[DecisionEngine] = None,
        extractor: Optional[ObservationExtractor] = None,
        routes: Optional[RouteTargets] = None,
    ):
        self.engine = engine or DecisionEngine()
        self.extractor = extractor or default_extractor
        self.routes = routes or RouteTargets()

    def extract(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map harness state → engine observation. Override if needed."""
        return self.extractor(state)

    def step(self, state: Dict[str, Any]) -> Decision:
        """Feed current state to the engine, return a decision."""
        obs = self.extract(state)
        return self.engine.step(obs)

    def inject(self, state: Dict[str, Any], decision: Decision) -> Dict[str, Any]:
        """
        Modify state based on engine decision.

        Default behaviour:
          - Alway annotate state with __judgment_* keys.
          - On CORRECT: append a system message with corrective advice.
          - On GATHER:  append a system message prompting info collection.
          - On ESCALATE / CONTINUE: no message injection.

        Subclasses for frameworks with typed message lists should override this.
        """
        state = dict(state)
        state[KEY_ACTION] = decision.action
        state[KEY_HEALTH] = decision.belief
        state[KEY_RATIONALE] = decision.rationale
        state[KEY_ADVICE] = (
            {
                "type": decision.corrective_advice.action_type,
                "summary": decision.corrective_advice.summary,
            }
            if decision.corrective_advice
            else None
        )

        # Append a message for corrective / gather so the LLM sees it
        messages = state.get("messages", [])
        if decision.action == ACTION_CORRECT and decision.corrective_advice:
            messages = list(messages) + [{
                "role": "system",
                "content": f"[JUDGMENT · CORRECTIVE] {decision.corrective_advice.summary}",
            }]
        elif decision.action == ACTION_GATHER:
            messages = list(messages) + [{
                "role": "system",
                "content": "[JUDGMENT · GATHERING] System uncertain — collect more information before acting.",
            }]
        state["messages"] = messages
        return state

    def route(self, decision: Decision) -> str:
        """Map engine action → harness node name. Override if needed."""
        return {
            ACTION_CONTINUE: self.routes.continue_target,
            ACTION_CORRECT: self.routes.correct_target,
            ACTION_ESCALATE: self.routes.escalate_target,
            ACTION_GATHER: self.routes.gather_target,
        }[decision.action]

    # Convenience: step + inject + route in one call (used by node functions)
    def process(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Full pipeline: extract → step → inject → route."""
        obs = self.extract(state)
        decision = self.engine.step(obs)
        new_state = self.inject(state, decision)
        target = self.route(decision)
        return new_state, target
