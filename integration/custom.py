"""
Custom Adapter — 30 lines of glue to add judgment to any Agent loop.

Two patterns:

  1. FUNCTION WRAPPER ─ wrap an existing step function
  2. CONTEXT MANAGER ─ with-block around a main loop

Usage (wrapper):
    from judgment.integration.custom import wrap_step

    @wrap_step(engine)
    def my_step(state: dict) -> dict:
        # your existing logic
        return {"tool_ok": True, "progress_delta": 0.1, ...}

Usage (context):
    from judgment.integration.custom import judgment_guard

    with judgment_guard(engine) as check:
        for step in range(max_steps):
            obs = execute_tool()
            action = check(obs)   # returns "continue"/"correct"/"escalate"/"gather"
            if action == "escalate":
                break
            if action == "correct":
                log(obs, check.last_advice)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager
from dataclasses import dataclass

from core.engine import (
    DecisionEngine, Decision,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)
from core.corrective import CorrectiveAdvice


# ---------------------------------------------------------------------------
# 1. Function wrapper — decorate any step function
# ---------------------------------------------------------------------------
def wrap_step(
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
):
    """
    Decorate a harness step function to add judgment after each call.

    The wrapped function returns (original_result, judgment_decision).

    Example:

        engine = DecisionEngine()

        @wrap_step(engine)
        def agent_step(state):
            # do work...
            return {"tool_ok": True, "progress_delta": 0.12, "error_count_delta": 0}

        result, decision = agent_step(current_state)
        if decision.action == "escalate":
            break
    """
    eng = engine or DecisionEngine()

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)

            # Extract observation from result
            if extractor:
                obs = extractor(result)
            else:
                obs = result if isinstance(result, dict) else {}

            decision = eng.step({
                "tool_ok": obs.get("tool_ok", True),
                "progress_delta": float(obs.get("progress_delta", 0.0)),
                "has_user_msg": bool(obs.get("has_user_msg", False)),
                "error_count_delta": int(obs.get("error_count_delta", 0)),
                "llm_text": obs.get("llm_text", None),
            })

            return result, decision

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 2. Context manager — inline check in any loop
# ---------------------------------------------------------------------------
@dataclass
class _JudgmentGuard:
    """Stateful checker returned by judgment_guard()."""

    engine: DecisionEngine
    last_decision: Optional[Decision] = None

    @property
    def last_advice(self) -> Optional[CorrectiveAdvice]:
        return self.last_decision.corrective_advice if self.last_decision else None

    def __call__(self, observation: Dict[str, Any]) -> str:
        """
        Check one observation. Returns action string.
        Use .last_decision for full diagnostics.
        """
        self.last_decision = self.engine.step(observation)
        return self.last_decision.action


@contextmanager
def judgment_guard(engine: Optional[DecisionEngine] = None, **kwargs):
    """
    Context manager that yields a check() function.

    Example:

        engine = DecisionEngine()

        with judgment_guard(engine) as check:
            for step in range(max_steps):
                obs = run_tool()
                action = check(obs)
                if action == "escalate":
                    print("Engine says stop:", check.last_advice.summary)
                    break
    """
    eng = engine or DecisionEngine(**kwargs)
    guard = _JudgmentGuard(engine=eng)
    try:
        yield guard
    finally:
        pass  # engine state persists for inspection


# ---------------------------------------------------------------------------
# 3. One-shot overload detection — simplest possible integration
# ---------------------------------------------------------------------------
def quick_check(
    engine: DecisionEngine,
    tool_ok: bool = True,
    progress_delta: float = 0.0,
    error_count_delta: int = 0,
    has_user_msg: bool = False,
    llm_text: Optional[str] = None,
) -> str:
    """
    One-line check for the simplest integration path.

    Returns "continue", "correct", "escalate", or "gather".

    Example:

        engine = DecisionEngine()
        for step in range(100):
            ok = call_llm_tool()
            action = quick_check(engine, tool_ok=ok, progress_delta=0.1)
            if action == "escalate":
                break
    """
    return engine.step({
        "tool_ok": tool_ok,
        "progress_delta": progress_delta,
        "has_user_msg": has_user_msg,
        "error_count_delta": error_count_delta,
        "llm_text": llm_text,
    }).action
