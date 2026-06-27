"""
CrewAI Adapter — drop judgment health monitoring into CrewAI agents.

Three integration patterns (no CrewAI import at module level):

  1. CALLBACK ─ step callback that runs DecisionEngine after each agent step
  2. TOOL     ─ CrewAI Tool for agents to self-check their health
  3. WRAPPER  ─ wrap call_tool to automatically feed observations

Usage (callback):

    from judgment.integration.crewai import create_judgment_callback
    from judgment import DecisionEngine

    engine = DecisionEngine()
    callback = create_judgment_callback(engine)

    agent = Agent(
        role="Developer",
        step_callback=callback,      # ← runs after each step
        ...
    )

Usage (tool):

    from judgment.integration.crewai import create_judgment_tool

    tool = create_judgment_tool(engine)
    agent = Agent(tools=[tool], ...)
    # Agent can now call "check_health" to inspect its own state

Usage (wrap tool calls):

    from judgment.integration.crewai import create_tool_wrapper

    wrapper = create_tool_wrapper(engine)
    # Then monkey-patch or subclass your tool's _run method
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

from core.engine import (
    DecisionEngine, Decision,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)


# ---------------------------------------------------------------------------
# 1. Step callback — runs after each CrewAI agent step
# ---------------------------------------------------------------------------
def create_judgment_callback(
    engine: Optional[DecisionEngine] = None,
    extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
    verbose: bool = False,
    on_escalate: Optional[Callable[[Decision], None]] = None,
) -> Callable:
    """
    Create a CrewAI step callback that feeds agent output to the DecisionEngine.

    Parameters
    ----------
    engine : DecisionEngine
        The judgment engine instance.
    extractor : callable or None
        Maps CrewAI step output → observation dict.
        If None, a best-effort extractor is used.
    verbose : bool
        Print engine decisions to stdout.
    on_escalate : callable or None
        Called when engine says ESCALATE. Receives the Decision.
        Use this to raise an exception, pause the crew, or notify.

    Returns
    -------
    callback : callable — suitable for Agent(step_callback=...)
    """
    eng = engine or DecisionEngine()

    def callback(step_output: Any) -> None:
        # Extract observation
        obs = {"tool_ok": True, "progress_delta": 0.05, "has_user_msg": False, "error_count_delta": 0}

        if extractor:
            obs = extractor(step_output)
        elif hasattr(step_output, "__dict__"):
            d = step_output.__dict__ if hasattr(step_output, "__dict__") else {}
            obs = {
                "tool_ok": not str(d.get("result", "")).lower().startswith("error"),
                "progress_delta": 0.05,
                "has_user_msg": False,
                "error_count_delta": 0 if not str(d.get("result", "")).lower().startswith("error") else 1,
            }
        elif isinstance(step_output, str):
            obs = {
                "tool_ok": not step_output.lower().startswith("error"),
                "progress_delta": 0.05,
                "has_user_msg": False,
                "error_count_delta": 0,
            }

        decision = eng.step(obs)

        if verbose:
            belief = decision.belief
            print(
                f"[judgment] {decision.action.upper()} "
                f"H={belief['healthy']:.2f} D={belief['degraded']:.2f} "
                f"B={belief['broken']:.2f}"
            )

        if decision.action == ACTION_CORRECT and decision.corrective_advice:
            if verbose:
                print(f"  → {decision.corrective_advice.summary}")

        if decision.action == ACTION_ESCALATE and on_escalate:
            on_escalate(decision)

    return callback


# ---------------------------------------------------------------------------
# 2. Health-check tool — agent can introspect its own state
# ---------------------------------------------------------------------------
@dataclass
class JudgmentToolConfig:
    """Configuration for a CrewAI-compatible judgment tool."""

    name: str = "check_health"
    description: str = (
        "Check the agent's current operational health. "
        "Returns health status (healthy/degraded/broken) and "
        "recommended action (continue/correct/escalate/gather). "
        "Call this when you suspect something is going wrong "
        "or the task is stalling."
    )
    engine: Optional[DecisionEngine] = field(default=None)

    def __post_init__(self):
        if self.engine is None:
            self.engine = DecisionEngine()


def create_judgment_tool(
    engine: Optional[DecisionEngine] = None,
    tool_name: str = "check_health",
):
    """
    Create a CrewAI Tool that lets agents call "check_health" to see
    their own health status via the DecisionEngine.

    The tool reads the engine's *current* belief state and returns a
    human-readable report. It does NOT advance the engine — the step
    callback handles that.

    Returns a dict with keys:
        name, description, func — ready for CrewAI BaseTool subclass.

    Usage:

        tool_spec = create_judgment_tool(engine)

        class HealthCheckTool(BaseTool):
            name: str = tool_spec["name"]
            description: str = tool_spec["description"]
            def _run(self, **kwargs) -> str:
                return tool_spec["func"]()

        agent = Agent(tools=[HealthCheckTool()], ...)
    """
    eng = engine or DecisionEngine()

    def _check() -> str:
        """Return health report from current engine state."""
        if not eng.decision_log:
            return "Health: unknown (no steps processed yet)."

        last = eng.decision_log[-1]
        b = last.belief
        lines = [
            f"Health Report:",
            f"  Healthy:  {b['healthy']:.3f}",
            f"  Degraded: {b['degraded']:.3f}",
            f"  Broken:   {b['broken']:.3f}",
            f"  Recommended action: {last.action.upper()}",
            f"  CUSUM drift: {last.drift:.3f} (alarm: {last.anomaly})",
        ]
        if last.corrective_advice:
            lines.append(f"  Advice: {last.corrective_advice.summary}")
        return "\n".join(lines)

    return {
        "name": tool_name,
        "description": JudgmentToolConfig().description,
        "func": _check,
    }


# ---------------------------------------------------------------------------
# 3. Tool-call wrapper — observe tool outputs
# ---------------------------------------------------------------------------
def create_tool_wrapper(
    engine: Optional[DecisionEngine] = None,
) -> Callable[[str, bool, float, int, bool, Optional[str]], str]:
    """
    Create a wrapper that feeds tool outputs to the engine.

    Returns a function `(tool_name, ok, progress, errors, user_msg, llm_text) → action`
    that you can call from inside your CrewAI tool's `_run()` method.

    Usage:

        engine = DecisionEngine()
        observe = create_tool_wrapper(engine)

        class MyTool(BaseTool):
            def _run(self, **kwargs) -> str:
                result = do_work()
                action = observe("my_tool", ok=True, progress=0.1, errors=0)
                if action == "escalate":
                    return "[ESCALATE] Agent health critical — stopping."
                return result
    """
    eng = engine or DecisionEngine()

    def observe(
        tool_name: str = "",
        ok: bool = True,
        progress: float = 0.05,
        errors: int = 0,
        user_msg: bool = False,
        llm_text: Optional[str] = None,
    ) -> str:
        decision = eng.step({
            "tool_ok": ok,
            "progress_delta": progress,
            "has_user_msg": user_msg,
            "error_count_delta": errors,
            "llm_text": llm_text,
        })
        return decision.action

    return observe
