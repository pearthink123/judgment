"""
MathDrivenDecisionLoop

Thin wrapper showing how to integrate the DecisionEngine into a classic
Agent harness loop (ReAct, Plan-Execute-Verify, LangGraph, CrewAI, etc.).

Usage:
    engine = DecisionEngine(seed=42)
    loop = MathDrivenDecisionLoop(engine)

    def my_executor(action: str, ctx: dict) -> dict:
        # Actually run the tool and return observation
        return {"tool_ok": True, "progress_delta": 0.1, ...}

    result = loop.run(initial_obs, my_executor)
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from core.engine import (
    DecisionEngine,
    Decision,
    ACTION_ESCALATE,
    ACTION_CONTINUE,
)


@dataclass
class HarnessTurn:
    step: int
    observation: Dict[str, Any]
    decision: Decision
    outcome: Dict[str, Any]


class MathDrivenDecisionLoop:
    """
    Math-augmented agent loop using the 3-layer DecisionEngine.

    You provide:
      - execute_action(action: str, context: dict) -> observation_dict
    """

    def __init__(
        self,
        engine: Optional[DecisionEngine] = None,
        max_steps: int = 30,
    ):
        self.engine = engine or DecisionEngine()
        self.max_steps = max_steps
        self.history: list[HarnessTurn] = []

    def run(
        self,
        initial_observation: Dict[str, Any],
        execute_action: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run until task done, escalation, or max steps.

        Parameters
        ----------
        initial_observation : dict
            Must contain: tool_ok, progress_delta, has_user_msg, error_count_delta
        execute_action : callable
            (action: str, context: dict) -> observation dict
        context : dict or None
            Passed through to execute_action each step.

        Returns
        -------
        dict with keys: status, steps, final_belief, [reason]
        """
        obs = dict(initial_observation)
        ctx = dict(context or {})

        for step in range(1, self.max_steps + 1):
            decision = self.engine.step(obs)

            # Execute
            outcome = execute_action(decision.action, {**ctx, "step": step})

            turn = HarnessTurn(
                step=step,
                observation=obs,
                decision=decision,
                outcome=outcome,
            )
            self.history.append(turn)

            obs = outcome  # next observation

            # Termination
            if outcome.get("task_completed"):
                return {
                    "status": "success",
                    "steps": step,
                    "final_belief": decision.belief,
                }

            if (
                decision.action == ACTION_ESCALATE
                and decision.belief.get("broken", 0) > 0.55
            ):
                return {
                    "status": "escalated",
                    "steps": step,
                    "reason": "engine requested escalation",
                    "final_belief": decision.belief,
                }

            if (
                step >= 5
                and decision.action == ACTION_ESCALATE
                and decision.anomaly
            ):
                return {
                    "status": "escalated",
                    "steps": step,
                    "reason": "persistent anomaly + high broken belief",
                    "final_belief": decision.belief,
                }

        return {
            "status": "max_steps",
            "steps": self.max_steps,
            "final_belief": (
                self.engine.decision_log[-1].belief
                if self.engine.decision_log
                else {}
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        successes = sum(1 for h in self.history if h.outcome.get("tool_ok"))
        return {
            "total_steps": len(self.history),
            "actions_taken": [h.decision.action for h in self.history],
            "avg_confidence": (
                sum(h.decision.confidence for h in self.history)
                / len(self.history)
            ),
            "anomalies_detected": sum(1 for h in self.history if h.decision.anomaly),
            "tool_success_rate": successes / len(self.history),
        }
