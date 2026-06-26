"""
MathDrivenDecisionLoop

A thin wrapper showing how to integrate the JudgmentEngine into a classic
Agent harness loop (ReAct-style or Plan-Execute-Verify).

This is deliberately simple and self-contained so it can be dropped into
LangGraph, CrewAI, custom loops, or DeepSeek's internal harness.
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from ..core import JudgmentEngine, Decision


@dataclass
class HarnessTurn:
    step: int
    observation: Dict[str, Any]
    decision: Decision
    outcome: Dict[str, Any]


class MathDrivenDecisionLoop:
    """
    Example math-augmented agent loop.

    You provide:
      - execute_action(action, context) -> outcome_dict
    """

    def __init__(self, engine: Optional[JudgmentEngine] = None, max_steps: int = 25):
        self.engine = engine or JudgmentEngine()
        self.max_steps = max_steps
        self.history: list[HarnessTurn] = []

    def run(
        self,
        initial_observation: Dict[str, Any],
        execute_action: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run until task done or max steps.
        """
        obs = dict(initial_observation)
        ctx = context or {"task": "coding_task"}

        for step in range(1, self.max_steps + 1):
            decision = self.engine.decide(obs, ctx)

            # Execute in the real harness
            outcome = execute_action(decision.action, {**ctx, "step": step})

            # Feedback
            self.engine.record_outcome(decision.action, outcome)

            turn = HarnessTurn(step=step, observation=obs, decision=decision, outcome=outcome)
            self.history.append(turn)

            obs = outcome   # next observation comes from outcome

            # Termination heuristics (real harness would be better)
            if outcome.get("task_completed"):
                return {"status": "success", "steps": step, "final_belief": decision.belief}

            if decision.belief.get("stuck", 0) > 0.82 and decision.action == "escalate_to_user":
                return {"status": "escalated", "steps": step, "reason": "high stuck risk"}

            if step >= 3 and decision.belief.get("task_success", 0) < 0.12:
                return {"status": "failed", "steps": step, "final_belief": decision.belief}

        return {"status": "max_steps", "steps": self.max_steps, "final_belief": self.engine.last_decision.belief if self.engine.last_decision else {}}

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        successes = sum(1 for h in self.history if h.outcome.get("tool_success"))
        return {
            "total_steps": len(self.history),
            "actions_taken": [h.decision.action for h in self.history],
            "avg_confidence": sum(h.decision.confidence for h in self.history) / len(self.history),
            "avg_trigger": sum(h.decision.trigger_intensity for h in self.history) / len(self.history),
            "tool_success_rate": successes / len(self.history),
        }
