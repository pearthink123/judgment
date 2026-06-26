"""
JudgmentEngine - The core math-driven decision module for Agent Harness.

This is the plug-in replacement (or augmentation) for heuristic/prompt-only
decision making inside ReAct, Plan-Execute-Verify, or custom loops.

Flow per turn:
  1. Receive observation + current context summary
  2. Bayesian update of belief state
  3. Hawkes / Poisson update trigger intensity (proactive urge)
  4. Compute EVOI for every candidate action
  5. Apply control signals (PID + stochastic)
  6. Select action + emit rich Decision with all math diagnostics

Designed to be framework agnostic.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from .hawkes import HawkesProcess
from .bayesian import BayesianStateEstimator, BeliefState
from .info_gain import ExpectedValueOfInformation, ActionValue
from .control import StochasticController, ControlSignal


@dataclass
class Decision:
    action: str
    confidence: float
    trigger_intensity: float
    evoi: float
    control: ControlSignal
    belief: Dict[str, float]
    math_diagnostics: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


class JudgmentEngine:
    """
    Main entry point.

    Example usage inside a harness loop:
        engine = JudgmentEngine()
        decision = engine.decide(observation, context)
        # then execute decision.action
        engine.record_outcome(decision.action, tool_result)
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.hawkes = HawkesProcess(mu=0.55, alpha=1.9, beta=0.85)
        self.bayesian = BayesianStateEstimator()
        self.evoi = ExpectedValueOfInformation()
        self.controller = StochasticController()

        self.step = 0
        self.error_accumulation = 0.0
        self.last_decision: Optional[Decision] = None

        self.action_history: List[str] = []
        self.decision_log: List[Decision] = []

    def decide(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """
        Main decision function.
        """
        self.step += 1
        context = context or {}

        # --- 1. Bayesian belief update ---
        belief_obj = self.bayesian.update_from_observation(observation, self.action_history[-1] if self.action_history else "")
        belief = belief_obj.to_dict()

        # --- 2. Hawkes self-exciting trigger ---
        # Mark strength depends on outcome signal
        mark = 1.0
        if observation.get("tool_success") is False:
            mark = 1.35
        if observation.get("error_count_delta", 0) > 0:
            mark = 1.6
        if observation.get("progress_delta", 0) > 0.15:
            mark = 0.85

        self.hawkes.add_event(self.step, mark=mark)
        trigger_intensity = self.hawkes.intensity()

        # --- 3. EVOI scores ---
        evoi_values = self.evoi.compute(
            belief,
            recent_history_len=len(self.action_history),
            error_accumulation=self.error_accumulation,
        )
        best_action, best_val = self.evoi.best_action(evoi_values)

        # --- 4. Control regulation ---
        control = self.controller.regulate(belief, self.error_accumulation, self.step)

        # Final score = EVOI * trigger * control factors
        final_scores: Dict[str, float] = {}
        for act, val in evoi_values.items():
            score = val.evoi * (trigger_intensity ** 0.6)
            score *= (control.aggressiveness ** 0.35)
            score *= control.throttle
            if control.exploration_bias > 0.6 and act in ("read_file", "think"):
                score *= 1.25   # favor info gathering
            final_scores[act] = score

        # Pick the winner (with small noise for robustness)
        best_action = max(final_scores, key=final_scores.get)
        best_evoi = evoi_values[best_action].evoi
        confidence = float(np.clip(0.5 + 0.4 * (best_evoi / (1 + trigger_intensity * 0.1)), 0.25, 0.96))

        # --- 5. Diagnostics & rationale ---
        math_diag = {
            "trigger_intensity": round(trigger_intensity, 3),
            "belief_entropy": round(self.bayesian.get_entropy(), 3),
            "final_score": round(final_scores[best_action], 3),
            "evoi_scores": {k: round(v.evoi, 2) for k, v in evoi_values.items()},
            "control": {
                "aggressiveness": round(control.aggressiveness, 3),
                "correction_gain": round(control.correction_gain, 3),
                "throttle": round(control.throttle, 3),
            },
            "hawkes_recent_events": len(self.hawkes.events),
        }

        rationale = self._generate_rationale(best_action, belief, trigger_intensity, best_evoi, control)

        decision = Decision(
            action=best_action,
            confidence=confidence,
            trigger_intensity=round(trigger_intensity, 3),
            evoi=round(best_evoi, 3),
            control=control,
            belief=belief,
            math_diagnostics=math_diag,
            rationale=rationale,
        )

        self.last_decision = decision
        self.decision_log.append(decision)
        self.action_history.append(best_action)

        return decision

    def record_outcome(self, action: str, outcome: Dict[str, Any]):
        """Call after executing the action to feed back real results."""
        if outcome.get("error_count_delta", 0) > 0 or outcome.get("tool_success") is False:
            self.error_accumulation = min(6.0, self.error_accumulation + 0.8)
        else:
            self.error_accumulation = max(0.0, self.error_accumulation * 0.65)

        # Re-inject observation into bayesian & hawkes for next round (already done in decide, but allows external loop)
        self.bayesian.update_from_observation(outcome, action)

    def get_diagnostics_dataframe(self):
        """Helper for dashboard / analysis."""
        import pandas as pd
        if not self.decision_log:
            return pd.DataFrame()
        rows = []
        for i, d in enumerate(self.decision_log):
            row = {
                "step": i,
                "action": d.action,
                "confidence": d.confidence,
                "trigger": d.trigger_intensity,
                "evoi": d.evoi,
                "task_success": d.belief.get("task_success", 0),
                "error_risk": d.belief.get("error_risk", 0),
                "stuck": d.belief.get("stuck", 0),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def reset(self):
        self.hawkes.reset()
        self.bayesian.reset()
        self.step = 0
        self.error_accumulation = 0.0
        self.action_history = []
        self.decision_log = []
        self.last_decision = None
        self.controller.pid.reset()

    def _generate_rationale(
        self,
        action: str,
        belief: Dict[str, float],
        trigger: float,
        evoi: float,
        control: ControlSignal,
    ) -> str:
        parts = []
        ts = belief.get("task_success", 0.5)
        err = belief.get("error_risk", 0.3)

        if trigger > 2.2:
            parts.append("high proactive urge (Hawkes clustering)")
        elif trigger > 1.3:
            parts.append("moderate urge to act")

        if evoi > 1.0:
            parts.append(f"high EVOI ({evoi:.2f})")
        elif evoi > 0.6:
            parts.append("reasonable information value")

        if err > 0.55:
            parts.append("error correction mode (PID)")
        if control.throttle < 0.7:
            parts.append("throttled due to low progress")

        if ts < 0.4:
            parts.append("low task success belief")

        if not parts:
            parts.append("balanced math signals")

        return f"Chose {action} because: " + ", ".join(parts) + "."
