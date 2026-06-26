"""
Expected Value of Information (EVOI / VOI) + Information Gain.

Quantifies: "How much do I expect to learn / improve my belief by taking this action?"

This directly attacks context rot and compound error: don't call expensive tools or
pollute context with low-value observations.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ActionValue:
    action: str
    evoi: float
    confidence: float
    expected_progress: float


class ExpectedValueOfInformation:
    """
    Approximate EVOI calculator for Harness actions.
    In real systems this can be replaced with more accurate tree search / POMDP value iteration.
    """

    def __init__(self, action_space: List[str] = None):
        self.action_space = action_space or [
            "think",
            "read_file",
            "edit_code",
            "run_tests",
            "verify",
            "escalate_to_user",
            "noop_wait",
        ]
        # Heuristic base values (can be learned)
        self.base_value = {
            "think": 0.38,
            "read_file": 0.52,
            "edit_code": 0.92,
            "run_tests": 0.88,
            "verify": 0.68,
            "escalate_to_user": 1.15,
            "noop_wait": 0.08,
        }

    def compute(
        self,
        belief: Dict[str, float],
        recent_history_len: int = 5,
        error_accumulation: float = 0.0,
    ) -> Dict[str, ActionValue]:
        """
        Compute EVOI for each action given current belief.
        Returns action -> ActionValue
        """
        results: Dict[str, ActionValue] = {}
        ts = belief.get("task_success", 0.5)
        err = belief.get("error_risk", 0.3)
        stuck = belief.get("stuck", 0.25)
        user_av = belief.get("user_available", 0.6)

        uncertainty = -ts * np.log(ts + 1e-8) - (1 - ts) * np.log(1 - ts + 1e-8)
        uncertainty = float(np.clip(uncertainty, 0.1, 1.8))

        progress = ts  # proxy
        for action in self.action_space:
            base = self.base_value.get(action, 0.4)

            # Value modulated by belief state + basic sequencing knowledge
            if action == "run_tests":
                # Strongly penalize testing before any code/progress
                if progress < 0.25:
                    val = 0.15
                else:
                    val = base * (0.6 + 0.7 * (1 - ts)) * (1.0 + 0.4 * err)
            elif action == "verify":
                val = base * (0.5 + 0.6 * ts) * (1.0 + 0.3 * err)
            elif action == "edit_code":
                val = base * (0.55 + 0.85 * (1 if progress < 0.5 else ts)) * (1.0 - 0.25 * stuck)
            elif action == "read_file":
                val = base * (0.75 + 0.45 * uncertainty)
            elif action == "escalate_to_user":
                val = base * user_av * (1.7 if stuck > 0.48 else 0.65)
            elif action == "think":
                val = base * (0.65 + 0.65 * uncertainty)
            else:
                val = base

            # Penalty for repeated low-value actions (error_accumulation simulates drift)
            val = val / (1.0 + 0.15 * error_accumulation)

            # Normalize to [0, ~2.2]
            evoi = float(np.clip(val, 0.05, 2.5))

            # Confidence in this value estimate (lower when high uncertainty)
            conf = float(np.clip(0.85 - 0.35 * uncertainty, 0.35, 0.95))

            expected_progress = float(np.clip(evoi * 0.55, 0.02, 0.6))

            results[action] = ActionValue(
                action=action,
                evoi=evoi,
                confidence=conf,
                expected_progress=expected_progress,
            )

        return results

    def best_action(self, values: Dict[str, ActionValue]) -> Tuple[str, ActionValue]:
        # Weighted by confidence
        scored = {
            a: v.evoi * (0.6 + 0.4 * v.confidence) for a, v in values.items()
        }
        best = max(scored, key=scored.get)
        return best, values[best]
