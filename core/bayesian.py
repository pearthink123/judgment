"""
Bayesian State Estimator.

Maintains probabilistic beliefs about hidden states of the task and environment:
- Task progress / success probability
- Error risk / stuck probability
- User context suitability for intervention

Uses simple conjugate updates (Beta for probabilities) and discrete Bayes filter.
This replaces vague "the model feels..." with actual distributions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BeliefState:
    """Probabilistic belief over key hidden variables."""
    task_success_prob: float = 0.5
    error_risk: float = 0.3
    user_available: float = 0.6   # P(user is interruptible / will respond positively)
    stuck_likelihood: float = 0.25

    def to_dict(self) -> Dict[str, float]:
        return {
            "task_success": self.task_success_prob,
            "error_risk": self.error_risk,
            "user_available": self.user_available,
            "stuck": self.stuck_likelihood,
        }


class BayesianStateEstimator:
    """
    Lightweight Bayesian belief updater.
    For production you could use particle filters or more sophisticated models.
    """

    def __init__(self, prior: BeliefState = None):
        self.belief = prior or BeliefState()
        self.history: list[BeliefState] = [self.belief]

    def update_from_observation(
        self,
        observation: Dict[str, Any],
        action_taken: str = "",
    ) -> BeliefState:
        """
        observation example:
        {
            "tool_success": True/False,
            "error_count_delta": 0 or 1 or 2,
            "progress_delta": 0.1~0.4,
            "user_response": "positive"/"neutral"/"negative"/None,
            "steps_taken": int
        }
        """
        b = self.belief

        # 1. Task success belief (Beta-like update)
        if "progress_delta" in observation:
            delta = float(observation["progress_delta"])
            success_likelihood = 0.5 + 0.5 * np.tanh(delta * 4)
            # Weighted Bayesian update (simple)
            b.task_success_prob = 0.7 * b.task_success_prob + 0.3 * success_likelihood
            b.task_success_prob = np.clip(b.task_success_prob, 0.05, 0.98)

        if "tool_success" in observation:
            if observation["tool_success"]:
                b.task_success_prob = min(0.98, b.task_success_prob + 0.08)
            else:
                b.task_success_prob = max(0.1, b.task_success_prob - 0.15)

        # 2. Error risk update
        err_delta = observation.get("error_count_delta", 0)
        if err_delta > 0:
            b.error_risk = min(0.95, b.error_risk + 0.12 * err_delta)
        else:
            b.error_risk = max(0.08, b.error_risk * 0.85)

        # 3. Stuck likelihood (compound errors heuristic + steps)
        steps = observation.get("steps_taken", 0)
        if steps > 8 and b.task_success_prob < 0.55:
            b.stuck_likelihood = min(0.85, b.stuck_likelihood + 0.07)
        else:
            b.stuck_likelihood = max(0.05, b.stuck_likelihood * 0.92)

        # 4. User availability
        if observation.get("user_response") == "positive":
            b.user_available = min(0.95, b.user_available + 0.15)
        elif observation.get("user_response") == "negative":
            b.user_available = max(0.15, b.user_available - 0.25)
        else:
            b.user_available = 0.6 * b.user_available + 0.4 * 0.55

        # Regularize
        b.task_success_prob = float(np.clip(b.task_success_prob, 0.02, 0.98))
        b.error_risk = float(np.clip(b.error_risk, 0.02, 0.95))
        b.user_available = float(np.clip(b.user_available, 0.1, 0.95))
        b.stuck_likelihood = float(np.clip(b.stuck_likelihood, 0.02, 0.92))

        self.history.append(BeliefState(
            task_success_prob=b.task_success_prob,
            error_risk=b.error_risk,
            user_available=b.user_available,
            stuck_likelihood=b.stuck_likelihood,
        ))
        if len(self.history) > 60:
            self.history = self.history[-60:]

        self.belief = b
        return b

    def get_entropy(self) -> float:
        """Rough uncertainty measure (higher = more uncertain -> higher value of info)."""
        probs = [
            self.belief.task_success_prob,
            1 - self.belief.task_success_prob,
            self.belief.error_risk,
            self.belief.stuck_likelihood,
        ]
        # Shannon entropy approx (binary terms)
        ent = 0.0
        for p in probs:
            p = max(1e-6, min(1 - 1e-6, p))
            ent -= p * np.log(p) + (1 - p) * np.log(1 - p)
        return ent / 4.0

    def reset(self):
        self.belief = BeliefState()
        self.history = [self.belief]
