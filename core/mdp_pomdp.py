"""
Simplified POMDP / Value of Information scaffolding.

For a real production Harness this would be a proper POMDP solver
(POUCT, DESPOT, or learned value function).

Here we provide a very lightweight rollout-based value estimator that
can be mixed into the JudgmentEngine.
"""

import numpy as np
from typing import Dict


class SimplePOMDPValueEstimator:
    """
    Very approximate expected cumulative value under current belief.
    Used to bias EVOI decisions for long-horizon compound error scenarios.
    """

    def __init__(self, horizon: int = 4):
        self.horizon = horizon

    def estimate_value(
        self,
        belief: Dict[str, float],
        action: str,
        future_evoi: float,
    ) -> float:
        ts = belief.get("task_success", 0.5)
        err = belief.get("error_risk", 0.3)
        stuck = belief.get("stuck", 0.25)

        # Base expected reward per step
        base = 0.4 * ts - 0.7 * err - 0.9 * stuck

        # Discounted multi-step
        value = base
        discount = 0.85
        for h in range(1, self.horizon):
            value += (base + 0.3 * future_evoi) * (discount ** h)

        # Bonus for actions that break stuck states
        if stuck > 0.45 and action in ("escalate_to_user", "verify"):
            value += 0.6

        return float(np.clip(value, -1.8, 2.8))
