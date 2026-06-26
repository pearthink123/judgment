"""
Control Theory components for Harness regulation.

- PIDController: classic proportional-integral-derivative for behavior modulation.
- StochasticController: simple stochastic optimal control / gain scheduling.

Used to dynamically adjust:
- How aggressive the agent is (exploration vs exploitation)
- Retry / correction gain when errors accumulate
- Trigger threshold based on long-term performance
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ControlSignal:
    aggressiveness: float   # >1.0 means act more boldly
    correction_gain: float  # higher when recovering from errors
    exploration_bias: float # 0..1 , how much to favor high-uncertainty actions
    throttle: float         # 0..1 multiplier on overall activity


class PIDController:
    """Discrete PID controller for error-driven regulation."""

    def __init__(
        self,
        kp: float = 1.6,
        ki: float = 0.25,
        kd: float = 0.8,
        setpoint: float = 0.85,  # target task_success
        output_limits: tuple = (0.4, 2.8),
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_output = 1.0

    def update(self, measured: float, dt: float = 1.0) -> float:
        error = self.setpoint - measured

        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        ) + 1.0   # bias so neutral = 1.0

        output = float(np.clip(output, *self.output_limits))
        self.prev_error = error
        self.last_output = output
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class StochasticController:
    """
    Combines PID output with stochastic elements and belief state
    to produce final control signals for the judgment engine.
    """

    def __init__(self):
        self.pid = PIDController()

    def regulate(
        self,
        belief: Dict[str, float],
        error_accum: float,
        steps: int,
    ) -> ControlSignal:
        ts = belief.get("task_success", 0.5)
        err = belief.get("error_risk", 0.3)
        stuck = belief.get("stuck", 0.25)

        agg = self.pid.update(ts)

        # When errors high, increase correction
        correction = 1.0 + min(1.6, 0.9 * err + 1.1 * stuck + 0.08 * error_accum)

        # Exploration bias: favor uncertain / information-rich actions when uncertain
        entropy_proxy = -ts * np.log(max(ts, 1e-5)) - (1-ts) * np.log(max(1-ts, 1e-5))
        exploration = float(np.clip(0.2 + 0.65 * (entropy_proxy - 0.6), 0.05, 0.92))

        # Global throttle: back off if we are looping too long without progress
        throttle = 1.0
        if steps > 12 and ts < 0.6:
            throttle = max(0.45, 1.0 - 0.04 * (steps - 8))

        return ControlSignal(
            aggressiveness=float(np.clip(agg, 0.5, 2.6)),
            correction_gain=float(np.clip(correction, 0.6, 3.2)),
            exploration_bias=exploration,
            throttle=float(np.clip(throttle, 0.3, 1.0)),
        )
