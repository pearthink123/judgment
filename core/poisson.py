"""
Poisson Process for baseline proactive trigger modeling.
Used to model the base rate at which an Agent 'wants' to act.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class PoissonEvent:
    time: float
    intensity: float


class PoissonProcess:
    """Homogeneous or inhomogeneous Poisson process for event triggering."""

    def __init__(self, base_rate: float = 0.8):
        self.base_rate = base_rate
        self.events: List[PoissonEvent] = []
        self.current_time = 0.0

    def intensity(self, t: float = None) -> float:
        """Current intensity (for homogeneous: constant)."""
        return self.base_rate

    def sample_interarrival(self, rng: np.random.Generator = None) -> float:
        """Sample time until next event."""
        if rng is None:
            rng = np.random.default_rng()
        # Exponential distribution
        return -np.log(1 - rng.random()) / self.base_rate

    def trigger_probability(self, dt: float = 1.0) -> float:
        """Probability at least one event occurs in interval dt."""
        lam = self.base_rate * dt
        return 1.0 - np.exp(-lam)

    def record_event(self, t: float, intensity: float):
        self.events.append(PoissonEvent(t, intensity))
        self.current_time = max(self.current_time, t)

    def reset(self):
        self.events = []
        self.current_time = 0.0
