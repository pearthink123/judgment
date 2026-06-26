"""
Hawkes Process (self-exciting point process).

Models clustering of events: one successful tool call or error can increase
the instantaneous 'urge' for the agent to act again.

This is much more powerful than simple Poisson for agent timing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HawkesEvent:
    time: float
    mark: float = 1.0   # event strength (e.g. success=1.0, error=1.5)


class HawkesProcess:
    """
    Exponential kernel Hawkes process:
    λ(t) = μ + Σ α * exp(-β * (t - t_i)) for past events
    """

    def __init__(
        self,
        mu: float = 0.6,      # background intensity
        alpha: float = 1.8,   # excitation strength
        beta: float = 0.9,    # decay rate (higher = forgets faster)
        max_history: int = 50,
    ):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.max_history = max_history
        self.events: List[HawkesEvent] = []
        self.current_time = 0.0

    def intensity(self, t: Optional[float] = None, history: Optional[List[HawkesEvent]] = None) -> float:
        """Compute current intensity at time t."""
        if t is None:
            t = self.current_time
        if history is None:
            history = self.events

        lam = self.mu
        for ev in history:
            dt = t - ev.time
            if dt > 0:
                lam += self.alpha * ev.mark * np.exp(-self.beta * dt)
        return max(lam, 1e-6)

    def add_event(self, t: float, mark: float = 1.0):
        """Record a new event (success, error, user interaction, etc.)."""
        self.events.append(HawkesEvent(t, mark))
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]
        self.current_time = max(self.current_time, t)

    def trigger_probability(self, dt: float = 1.0) -> float:
        """Approx prob of at least one excitation in next dt."""
        lam = self.intensity()
        return 1.0 - np.exp(-lam * dt)

    def reset(self):
        self.events = []
        self.current_time = 0.0

    def get_recent_intensity_curve(self, window: float = 10.0, steps: int = 50) -> np.ndarray:
        """Return intensity trajectory for plotting."""
        if not self.events:
            return np.full(steps, self.mu)
        t_start = max(0.0, self.current_time - window)
        times = np.linspace(t_start, self.current_time, steps)
        return np.array([self.intensity(t) for t in times])
