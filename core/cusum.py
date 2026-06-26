"""
CUSUM Anomaly Detection — Layer 1 of the decision stack.

Computes a cumulative drift statistic S_t from the observation stream.
When S_t crosses a threshold h, it signals that the system has deviated
from the "healthy" regime — triggering state-estimation re-evaluation
in Layer 2.

Form (Page, 1954):
    S_t = max(0, S_{t-1} + L_t)
    L_t = log [ f₁(o_t) / f₀(o_t) ]   ← log-likelihood ratio

where:
    f₀ = observation likelihood under "in-control" (healthy) model
    f₁ = observation likelihood under "out-of-control" (degraded/broken) model

The Hawkes baseline corrects f₀ by adjusting for event clustering: an
observation that was "due to happen" (high λ_d) carries less surprise.

References:
  Page, E. S. (1954). Biometrika, 41(1/2), 100–115.
  Tartakovsky, A., Nikiforov, I., & Basseville, M. (2014).
    Sequential Analysis: Hypothesis Testing and Changepoint Detection. CRC Press.
"""

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class CUSUMDetector:
    """
    CUSUM (cumulative sum) change-point detector for agent harnesses.

    Parameters
    ----------
    h : float
        Alarm threshold.  When S_t > h the detector fires and resets.
        Higher h → fewer false alarms, slower detection.
    gamma : float
        Hawkes correction strength.  γ·log(λ_d) is subtracted from the
        surprisal so that temporally-clustered (expected) events are
        less surprising.
    drift_floor : float
        Minimum per-step drift — prevents S_t decaying to zero too fast
        during mild anomalies.
    """

    def __init__(
        self,
        h: float = 4.0,
        gamma: float = 0.35,
        drift_floor: float = -0.15,
    ):
        self.h = float(h)
        self.gamma = float(gamma)
        self.drift_floor = float(drift_floor)

        # Running state
        self.S: float = 0.0          # cumulative drift
        self.t: int = 0              # step counter
        self.alarm_history: list[int] = []  # steps at which alarm fired
        self.S_history: list[float] = []    # full trace of S_t for plotting

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        surprisal_healthy: float,
        hawkes_intensity: float = 1.0,
        surprisal_degraded: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Feed one observation and return anomaly signal.

        Parameters
        ----------
        surprisal_healthy : float
            -log P(o_t | H₀), from Layer 2's expected log-likelihood
            (or directly from the HMM).
        hawkes_intensity : float
            λ_d for the most relevant event type.  High λ → the observation
            was expected under the healthy temporal model → reduce drift.
        surprisal_degraded : float or None
            -log P(o_t | H₁).  If None, a default "uninformative" value
            is used.

        Returns
        -------
        dict with keys:
            S          — updated cumulative drift
            L          — this step's log-likelihood ratio
            alarm      — bool, whether S > h this step
            hawkes_correction — amount subtracted by Hawkes factor
        """
        self.t += 1

        # --- Hawkes correction ---
        # If Hawkes says this event was "due to happen" (λ is high),
        # the healthy-model surprisal is inflated → subtract a term.
        hawkes_correction = self.gamma * np.log(max(hawkes_intensity, 1e-6))

        # Corrected healthy surprisal
        corrected_surprisal = surprisal_healthy - hawkes_correction

        # --- Log-likelihood ratio ---
        # L_t = -log f₁ + log f₀  = surprisal_degraded - surprisal_healthy
        # But we want L_t > 0 when the observation is MORE likely under f₁.
        # i.e. L_t = corrected_surprisal - surprisal_degraded
        if surprisal_degraded is None:
            # Use a constant "out-of-control" baseline (uniform-ish).
            # A typical degraded surprisal is lower because degraded model
            # *expects* failures → less surprised by them.
            # We use the *corrected healthy surprisal* minus a pessimistic
            # constant as a one-sided CUSUM.
            L = corrected_surprisal - 0.35
        else:
            # L > 0 when healthy is more surprised than degraded.
            L = corrected_surprisal - surprisal_degraded

        # Prevent individual steps from dragging S down too far
        # (this is a one-sided CUSUM: we only care about positive drift)
        L = max(L, self.drift_floor)

        # --- Cumulative drift ---
        self.S = max(0.0, self.S + L)

        # --- Alarm ---
        alarm = self.S > self.h
        if alarm:
            self.S = 0.0   # reset after alarm
            self.alarm_history.append(self.t)

        self.S_history.append(self.S)

        return {
            "S": round(self.S, 4),
            "L": round(L, 4),
            "alarm": alarm,
            "hawkes_correction": round(hawkes_correction, 4),
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @property
    def is_alarmed(self) -> bool:
        """True if most recent step triggered an alarm."""
        return (
            len(self.alarm_history) > 0
            and self.alarm_history[-1] == self.t
        )

    @property
    def alarm_count(self) -> int:
        return len(self.alarm_history)

    def reset(self):
        self.S = 0.0
        self.t = 0
        self.alarm_history = []
        self.S_history = []

    def get_summary(self) -> Dict[str, object]:
        return {
            "S_current": round(self.S, 4),
            "threshold": self.h,
            "alarms_fired": len(self.alarm_history),
            "alarm_steps": list(self.alarm_history),
            "mean_drift": (
                round(float(np.mean(self.S_history)), 4)
                if self.S_history
                else 0.0
            ),
        }
