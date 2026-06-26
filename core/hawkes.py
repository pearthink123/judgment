"""
Multivariate Marked Hawkes Process — baseline likelihood provider for Layer 1.

Models the temporal clustering of agent-harness events across D=4 types.
It no longer makes decisions; it computes how "expected" an observation is
given the event history, feeding into CUSUM anomaly detection.

Intensity for type d at time t:
    λ_d(t) = μ_d + Σ_{k: t_k < t} α_{d, e_k} · m_k · exp(-β · (t - t_k))

Stationarity condition: ρ(A / β) < 1  where A = [α_{d,e}].
With default β=1.0 and max-row-sum(A)=0.90, stationarity holds. ✓

References:
  Hawkes, A. G. (1971). Biometrika, 58(1), 83–90.
  Brémaud, P. & Massoulié, L. (1996). Annals of Probability, 24(3), 1563–1588.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------
EVENT_SUCCESS = 0
EVENT_ERROR = 1
EVENT_USER = 2
EVENT_TOOL = 3

EVENT_NAMES: Dict[int, str] = {
    0: "success",
    1: "error",
    2: "user_interaction",
    3: "tool_call",
}

N_TYPES: int = 4

# ---------------------------------------------------------------------------
# Mark distributions — Beta(a, b) scaled to type-appropriate [lo, hi]
# ---------------------------------------------------------------------------
MARK_CONFIG: Dict[int, Dict[str, float]] = {
    EVENT_SUCCESS: {"a": 2.0, "b": 2.0, "lo": 0.30, "hi": 1.00},
    EVENT_ERROR:   {"a": 5.0, "b": 2.0, "lo": 0.50, "hi": 1.80},
    EVENT_USER:    {"a": 3.0, "b": 3.0, "lo": 0.50, "hi": 1.50},
    EVENT_TOOL:    {"a": 4.0, "b": 2.0, "lo": 0.40, "hi": 1.20},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class HawkesEvent:
    """A single event in the Hawkes history."""
    time: float
    event_type: int     # 0=success, 1=error, 2=user_interaction, 3=tool_call
    mark: float         # magnitude drawn from type-specific Beta


@dataclass
class HawkesDiagnostics:
    """Lightweight snapshot for dashboards / debugging."""
    intensities: np.ndarray            # λ ∈ ℝ⁴ at current time
    baseline: np.ndarray               # μ (background vector)
    n_events: int
    recent_events: List[HawkesEvent]   # up to last 10


# ---------------------------------------------------------------------------
# Main process
# ---------------------------------------------------------------------------
class HawkesProcess:
    """
    Exponential-kernel multivariate Hawkes with typed marks.

    Parameters
    ----------
    mu : np.ndarray, shape (4,)
        Baseline intensity for each event type.
    alpha : np.ndarray, shape (4, 4)
        Excitation matrix; alpha[d, e] = how much an event of type e
        excites the intensity of type d.
    beta : float
        Shared exponential decay (per step, not per second).
    max_history : int
        Maximum number of events retained.
    rng : np.random.Generator or None
        Reproducible mark sampling.
    """

    DEFAULT_MU = np.array([0.30, 0.15, 0.08, 0.50], dtype=np.float64)

    # Row sums: [0.43, 0.41, 0.37, 0.90] — all < β=1.0 → stationary ✓
    DEFAULT_ALPHA = np.array([
        #  suc   err   usr   tool
        [0.15, 0.00, 0.10, 0.18],   # success ← *
        [0.00, 0.35, 0.00, 0.06],   # error   ← *
        [0.08, 0.25, 0.04, 0.00],   # user    ← *
        [0.20, 0.40, 0.12, 0.18],   # tool    ← *
    ], dtype=np.float64)

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        beta: float = 1.0,
        max_history: int = 100,
        rng: Optional[np.random.Generator] = None,
    ):
        self.mu = mu.copy() if mu is not None else self.DEFAULT_MU.copy()
        self.alpha = alpha.copy() if alpha is not None else self.DEFAULT_ALPHA.copy()
        self.beta = float(beta)
        self.max_history = int(max_history)

        if self.mu.shape != (N_TYPES,):
            raise ValueError(f"mu must be shape ({N_TYPES},), got {self.mu.shape}")
        if self.alpha.shape != (N_TYPES, N_TYPES):
            raise ValueError(
                f"alpha must be ({N_TYPES},{N_TYPES}), got {self.alpha.shape}"
            )

        self.rng = rng if rng is not None else np.random.default_rng()
        self.events: List[HawkesEvent] = []
        self.current_time: float = 0.0

    # ------------------------------------------------------------------
    # Intensity
    # ------------------------------------------------------------------
    def intensity(self, t: Optional[float] = None) -> np.ndarray:
        """Return λ_d(t) for all d ∈ {0,1,2,3} as a (4,) array."""
        if t is None:
            t = self.current_time

        lam = self.mu.astype(np.float64).copy()

        for ev in self.events:
            dt = t - ev.time
            if dt <= 0:
                continue
            lam += (
                self.alpha[:, ev.event_type]
                * ev.mark
                * np.exp(-self.beta * dt)
            )

        return np.maximum(lam, 1e-8)

    # ------------------------------------------------------------------
    # Single-type helpers (convenience for callers)
    # ------------------------------------------------------------------
    def intensity_for(self, event_type: int, t: Optional[float] = None) -> float:
        """λ_d(t) for a single event type."""
        return float(self.intensity(t)[event_type])

    def surprisal(self, event_type: int, t: Optional[float] = None) -> float:
        """
        Negative log-likelihood of observing *event_type* under the Hawkes model.

        Under the Poisson-with-rate-λ view:
            surprisal ≈ -log λ_d(t)

        High surprisal → event was unexpected → feeds CUSUM drift.
        The constant term (log factorial, etc.) is dropped because CUSUM
        only needs the log-likelihood *difference* between H₀ and H₁.
        """
        lam = max(self.intensity_for(event_type, t), 1e-8)
        return -np.log(lam)

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------
    def add_event(
        self,
        t: float,
        event_type: int,
        mark: Optional[float] = None,
    ):
        """Record a typed event.  If mark is None, sample from the type's Beta."""
        if event_type not in (0, 1, 2, 3):
            raise ValueError(f"Unknown event_type {event_type}")

        if mark is None:
            mark = self._sample_mark(event_type)

        self.events.append(
            HawkesEvent(time=float(t), event_type=event_type, mark=float(mark))
        )

        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]

        self.current_time = max(self.current_time, float(t))

    def add_observation(
        self,
        t: float,
        tool_ok: bool,
        has_user_msg: bool,
        progress_delta: float,
        error_count_delta: int,
    ):
        """
        Parse a raw observation into typed Hawkes events.

        One observation step can emit up to 4 events (one per type).
        """
        # success ↔ error (mutually exclusive per tool call)
        if tool_ok:
            self.add_event(
                t, EVENT_SUCCESS,
                mark=0.3 + 0.7 * min(abs(progress_delta), 1.0),
            )
        else:
            severity = 0.8 + 0.6 * min(float(error_count_delta), 2.0)
            self.add_event(t, EVENT_ERROR, mark=min(severity, 1.8))

        # Every step involves a tool call
        self.add_event(t, EVENT_TOOL)

        # User interaction
        if has_user_msg:
            self.add_event(t, EVENT_USER)

    # ------------------------------------------------------------------
    # Mark sampling
    # ------------------------------------------------------------------
    def _sample_mark(self, event_type: int) -> float:
        cfg = MARK_CONFIG[event_type]
        x = self.rng.beta(cfg["a"], cfg["b"])
        return cfg["lo"] + (cfg["hi"] - cfg["lo"]) * x

    @staticmethod
    def sample_mark(
        event_type: int, rng: Optional[np.random.Generator] = None
    ) -> float:
        """Static sampling helper."""
        if rng is None:
            rng = np.random.default_rng()
        cfg = MARK_CONFIG[event_type]
        x = rng.beta(cfg["a"], cfg["b"])
        return cfg["lo"] + (cfg["hi"] - cfg["lo"]) * x

    # ------------------------------------------------------------------
    # Stationarity check
    # ------------------------------------------------------------------
    def check_stationarity(self) -> bool:
        """ρ(A / β) < 1 ?"""
        spectral_radius = max(abs(np.linalg.eigvals(self.alpha)))
        return (spectral_radius / self.beta) < 1.0

    @property
    def spectral_radius(self) -> float:
        return float(max(abs(np.linalg.eigvals(self.alpha))))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_diagnostics(self) -> HawkesDiagnostics:
        recent = list(self.events[-10:])
        return HawkesDiagnostics(
            intensities=self.intensity(),
            baseline=self.mu.copy(),
            n_events=len(self.events),
            recent_events=recent,
        )

    def get_intensity_trajectory(
        self, window: float = 10.0, steps: int = 50
    ) -> np.ndarray:
        """Return (steps, 4) intensity array for plotting."""
        if not self.events:
            return np.tile(self.mu, (steps, 1))
        t_start = max(0.0, self.current_time - window)
        times = np.linspace(t_start, self.current_time, steps)
        return np.array([self.intensity(t) for t in times])

    def reset(self):
        self.events = []
        self.current_time = 0.0
