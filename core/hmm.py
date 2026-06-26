"""
Hidden Markov Model — Layer 2: latent state estimation.

A 3-state discrete HMM for inferring the agent's operational health:

    H  = Healthy   — tool success ~80%, positive progress, rare user msgs
    D  = Degraded  — tool success ~50%, slow/zero progress, errors cluster
    B  = Broken    — tool success <30%, negative progress, user intervention needed

Inference uses the Forward algorithm in log-space for numerical stability.
Observation likelihood assumes dimension-conditional independence given
the latent state (product model).

References:
  Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models."
    Proceedings of the IEEE, 77(2), 257–286.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
STATE_HEALTHY = 0
STATE_DEGRADED = 1
STATE_BROKEN = 2

STATE_NAMES: Dict[int, str] = {0: "healthy", 1: "degraded", 2: "broken"}
N_STATES: int = 3

# ---------------------------------------------------------------------------
# Observation dimension enumeration
#   dim 0: tool_ok       (2 cats: fail, ok)
#   dim 1: progress      (3 cats: neg, zero, pos)
#   dim 2: user_msg      (2 cats: silent, msg)
#   dim 3: error_trend   (2 cats: stable, rising)
# ---------------------------------------------------------------------------
DIM_TOOL = 0
DIM_PROGRESS = 1
DIM_USER = 2
DIM_ERROR = 3

N_DIMS: int = 4

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

# Prior over initial state: slightly biased toward Healthy
DEFAULT_PRIOR = np.array([0.65, 0.28, 0.07], dtype=np.float64)

# Transition matrix T[s_from][s_to]
# Designed with Markov inertia (diagonal-dominant) and an entropic drift
# (H→D more likely than D→H; B is semi-absorbing).
DEFAULT_TRANSITION = np.array([
    #   H       D       B
    [0.80,   0.17,   0.03   ],   # from Healthy
    [0.15,   0.65,   0.20   ],   # from Degraded
    [0.02,   0.10,   0.88   ],   # from Broken
], dtype=np.float64)

# Emission tables — P(obs_dim | state)
# Each is a list of (n_categories,) arrays, one per state.

# Dim 0: tool_ok — P(fail | s), P(ok | s)   ← stored [P(fail), P(ok)]
EMISSION_TOOL = np.array([
    [0.20, 0.80],   # Healthy
    [0.50, 0.50],   # Degraded
    [0.75, 0.25],   # Broken
], dtype=np.float64)

# Dim 1: progress — P(neg | s), P(zero | s), P(pos | s)
EMISSION_PROGRESS = np.array([
    [0.05, 0.20, 0.75],   # Healthy
    [0.10, 0.55, 0.35],   # Degraded
    [0.30, 0.60, 0.10],   # Broken
], dtype=np.float64)

# Dim 2: user_msg — P(silent | s), P(msg | s)
EMISSION_USER = np.array([
    [0.95, 0.05],   # Healthy
    [0.85, 0.15],   # Degraded
    [0.75, 0.25],   # Broken
], dtype=np.float64)

# Dim 3: error_trend — P(stable | s), P(rising | s)
EMISSION_ERROR = np.array([
    [0.90, 0.10],   # Healthy
    [0.50, 0.50],   # Degraded
    [0.20, 0.80],   # Broken
], dtype=np.float64)

EMISSION_TABLES: Dict[int, np.ndarray] = {
    DIM_TOOL: EMISSION_TOOL,
    DIM_PROGRESS: EMISSION_PROGRESS,
    DIM_USER: EMISSION_USER,
    DIM_ERROR: EMISSION_ERROR,
}


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------
def encode_observation(
    tool_ok: bool,
    progress_delta: float,
    has_user_msg: bool,
    error_count_delta: int,
) -> Dict[int, int]:
    """
    Map raw observation signals to discrete category indices.

    Returns dict: {dim_index: category_index}

    Categories:
      tool_ok:       0=fail, 1=ok
      progress_delta: 0=neg, 1=zero, 2=pos
      user_msg:      0=silent, 1=msg
      error_trend:   0=stable, 1=rising
    """
    # tool
    tool_cat = 1 if tool_ok else 0

    # progress
    if progress_delta > 0.02:
        prog_cat = 2       # pos
    elif progress_delta < -0.01:
        prog_cat = 0       # neg
    else:
        prog_cat = 1       # zero

    # user
    user_cat = 1 if has_user_msg else 0

    # error trend (discretised at a single step — CUSUM tracks longer trends)
    err_cat = 1 if error_count_delta > 0 else 0

    return {
        DIM_TOOL: tool_cat,
        DIM_PROGRESS: prog_cat,
        DIM_USER: user_cat,
        DIM_ERROR: err_cat,
    }


# ---------------------------------------------------------------------------
# HMM class
# ---------------------------------------------------------------------------
class HiddenMarkovModel:
    """
    3-state discrete HMM with log-space Forward filtering.

    Parameters
    ----------
    prior : np.ndarray, shape (3,)
        Initial state distribution π_s = P(S₁ = s).
    transition : np.ndarray, shape (3, 3)
        State transition matrix T[s_from][s_to].
    emission_tables : dict
        {dim: np.ndarray of shape (3, n_cats)}.
    """

    def __init__(
        self,
        prior: Optional[np.ndarray] = None,
        transition: Optional[np.ndarray] = None,
        emission_tables: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.prior = prior.copy() if prior is not None else DEFAULT_PRIOR.copy()
        self.T = transition.copy() if transition is not None else DEFAULT_TRANSITION.copy()
        self.B = (
            {k: v.copy() for k, v in emission_tables.items()}
            if emission_tables is not None
            else {k: v.copy() for k, v in EMISSION_TABLES.items()}
        )

        # Validate shapes
        assert self.prior.shape == (N_STATES,)
        assert self.T.shape == (N_STATES, N_STATES)
        for dim, tbl in self.B.items():
            assert tbl.shape[0] == N_STATES, f"Dim {dim} table must have {N_STATES} rows"

        # Log-transform for numerical stability
        self.log_prior = np.log(self.prior + 1e-12)
        self.log_T = np.log(self.T + 1e-12)
        self.log_B: Dict[int, np.ndarray] = {
            dim: np.log(tbl + 1e-12) for dim, tbl in self.B.items()
        }

        # Running state
        self.log_alpha: Optional[np.ndarray] = None   # current forward messages
        self.t: int = 0                                 # step counter

    # ------------------------------------------------------------------
    # Observation log-likelihood (product model with independence assumption)
    # ------------------------------------------------------------------
    def log_obs_likelihood(self, obs_cats: Dict[int, int]) -> np.ndarray:
        """
        Compute log P(o_t | s) for each state s (public wrapper).

        Under dimension-conditional independence:
            log P(o | s) = Σ_dim log P(o_dim | s)

        Returns
        -------
        np.ndarray of shape (3,) — log-likelihood per state.
        """
        return self._log_obs_likelihood(obs_cats)

    def _log_obs_likelihood(self, obs_cats: Dict[int, int]) -> np.ndarray:
        """Internal implementation — see log_obs_likelihood."""
        log_lik = np.zeros(N_STATES, dtype=np.float64)
        for dim, cat in obs_cats.items():
            log_lik += self.log_B[dim][:, cat]
        return log_lik

    # ------------------------------------------------------------------
    # Forward step (online filter)
    # ------------------------------------------------------------------
    def forward_step(self, obs_cats: Dict[int, int]) -> np.ndarray:
        """
        Single online forward-filtering step.

        Returns
        -------
        belief : np.ndarray, shape (3,)
            P(S_t = s | o_{1:t}) — current filtered posterior.
        """
        log_lik = self._log_obs_likelihood(obs_cats)

        if self.log_alpha is None:
            # First step: α₁(s) = π_s · P(o₁ | s)
            log_alpha = self.log_prior + log_lik
        else:
            # α_t(s) = P(o_t | s) · Σ_{s'} α_{t-1}(s') · T_{s'→s}
            # Log-space: log α_t(s) = log_lik(s) + logsumexp_{s'} [log α_{t-1}(s') + log T(s', s)]
            log_alpha = log_lik.copy()
            for s_to in range(N_STATES):
                terms = self.log_alpha + self.log_T[:, s_to]
                log_alpha[s_to] += _logsumexp(terms)

        self.log_alpha = log_alpha
        self.t += 1

        # Normalize to get belief
        return self.belief()

    def belief(self) -> np.ndarray:
        """Current filtered posterior b(s) = P(S_t = s | o_{1:t})."""
        if self.log_alpha is None:
            return self.prior.copy()
        return _safe_softmax(self.log_alpha)

    def belief_dict(self) -> Dict[str, float]:
        """Return belief as a dict keyed by state names."""
        b = self.belief()
        return {
            "healthy": float(b[STATE_HEALTHY]),
            "degraded": float(b[STATE_DEGRADED]),
            "broken": float(b[STATE_BROKEN]),
        }

    def most_likely_state(self) -> int:
        """MAP state estimate (0, 1, or 2)."""
        return int(np.argmax(self.belief()))

    # ------------------------------------------------------------------
    # Batch forward
    # ------------------------------------------------------------------
    def forward_batch(
        self, obs_sequence: List[Dict[int, int]]
    ) -> List[np.ndarray]:
        """Run Forward on a batch, returning belief at each step."""
        beliefs: List[np.ndarray] = []
        for obs_cats in obs_sequence:
            b = self.forward_step(obs_cats)
            beliefs.append(b)
        return beliefs

    # ------------------------------------------------------------------
    # Viterbi (optional — for offline analysis)
    # ------------------------------------------------------------------
    def viterbi(self, obs_sequence: List[Dict[int, int]]) -> List[int]:
        """Most-likely state *sequence* via Viterbi decoding."""
        if not obs_sequence:
            return []

        T_steps = len(obs_sequence)
        log_delta = np.zeros((T_steps, N_STATES), dtype=np.float64)
        psi = np.zeros((T_steps, N_STATES), dtype=np.int32)

        # Init
        log_lik_0 = self._log_obs_likelihood(obs_sequence[0])
        log_delta[0] = self.log_prior + log_lik_0
        psi[0] = 0

        # Recurse
        for t_idx in range(1, T_steps):
            log_lik = self._log_obs_likelihood(obs_sequence[t_idx])
            for s_to in range(N_STATES):
                scores = log_delta[t_idx - 1] + self.log_T[:, s_to]
                best = int(np.argmax(scores))
                psi[t_idx, s_to] = best
                log_delta[t_idx, s_to] = log_lik[s_to] + scores[best]

        # Backtrack
        path = [0] * T_steps
        path[-1] = int(np.argmax(log_delta[-1]))
        for t_idx in range(T_steps - 2, -1, -1):
            path[t_idx] = int(psi[t_idx + 1, path[t_idx + 1]])

        return path

    # ------------------------------------------------------------------
    # Expected observation log-likelihood under current belief (for CUSUM)
    # ------------------------------------------------------------------
    def expected_log_lik(self, obs_cats: Dict[int, int]) -> float:
        """
        E_{s ~ belief}[log P(o_t | s)] — the model's surprise at this
        observation, marginalised over the current belief.

        Low values → surprising observation (feeds CUSUM drift).
        """
        log_lik = self._log_obs_likelihood(obs_cats)
        b = self.belief()
        return float(np.dot(b, log_lik))

    def reset(self):
        self.log_alpha = None
        self.t = 0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _logsumexp(x: np.ndarray) -> float:
    """Log-sum-exp in a numerically stable way."""
    c = x.max()
    if np.isinf(c) and c < 0:
        return -np.inf
    return float(c + np.log(np.sum(np.exp(x - c))))


def _safe_softmax(log_x: np.ndarray) -> np.ndarray:
    """Softmax from log-space, clipped away from zero."""
    x = log_x - _logsumexp(log_x)
    p = np.exp(x)
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum()
