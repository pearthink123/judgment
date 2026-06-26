"""
Belief-MDP solver — optimal action selection under partial observability.

Converts the HMM into a POMDP by adding action-dependent transitions
and a reward function, then solves via exact value iteration on a
discretised belief simplex.

3 states × 4 actions × ~24 observations × 231 grid points.
Offline solve: milliseconds. Online lookup: microseconds.

Reference:
  Kaelbling, Littman & Cassandra (1998). "Planning and Acting in
  Partially Observable Stochastic Domains." AIJ, 101(1-2), 99–134.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from itertools import product

from .hmm import (
    STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN,
    STATE_NAMES, N_STATES,
    EMISSION_TABLES, DIM_TOOL, DIM_PROGRESS, DIM_USER, DIM_ERROR,
)


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """
    Reward matrix R(s,a).  All values in units of "one effective progress step".

    Derivation: each value is the net expected progress gain (positive) or
    loss (negative) relative to one normal step being worth +1.

    Three presets are provided; users can also set individual fields.
    """

    # continue
    continue_H: float = 1.0
    continue_D: float = -0.5
    continue_B: float = -3.0

    # correct
    correct_H: float = -0.3
    correct_D: float = 2.0
    correct_B: float = 0.5

    # escalate
    escalate_H: float = -2.0
    escalate_D: float = -0.5
    escalate_B: float = 2.0

    # gather
    gather_H: float = 0.2
    gather_D: float = 0.8
    gather_B: float = -1.0

    gamma: float = 0.95

    # -------- helpers --------
    def matrix(self) -> np.ndarray:
        """Return R as a (3, 4) array: rows = states, cols = actions."""
        return np.array([
            [self.continue_H, self.correct_H, self.escalate_H, self.gather_H],
            [self.continue_D, self.correct_D, self.escalate_D, self.gather_D],
            [self.continue_B, self.correct_B, self.escalate_B, self.gather_B],
        ], dtype=np.float64)

    @classmethod
    def preset(cls, name: str) -> "RewardConfig":
        presets = {
            "general": cls(),  # default — see class-level doc
            "conservative": cls(   # ops / finance — missing an error is worse
                continue_B=-6.0,
                escalate_H=-1.0,
                escalate_D=0.0,
                escalate_B=3.0,
                correct_B=1.0,
                gather_B=-2.0,
            ),
            "permissive": cls(     # experimental / creative — false alarms are worse
                continue_B=-1.5,
                escalate_H=-3.0,
                escalate_D=-1.5,
                escalate_B=1.5,
                correct_D=1.0,
                gather_D=0.3,
            ),
        }
        return presets.get(name, cls())


# ---------------------------------------------------------------------------
# Action constants (POMDP-level — match engine constants)
# ---------------------------------------------------------------------------
ACT_CONTINUE = 0
ACT_CORRECT = 1
ACT_ESCALATE = 2
ACT_GATHER = 3

N_ACTIONS = 4

ACTION_NAMES_POMDP: Dict[int, str] = {
    0: "continue",
    1: "correct",
    2: "escalate",
    3: "gather",
}

# ---------------------------------------------------------------------------
# Action-dependent transition matrices  P(s' | s, a)
# ---------------------------------------------------------------------------
# Design rationale for each action:
#   continue  — slight degradation trend, Broken is absorbing
#   correct   — strong recovery from Degraded, moderate from Broken
#   escalate  — strongest recovery (user helps), but high cost
#   gather    — mild improvement, slower

DEFAULT_TRANSITIONS = {
    ACT_CONTINUE: np.array([
        #   H       D       B
        [0.75,   0.22,   0.03   ],   # from Healthy
        [0.08,   0.70,   0.22   ],   # from Degraded
        [0.01,   0.04,   0.95   ],   # from Broken
    ], dtype=np.float64),

    ACT_CORRECT: np.array([
        [0.85,   0.14,   0.01   ],
        [0.45,   0.50,   0.05   ],
        [0.08,   0.35,   0.57   ],
    ], dtype=np.float64),

    ACT_ESCALATE: np.array([
        [0.92,   0.07,   0.01   ],
        [0.55,   0.40,   0.05   ],
        [0.50,   0.35,   0.15   ],
    ], dtype=np.float64),

    ACT_GATHER: np.array([
        [0.88,   0.11,   0.01   ],
        [0.25,   0.68,   0.07   ],
        [0.03,   0.15,   0.82   ],
    ], dtype=np.float64),
}

# ---------------------------------------------------------------------------
# Observation space enumeration
# ---------------------------------------------------------------------------
def _enumerate_observations() -> List[Tuple[int, ...]]:
    """Enumerate all 2×3×2×2 = 24 possible discrete observations."""
    cats = [
        list(range(EMISSION_TABLES[DIM_TOOL].shape[1])),      # 2
        list(range(EMISSION_TABLES[DIM_PROGRESS].shape[1])),  # 3
        list(range(EMISSION_TABLES[DIM_USER].shape[1])),      # 2
        list(range(EMISSION_TABLES[DIM_ERROR].shape[1])),     # 2
    ]
    return list(product(*cats))


ALL_OBSERVATIONS: List[Tuple[int, ...]] = _enumerate_observations()
N_OBSERVATIONS: int = len(ALL_OBSERVATIONS)  # 24


def observation_prob(o_idx: int) -> np.ndarray:
    """
    P(o | s) for a given observation index, as a (3,) array.

    Dimension-conditional independence: P(o|s) = ∏_dim P(o_dim | s).
    """
    obs_tuple = ALL_OBSERVATIONS[o_idx]
    prob = np.ones(N_STATES, dtype=np.float64)
    for dim, cat in enumerate(obs_tuple):
        prob *= EMISSION_TABLES[dim][:, cat]
    return prob


# ---------------------------------------------------------------------------
# Belief simplex discretisation
# ---------------------------------------------------------------------------
def discretise_simplex(resolution: float = 0.05) -> Tuple[np.ndarray, Dict[Tuple[float, ...], int]]:
    """
    Generate grid points on the 2-simplex.

    Returns
    -------
    grid : np.ndarray of shape (N, 3)
        Each row is a belief vector (b0, b1, b2) with sum 1.
    index_of : dict
        Maps (b0, b1, b2) tuple to grid index.
    """
    points: List[np.ndarray] = []
    index_of: Dict[Tuple[float, ...], int] = {}

    for i in range(int(1.0 / resolution) + 1):
        b0 = round(i * resolution, 6)
        for j in range(int((1.0 - b0) / resolution) + 1):
            b1 = round(j * resolution, 6)
            b2 = round(1.0 - b0 - b1, 6)
            if b2 < -1e-9:
                continue
            vec = np.array([b0, b1, max(b2, 0.0)], dtype=np.float64)
            vec = vec / vec.sum()  # renormalise for float safety
            key = (round(float(vec[0]), 6), round(float(vec[1]), 6), round(float(vec[2]), 6))
            index_of[key] = len(points)
            points.append(vec)

    return np.array(points), index_of


def nearest_grid_point(belief: np.ndarray, grid: np.ndarray) -> int:
    """Find the index of the nearest grid point (L1 distance)."""
    diff = np.abs(grid - belief[np.newaxis, :])
    return int(np.argmin(diff.sum(axis=1)))


# ---------------------------------------------------------------------------
# Belief-MDP solver
# ---------------------------------------------------------------------------
@dataclass
class POMDPPolicy:
    """Result of belief-MDP value iteration."""
    grid: np.ndarray                    # (N, 3) belief points
    V: np.ndarray                       # (N,) value function
    Q: np.ndarray                       # (N, 4) Q-values
    policy: np.ndarray                  # (N,) best action indices
    reward: RewardConfig
    transitions: Dict[int, np.ndarray]  # P(s'|s,a)
    n_iterations: int
    converged: bool

    def __repr__(self) -> str:
        return (
            f"POMDPPolicy(n_points={len(self.grid)}, "
            f"gamma={self.reward.gamma}, "
            f"iterations={self.n_iterations}, "
            f"converged={self.converged})"
        )

    def best_action(self, belief: np.ndarray) -> int:
        """Return best action idx for a belief vector (3,)."""
        idx = nearest_grid_point(belief, self.grid)
        return int(self.policy[idx])

    def action_name(self, belief: np.ndarray) -> str:
        return ACTION_NAMES_POMDP[self.best_action(belief)]

    def q_values(self, belief: np.ndarray) -> Dict[str, float]:
        idx = nearest_grid_point(belief, self.grid)
        return {ACTION_NAMES_POMDP[a]: float(self.Q[idx, a]) for a in range(N_ACTIONS)}

    def value(self, belief: np.ndarray) -> float:
        idx = nearest_grid_point(belief, self.grid)
        return float(self.V[idx])


def solve_belief_mdp(
    reward: Optional[RewardConfig] = None,
    transitions: Optional[Dict[int, np.ndarray]] = None,
    resolution: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> POMDPPolicy:
    """
    Exact value iteration on a discretised belief simplex.

    Parameters
    ----------
    reward : RewardConfig
    transitions : dict {action_idx: np.ndarray (3,3)}
        Action-dependent state transition matrices.
    resolution : float
        Belief simplex discretisation step.
    max_iter : int
        Maximum value iteration sweeps.
    tol : float
        Convergence threshold (max |ΔV|).

    Returns
    -------
    POMDPPolicy with grid, V, Q, and policy lookup table.
    """
    r = reward or RewardConfig()
    T = transitions or DEFAULT_TRANSITIONS
    R = r.matrix()   # (3, 4)
    gamma = r.gamma

    # --- discretise ---
    grid, _ = discretise_simplex(resolution)
    N_points = len(grid)

    # --- precompute P(o | s) for every observation ---
    obs_probs = np.array([observation_prob(i) for i in range(N_OBSERVATIONS)])  # (24, 3)

    # --- precompute for each action: E[P(o | s') P(s' | s, a) b(s)] ---
    # We'll compute this inside the Bellman loop.

    # --- initialise ---
    V = np.zeros(N_points, dtype=np.float64)     # old V
    Q = np.zeros((N_points, N_ACTIONS), dtype=np.float64)

    for it in range(max_iter):
        delta = 0.0

        for bi in range(N_points):
            b = grid[bi]  # (3,)

            # Immediate reward for each action
            imm_r = b @ R  # (4,)

            for a in range(N_ACTIONS):
                # --- Expected future value ---
                # E[V(b')] = Σ_o P(o|b,a) · V(b')
                #
                # b'(s') = P(o|s') Σ_s P(s'|s,a) b(s) / P(o|b,a)
                # P(o|b,a) = Σ_s P(o|s) b(s)   (since P(o|s,a)=P(o|s))
                #
                # We compute this as:
                #   numerator(s') = P(o|s') · Σ_s P(s'|s,a) · b(s)
                #   denominator = Σ_{s'} numerator(s')
                #   b'(s') = numerator(s') / denominator
                #   E[V] = Σ_o denominator · V(b')

                T_a = T[a]           # (3, 3)
                b_next_prior = T_a.T @ b  # (3,) — Σ_s P(s'|s,a)·b(s)

                expected_v = 0.0
                for oi in range(N_OBSERVATIONS):
                    po_s = obs_probs[oi]          # (3,) — P(o|s)
                    # P(o|b,a) = Σ_s P(o|s) b(s)  (action-independent emission)
                    po_ba = float(np.dot(po_s, b))

                    if po_ba < 1e-10:
                        continue  # observation essentially impossible

                    # b'(s') = P(o|s') · b_next_prior(s') / P(o|b,a)
                    b_next = po_s * b_next_prior / po_ba
                    b_next = np.clip(b_next, 0.0, 1.0)
                    b_next = b_next / b_next.sum()

                    ni = nearest_grid_point(b_next, grid)
                    expected_v += po_ba * V[ni]

                Q[bi, a] = imm_r[a] + gamma * expected_v

            new_v = Q[bi].max()
            delta = max(delta, abs(new_v - V[bi]))
            V[bi] = new_v

        if delta < tol:
            break

    # --- extract policy ---
    policy = np.argmax(Q, axis=1)

    return POMDPPolicy(
        grid=grid,
        V=V,
        Q=Q,
        policy=policy,
        reward=r,
        transitions=T,
        n_iterations=it + 1,
        converged=(delta < tol),
    )


# ---------------------------------------------------------------------------
# Module-level cache — solve once, use everywhere
# ---------------------------------------------------------------------------
_default_policy: Optional[POMDPPolicy] = None


def get_policy(
    reward: Optional[RewardConfig] = None,
    resolution: float = 0.05,
    force_recompute: bool = False,
) -> POMDPPolicy:
    """Get or compute the default POMDP policy (cached)."""
    global _default_policy
    if _default_policy is None or force_recompute or reward is not None:
        _default_policy = solve_belief_mdp(reward=reward, resolution=resolution)
    return _default_policy
