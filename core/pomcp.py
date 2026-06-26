"""
POMCP — Partially Observable Monte Carlo Planning (Silver & Veness, 2010).

Online particle-based UCT for POMDPs.  No grid discretisation — scales
to larger state spaces without the curse of dimensionality.

Replaces the grid value-iteration solver when ``use_pomcp=True`` in the
DecisionEngine.  The grid solver remains as a fast fallback.

Reference:
  Silver, D. & Veness, J. (2010). "Monte-Carlo Planning in Large POMDPs."
  Advances in Neural Information Processing Systems, 23.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .hmm import N_STATES, STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN
from .pomdp import (
    ACT_CONTINUE, ACT_CORRECT, ACT_ESCALATE, ACT_GATHER,
    N_ACTIONS, ACTION_NAMES_POMDP,
    RewardConfig,
    DEFAULT_TRANSITIONS,
    observation_prob,
    ALL_OBSERVATIONS,
)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------
class _POMCPNode:
    """A single history-node in the POMCP search tree."""

    __slots__ = ("N", "V", "N_a", "Q_a", "children", "particles")

    def __init__(self):
        self.N: int = 0                           # total visits
        self.V: float = 0.0                       # mean return from here
        self.N_a: np.ndarray = np.zeros(N_ACTIONS, dtype=np.int32)
        self.Q_a: np.ndarray = np.zeros(N_ACTIONS, dtype=np.float64)
        self.children: Dict[Tuple[int, int], _POMCPNode] = {}  # (a, o_idx) → node
        self.particles: Optional[np.ndarray] = None  # state indices, shape (K,)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
@dataclass
class POMCPSearchInfo:
    """Diagnostics from the most recent POMCP search."""
    simulations: int
    max_depth_reached: int
    root_visits: int
    q_values: Dict[str, float]
    best_action: str
    best_q: float
    runner_up_q: float
    tree_size: int   # total nodes in tree


class POMCPPlanner:
    """
    Online POMDP solver using particle-filter UCT.

    Parameters
    ----------
    transitions : dict {a: (3,3) ndarray}
        Action-conditional state transition matrices.
    reward_config : RewardConfig
    n_simulations : int
        Number of MCTS rollouts per search (default 1000).
    n_particles : int
        Number of belief particles (default 200).
    max_depth : int
        Maximum rollout depth (default 12).
    ucb_c : float
        UCB1 exploration constant (default 1.4).
    n_visit_threshold : int
        Visits before a node is expanded instead of roll-out (default 5).
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        transitions: Optional[Dict[int, np.ndarray]] = None,
        reward_config: Optional[RewardConfig] = None,
        n_simulations: int = 1000,
        n_particles: int = 200,
        max_depth: int = 12,
        ucb_c: float = 1.4,
        n_visit_threshold: int = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.T = transitions or DEFAULT_TRANSITIONS
        self.reward = reward_config or RewardConfig()
        self.R = self.reward.matrix()   # (3, 4)
        self.gamma = self.reward.gamma

        self.n_simulations = int(n_simulations)
        self.n_particles = int(n_particles)
        self.max_depth = int(max_depth)
        self.ucb_c = float(ucb_c)
        self.n_visit_threshold = int(n_visit_threshold)

        self.rng = rng if rng is not None else np.random.default_rng()

        # Pre-compute observation probabilities
        self._obs_probs = np.array(
            [observation_prob(i) for i in range(len(ALL_OBSERVATIONS))]
        )  # (N_OBS, 3)

        self._n_obs = len(ALL_OBSERVATIONS)

        # Root tree
        self._root: Optional[_POMCPNode] = None
        self._last_info: Optional[POMCPSearchInfo] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def search(self, belief: np.ndarray) -> int:
        """
        Run POMCP from the current belief and return the best action index.

        Parameters
        ----------
        belief : np.ndarray, shape (3,)
            Filtered belief vector P(S_t = s | o_{1:t}).

        Returns
        -------
        action_idx : int  (0=continue, 1=correct, 2=escalate, 3=gather)
        """
        # Generate particles from belief
        particles = self.rng.choice(
            N_STATES, size=self.n_particles, p=belief,
        ).astype(np.int32)

        # Reset tree — keep old root if re-using is desired, but for
        # simplicity we rebuild each step.  In production, particles from
        # the previous root can seed the next search (belief tracking).
        self._root = _POMCPNode()
        self._root.particles = particles

        max_depth_seen = 0

        for _ in range(self.n_simulations):
            # Sample a state from the root particle set
            s = int(particles[self.rng.integers(0, self.n_particles)])
            depth = self._simulate(s, self._root, 0)
            max_depth_seen = max(max_depth_seen, depth)

        # Best action
        best_a = int(np.argmax(self._root.Q_a))
        q_vals = self._root.Q_a.copy()

        # Runner-up
        sorted_q = sorted(q_vals, reverse=True)
        runner_up = sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]

        # Diagnostics
        tree_size = self._count_nodes(self._root)
        self._last_info = POMCPSearchInfo(
            simulations=self.n_simulations,
            max_depth_reached=max_depth_seen,
            root_visits=int(self._root.N),
            q_values={ACTION_NAMES_POMDP[a]: float(q_vals[a]) for a in range(N_ACTIONS)},
            best_action=ACTION_NAMES_POMDP[best_a],
            best_q=float(q_vals[best_a]),
            runner_up_q=float(runner_up),
            tree_size=tree_size,
        )

        return best_a

    @property
    def last_info(self) -> Optional[POMCPSearchInfo]:
        return self._last_info

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate(self, s: int, node: _POMCPNode, depth: int) -> float:
        """
        One MCTS simulation.  Returns total discounted return from (s, node).

        At each node:
          1. Select action (UCB if enough visits, otherwise random)
          2. Step generative model: s' ~ T, o ~ O, r = R
          3. Expand child node, recurse
          4. Backpropagate r + γ·V(child) to parent

        The n_visit_threshold controls when to switch from random exploration
        to UCB-based exploitation — but backprop always happens.
        """
        if depth >= self.max_depth:
            return 0.0

        # --- Action selection ---
        if node.N >= self.n_visit_threshold:
            # UCB selection
            node_total = node.N + 1
            ucb_values = np.full(N_ACTIONS, -np.inf, dtype=np.float64)
            for a in range(N_ACTIONS):
                if node.N_a[a] == 0:
                    ucb_values[a] = np.inf  # explore unvisited first
                else:
                    exploration = self.ucb_c * np.sqrt(
                        np.log(node_total) / node.N_a[a]
                    )
                    ucb_values[a] = node.Q_a[a] + exploration
            a = int(np.argmax(ucb_values))
        else:
            # Random action for new nodes (ensures breadth)
            a = int(self.rng.integers(0, N_ACTIONS))

        # --- Generative model step ---
        s_next = int(self.rng.choice(N_STATES, p=self.T[a][s]))
        obs_probs_s = self._obs_probs[:, s_next]
        o_idx = int(self.rng.choice(self._n_obs, p=obs_probs_s))
        r = float(self.R[s, a])

        # --- Descend / expand ---
        key = (a, o_idx)
        if key not in node.children:
            node.children[key] = _POMCPNode()
        child = node.children[key]

        # --- Recurse ---
        total_r = r + self.gamma * self._simulate(s_next, child, depth + 1)

        # --- Backpropagate ---
        node.N += 1
        node.N_a[a] += 1
        node.Q_a[a] += (total_r - node.Q_a[a]) / node.N_a[a]
        node.V += (total_r - node.V) / node.N

        return float(total_r)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_nodes(node: _POMCPNode) -> int:
        count = 1
        for child in node.children.values():
            count += POMCPPlanner._count_nodes(child)
        return count

    def reset(self):
        """Clear search tree.  last_info persists for read-only diagnostics."""
        self._root = None


# ---------------------------------------------------------------------------
# Belief particle sampling
# ---------------------------------------------------------------------------
def belief_to_particles(
    belief: np.ndarray,
    n_particles: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample n_particles state indices from a belief vector."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(N_STATES, size=n_particles, p=belief).astype(np.int32)
