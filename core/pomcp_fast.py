"""
POMCP Fast — performance-optimised online MCTS planner.

Three optimisations over baseline pomcp.py:

  1. BATCH RNG — all random numbers pre-sampled before simulation loop.
     Eliminates numpy RNG call overhead in the hot path.
  2. EARLY STOPPING — halts when best-action Q-margin exceeds threshold
     for N consecutive checks. Reduces work on clear-cut decisions.
  3. ITERATIVE + INLINE — flat loop replaces recursive _simulate;
     path recorded for single-pass backprop. No Python call overhead.

Usage is identical:

    from core.pomcp_fast import FastPOMCPPlanner
    planner = FastPOMCPPlanner(n_simulations=500)
    action = planner.search(belief)
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .hmm import N_STATES
from .pomdp import (
    ACT_CONTINUE, ACT_CORRECT, ACT_ESCALATE, ACT_GATHER,
    N_ACTIONS, ACTION_NAMES_POMDP,
    RewardConfig,
    DEFAULT_TRANSITIONS,
    observation_prob,
    ALL_OBSERVATIONS,
)


# ---------------------------------------------------------------------------
# Pre-computed tables
# ---------------------------------------------------------------------------
_OBS_PROBS = np.array([observation_prob(i) for i in range(len(ALL_OBSERVATIONS))])
_N_OBS = len(ALL_OBSERVATIONS)

# Precompute cumulative transition + observation CDFs for searchsorted
_T_CDF = {a: np.cumsum(T_mat, axis=1).astype(np.float64)
          for a, T_mat in DEFAULT_TRANSITIONS.items()}
_O_CDF = np.cumsum(_OBS_PROBS, axis=1).astype(np.float64)  # (24, 3)


# ---------------------------------------------------------------------------
# Search info
# ---------------------------------------------------------------------------
@dataclass
class FastSearchInfo:
    simulations_requested: int
    simulations_run: int           # actual sims (may be < requested due to early stop)
    early_stopped: bool
    q_values: Dict[str, float]
    best_action: str
    best_q: float
    runner_up_q: float
    elapsed_ms: float

    # Compatibility with POMCPSearchInfo API used by engine
    @property
    def simulations(self) -> int:
        return self.simulations_requested

    tree_size: int = 1             # batch-pre-sampled doesn't track tree size
    max_depth_reached: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FastPOMCPPlanner:
    """
    Performance-optimised POMCP.

    Parameters
    ----------
    n_simulations : int — total MCTS rollouts (default 1000)
    n_particles : int — belief particles (default 200)
    max_depth : int (default 12)
    ucb_c : float (default 1.4)
    n_visit_threshold : int — visits before UCB (default 5)
    early_stop_margin : float — min Q-margin (default 0.5)
    early_stop_stability : int — consecutive stable checks (default 3)
    early_stop_check_every : int — sims between checks (default 50)
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
        early_stop_margin: float = 0.5,
        early_stop_stability: int = 3,
        early_stop_check_every: int = 50,
        rng: Optional[np.random.Generator] = None,
    ):
        self.T = transitions or DEFAULT_TRANSITIONS
        self.R = (reward_config or RewardConfig()).matrix()
        self.gamma = (reward_config or RewardConfig()).gamma

        self.n_simulations = int(n_simulations)
        self.n_particles = int(n_particles)
        self.max_depth = int(max_depth)
        self.ucb_c = float(ucb_c)
        self.n_visit_threshold = int(n_visit_threshold)
        self.early_stop_margin = float(early_stop_margin)
        self.early_stop_stability = int(early_stop_stability)
        self.early_stop_check_every = int(early_stop_check_every)

        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_info: Optional[FastSearchInfo] = None

        # Pre-compute T CDFs for fast sampling
        self._T_cdf = {
            a: np.cumsum(T_mat, axis=1).astype(np.float64)
            for a, T_mat in self.T.items()
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def search(self, belief: np.ndarray) -> int:
        t0 = time.perf_counter()

        particles = self.rng.choice(
            N_STATES, size=self.n_particles, p=belief,
        ).astype(np.int32)

        # ---- BATCH pre-sample ALL random data ----
        n_samples = self.n_simulations * self.max_depth
        # initial state indices
        init_idx = self.rng.integers(0, self.n_particles, size=self.n_simulations).astype(np.int32)
        # per-step: uniform for transition, uniform for observation, uniform for random action
        rng_batch = self.rng.random(size=(self.n_simulations, self.max_depth, 3)).astype(np.float32)
        # random actions for new-node exploration
        rand_actions = self.rng.integers(0, N_ACTIONS, size=(self.n_simulations, self.max_depth)).astype(np.int8)

        # ---- Build tree iteratively ----
        root = _FastNode()
        best_history: List[int] = []
        sims_run = 0
        early_stopped = False

        for sim_i in range(self.n_simulations):
            # ---- Early-stop check ----
            if sim_i > 0 and sim_i % self.early_stop_check_every == 0:
                best_a = int(np.argmax(root.Q_a))
                best_history.append(best_a)
                if len(best_history) >= self.early_stop_stability:
                    recent = best_history[-self.early_stop_stability:]
                    if len(set(recent)) == 1:
                        qs = sorted(root.Q_a, reverse=True)
                        if qs[0] - qs[1] > self.early_stop_margin:
                            early_stopped = True
                            break

            sims_run += 1
            s = int(particles[init_idx[sim_i]])
            node = root

            # Store (node, action, state_before_action) for backprop
            path: List[Tuple[_FastNode, int, int]] = []
            # Reserve capacity
            path_append = path.append

            rng_row = rng_batch[sim_i]
            act_row = rand_actions[sim_i]

            for d in range(self.max_depth):
                # ---- Action selection ----
                if node.N >= self.n_visit_threshold:
                    total_n = node.N + 1
                    best_a = -1
                    best_ucb = -np.inf
                    for a in range(N_ACTIONS):
                        if node.N_a[a] == 0:
                            best_a = a
                            break
                        explore = self.ucb_c * np.sqrt(np.log(total_n) / node.N_a[a])
                        ucb_val = node.Q_a[a] + explore
                        if ucb_val > best_ucb:
                            best_ucb = ucb_val
                            best_a = a
                    a = best_a
                else:
                    a = int(act_row[d])

                path_append((node, a, s))

                # ---- Generative model (searchsorted for fast sampling) ----
                u_s = rng_row[d, 0]
                t_cdf = self._T_cdf[a][s]
                s_next = int(np.searchsorted(t_cdf, u_s))
                if s_next >= N_STATES:
                    s_next = N_STATES - 1

                u_o = rng_row[d, 1]
                o_cdf = _O_CDF[:, s_next]
                o_idx = int(np.searchsorted(o_cdf, u_o))
                if o_idx >= _N_OBS:
                    o_idx = _N_OBS - 1

                # ---- Descend ----
                key = (a, o_idx)
                child = node.children.get(key)
                if child is None:
                    child = _FastNode()
                    node.children[key] = child

                s = s_next
                node = child

            # ---- Backprop ----
            cumulative_r = 0.0
            for node_p, a_p, s_p in reversed(path):
                r_val = float(self.R[s_p, a_p])
                cumulative_r = r_val + self.gamma * cumulative_r
                node_p.N += 1
                node_p.N_a[a_p] += 1
                node_p.Q_a[a_p] += (cumulative_r - node_p.Q_a[a_p]) / node_p.N_a[a_p]

        # ---- Final action ----
        best_a = int(np.argmax(root.Q_a))
        q_sorted = sorted(root.Q_a, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000

        self.last_info = FastSearchInfo(
            simulations_requested=self.n_simulations,
            simulations_run=sims_run,
            early_stopped=early_stopped,
            q_values={ACTION_NAMES_POMDP[a]: float(root.Q_a[a]) for a in range(N_ACTIONS)},
            best_action=ACTION_NAMES_POMDP[best_a],
            best_q=float(q_sorted[0]),
            runner_up_q=float(q_sorted[1]) if len(q_sorted) > 1 else 0.0,
            elapsed_ms=round(elapsed, 1),
        )

        return best_a

    def reset(self):
        self.last_info = None


# ---------------------------------------------------------------------------
# Lightweight node (no dataclass overhead)
# ---------------------------------------------------------------------------
class _FastNode:
    __slots__ = ("N", "N_a", "Q_a", "children")

    def __init__(self):
        self.N: int = 0
        self.N_a: np.ndarray = np.zeros(N_ACTIONS, dtype=np.int32)
        self.Q_a: np.ndarray = np.zeros(N_ACTIONS, dtype=np.float64)
        self.children: Dict[Tuple[int, int], _FastNode] = {}
