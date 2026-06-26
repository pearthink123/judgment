"""
Baum-Welch (EM) training for HMM parameters — Layer 2 learning.

Learns the transition matrix T and emission tables B from agent run logs.
Supports semi-supervised mode: a small number of state labels anchor the
semantics of Healthy / Degraded / Broken so the unsupervised EM doesn't
permute them.

All computation is in log-space for numerical stability.

Reference:
  Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models."
  Proceedings of the IEEE, 77(2), 257–286.  §III-C.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .hmm import (
    HiddenMarkovModel,
    encode_observation,
    N_STATES,
    STATE_HEALTHY,
    STATE_DEGRADED,
    STATE_BROKEN,
    EMISSION_TABLES,
    DIM_TOOL, DIM_PROGRESS, DIM_USER, DIM_ERROR,
)


# ---------------------------------------------------------------------------
# Training data helpers
# ---------------------------------------------------------------------------
def observation_from_dict(obs: Dict) -> Dict[int, int]:
    """Convert a raw obs dict to HMM categorical encoding."""
    return encode_observation(
        tool_ok=bool(obs.get("tool_ok", True)),
        progress_delta=float(obs.get("progress_delta", 0.0)),
        has_user_msg=bool(obs.get("has_user_msg", False)),
        error_count_delta=int(obs.get("error_count_delta", 0)),
    )


def build_observation_sequences(
    logs: List[List[Dict]],
) -> List[List[Dict[int, int]]]:
    """
    Convert raw log data into HMM observation sequences.

    Parameters
    ----------
    logs : list of trajectories
        Each trajectory is a list of observation dicts (one per step).

    Returns
    -------
    List of sequences, each being a list of encoded observation dicts.
    """
    return [[observation_from_dict(obs) for obs in traj] for traj in logs]


# ---------------------------------------------------------------------------
# Log-sum-exp helpers
# ---------------------------------------------------------------------------
def _logsumexp(x: np.ndarray) -> float:
    c = x.max()
    if np.isinf(c) and c < 0:
        return -np.inf
    return float(c + np.log(np.sum(np.exp(x - c))))


# ---------------------------------------------------------------------------
# Baum-Welch
# ---------------------------------------------------------------------------
def baum_welch(
    sequences: List[List[Dict[int, int]]],
    labels: Optional[List[Dict[int, int]]] = None,
    n_iter: int = 50,
    tol: float = 1e-4,
    init_prior: Optional[np.ndarray] = None,
    init_T: Optional[np.ndarray] = None,
    init_B: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray, List[float]]:
    """
    Baum-Welch EM algorithm for HMM parameter estimation.

    Parameters
    ----------
    sequences : list of encoded observation sequences
        Each is a list of {dim: cat} dicts.
    labels : optional list of {step_idx: state_idx} dicts
        Semi-supervised anchors.  When a label is provided for step t,
        γ_t(i) is clamped to 1 for the labelled state and 0 otherwise.
    n_iter : int
        Max EM iterations.
    tol : float
        Convergence threshold on log-likelihood.
    init_prior, init_T, init_B : optional initial parameters.

    Returns
    -------
    prior : np.ndarray (3,)
    T : np.ndarray (3, 3)
    B : dict {dim: np.ndarray (3, n_cats)}
    log_lik_history : list of float
    """
    n_seqs = len(sequences)
    labels = labels or [{} for _ in range(n_seqs)]

    # --- initialise ---
    if init_prior is not None:
        prior = init_prior.copy()
    else:
        prior = np.array([0.65, 0.28, 0.07], dtype=np.float64)

    if init_T is not None:
        T = init_T.copy()
    else:
        T = np.array([
            [0.80, 0.17, 0.03],
            [0.15, 0.65, 0.20],
            [0.02, 0.10, 0.88],
        ], dtype=np.float64)

    if init_B is not None:
        B = {dim: tbl.copy() for dim, tbl in init_B.items()}
    else:
        B = {dim: tbl.copy() for dim, tbl in EMISSION_TABLES.items()}

    # Small smoothing to avoid zeros
    for dim in B:
        B[dim] = np.maximum(B[dim], 1e-4)
        B[dim] = B[dim] / B[dim].sum(axis=1, keepdims=True)

    # Pre-compute dimensions
    n_cats_per_dim = {dim: B[dim].shape[1] for dim in B}

    log_lik_history: List[float] = []

    for it in range(n_iter):
        # --- accumulate expected counts ---
        expected_T_num = np.zeros((N_STATES, N_STATES), dtype=np.float64)
        expected_T_den = np.zeros(N_STATES, dtype=np.float64)
        expected_prior_num = np.zeros(N_STATES, dtype=np.float64)

        expected_B_num: Dict[int, np.ndarray] = {
            dim: np.zeros((N_STATES, n_cats_per_dim[dim]), dtype=np.float64)
            for dim in B
        }
        expected_B_den = np.zeros(N_STATES, dtype=np.float64)

        total_log_lik = 0.0

        for seq_idx, obs_seq in enumerate(sequences):
            T_steps = len(obs_seq)
            if T_steps < 2:
                continue

            seq_labels = labels[seq_idx]

            # ---- log-space emission lookup ----
            log_B_seq = np.zeros((T_steps, N_STATES), dtype=np.float64)
            for t_idx, obs_cats in enumerate(obs_seq):
                for dim, cat in obs_cats.items():
                    log_B_seq[t_idx] += np.log(B[dim][:, cat] + 1e-12)

            # ---- Forward ----
            log_alpha = np.zeros((T_steps, N_STATES), dtype=np.float64)
            log_alpha[0] = np.log(prior + 1e-12) + log_B_seq[0]
            for t_idx in range(1, T_steps):
                for s_to in range(N_STATES):
                    terms = log_alpha[t_idx - 1] + np.log(T[:, s_to] + 1e-12)
                    log_alpha[t_idx, s_to] = log_B_seq[t_idx, s_to] + _logsumexp(terms)

            # ---- Backward ----
            log_beta = np.zeros((T_steps, N_STATES), dtype=np.float64)
            # log_beta[T-1] = log(1) = 0
            for t_idx in range(T_steps - 2, -1, -1):
                for s_from in range(N_STATES):
                    terms = (
                        np.log(T[s_from, :] + 1e-12)
                        + log_B_seq[t_idx + 1]
                        + log_beta[t_idx + 1]
                    )
                    log_beta[t_idx, s_from] = _logsumexp(terms)

            # ---- Sequence log-likelihood ----
            ll = _logsumexp(log_alpha[-1])
            total_log_lik += ll

            # ---- Gamma (state occupation) ----
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)

            # ---- Apply semi-supervised labels ----
            for t_idx, labelled_state in seq_labels.items():
                if 0 <= t_idx < T_steps:
                    gamma[t_idx] = 0.0
                    gamma[t_idx, labelled_state] = 1.0

            # ---- Xi (state-pair occupation) ----
            xi = np.zeros((T_steps - 1, N_STATES, N_STATES), dtype=np.float64)
            for t_idx in range(T_steps - 1):
                for s_from in range(N_STATES):
                    for s_to in range(N_STATES):
                        xi[t_idx, s_from, s_to] = (
                            np.exp(
                                log_alpha[t_idx, s_from]
                                + np.log(T[s_from, s_to] + 1e-12)
                                + log_B_seq[t_idx + 1, s_to]
                                + log_beta[t_idx + 1, s_to]
                                - ll
                            )
                        )

            # ---- Accumulate expected counts ----
            expected_prior_num += gamma[0]

            for s_from in range(N_STATES):
                for s_to in range(N_STATES):
                    expected_T_num[s_from, s_to] += xi[:, s_from, s_to].sum()

                expected_T_den[s_from] += gamma[:-1, s_from].sum()

            for t_idx in range(T_steps):
                for dim in B:
                    cat = obs_seq[t_idx].get(dim)
                    if cat is not None:
                        expected_B_num[dim][:, cat] += gamma[t_idx]
                    expected_B_den += gamma[t_idx]

        # ---- M-step ----
        # prior
        prior_new = expected_prior_num / expected_prior_num.sum()
        prior_new = np.maximum(prior_new, 1e-4)
        prior_new = prior_new / prior_new.sum()

        # transition
        T_new = np.zeros_like(T)
        for s_from in range(N_STATES):
            denom = expected_T_den[s_from]
            if denom > 1e-10:
                T_new[s_from] = expected_T_num[s_from] / denom
            else:
                T_new[s_from] = T[s_from]  # keep old
            T_new[s_from] = np.maximum(T_new[s_from], 1e-4)
            T_new[s_from] = T_new[s_from] / T_new[s_from].sum()

        # emission
        B_new: Dict[int, np.ndarray] = {}
        for dim in B:
            B_new[dim] = np.zeros_like(B[dim])
            for s in range(N_STATES):
                denom = expected_B_den[s]
                if denom > 1e-10:
                    B_new[dim][s] = expected_B_num[dim][s] / denom
                else:
                    B_new[dim][s] = B[dim][s]
                B_new[dim][s] = np.maximum(B_new[dim][s], 1e-4)
                B_new[dim][s] = B_new[dim][s] / B_new[dim][s].sum()

        # ---- Check convergence ----
        log_lik_history.append(total_log_lik)

        if it > 2:
            delta = log_lik_history[-1] - log_lik_history[-2]
            if abs(delta) < tol:
                prior = prior_new
                T = T_new
                B = B_new
                break

        prior = prior_new
        T = T_new
        B = B_new

    return prior, T, B, log_lik_history


# ---------------------------------------------------------------------------
# Convenience: train and create a new HMM
# ---------------------------------------------------------------------------
def train_hmm(
    logs: List[List[Dict]],
    labels: Optional[List[Dict[int, int]]] = None,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> HiddenMarkovModel:
    """
    Train an HMM from raw agent logs.

    Parameters
    ----------
    logs : list of trajectories
        Each trajectory is a list of raw observation dicts.
    labels : optional semi-supervised labels
        list of {step_idx: state_idx} per trajectory.
    n_iter, tol : EM hyperparameters.

    Returns
    -------
    HiddenMarkovModel with learned parameters.
    """
    sequences = build_observation_sequences(logs)
    prior, T, B, _ = baum_welch(
        sequences, labels=labels, n_iter=n_iter, tol=tol,
    )
    return HiddenMarkovModel(prior=prior, transition=T, emission_tables=B)
