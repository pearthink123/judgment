"""
Emission models — pluggable observation likelihoods for the HMM.

Two modes:
  1. DiscreteEmission — table lookup (current default, 7 dimensions)
  2. ContinuousEmission — Gaussian/Poisson/Bernoulli PDFs (no discretisation)

Both implement:
    log_prob(obs: dict) -> np.ndarray of shape (3,)  — log P(o|s) per state

Usage:
    from core.emission import ContinuousEmission

    em = ContinuousEmission()         # default params from discrete tables
    hmm = HiddenMarkovModel(emission_model=em)
    # hmm.forward_step() now uses continuous log-likelihoods
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .hmm import (
    N_STATES,
    EMISSION_TABLES,
    DIM_TOOL, DIM_PROGRESS, DIM_USER, DIM_ERROR, DIM_LENGTH, DIM_NOVELTY, DIM_NEGATION,
)


# ======================================================================
# Base class
# ======================================================================
class EmissionModel(ABC):
    """Abstract emission model — P(o | s)."""

    @abstractmethod
    def log_prob(self, obs: Dict[int, float]) -> np.ndarray:
        """
        log P(obs | s) for each state s ∈ {0, 1, 2}.

        Parameters
        ----------
        obs : dict {dim_index: value}
            Value type depends on the model:
              - Discrete: int category index
              - Continuous: float (probability for Bernoulli, count for Poisson,
                z-score or continuous value for Gaussian)

        Returns
        -------
        np.ndarray of shape (3,) — log-likelihood per state.
        """
        ...


# ======================================================================
# Discrete emission (current default, unchanged)
# ======================================================================
class DiscreteEmission(EmissionModel):
    """Table-lookup emission — the current default behaviour."""

    def __init__(self, tables: Optional[Dict[int, np.ndarray]] = None):
        self.tables = {k: v.copy() for k, v in (tables or EMISSION_TABLES).items()}
        self.log_tables = {
            dim: np.log(tbl + 1e-12) for dim, tbl in self.tables.items()
        }

    def log_prob(self, obs: Dict[int, float]) -> np.ndarray:
        """obs is {dim: category_index}."""
        log_lik = np.zeros(N_STATES, dtype=np.float64)
        for dim, cat in obs.items():
            cat_int = int(cat)
            if dim in self.log_tables and cat_int < self.log_tables[dim].shape[1]:
                log_lik += self.log_tables[dim][:, cat_int]
        return log_lik


# ======================================================================
# Continuous emission — Gaussian + Poisson + Bernoulli
# ======================================================================

# ---------- helper: log-PDF of standard distributions ----------

def _log_gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """log N(x | mu, sigma^2). sigma clamped > 1e-4."""
    s = max(sigma, 1e-4)
    return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x - mu) / s) ** 2


def _log_poisson_pmf(k: float, lam: float) -> float:
    """log Poisson(k | lambda). Uses Stirling for factorial (continuous-safe)."""
    k_int = max(0, int(round(k)))
    lam_clamped = max(lam, 1e-4)
    # log(k!) ≈ k*log(k) - k + 0.5*log(2*pi*k)  for k > 0
    if k_int == 0:
        log_fact = 0.0
    else:
        log_fact = k_int * np.log(k_int) - k_int + 0.5 * np.log(2 * np.pi * k_int)
    return k_int * np.log(lam_clamped) - lam_clamped - log_fact


def _log_bernoulli_pmf(x: float, p: float) -> float:
    """log Bernoulli(x | p). x ∈ [0, 1] treated as probability of success."""
    p_clamped = np.clip(p, 1e-6, 1 - 1e-6)
    # Treat x as a soft observation: P(obs | success) contribution
    # Binary case: x = 0.0 or 1.0 → exact Bernoulli
    # Continuous case: x ∈ (0,1) → weighted
    if x <= 0.0:
        return np.log(1 - p_clamped)
    elif x >= 1.0:
        return np.log(p_clamped)
    else:
        return x * np.log(p_clamped) + (1 - x) * np.log(1 - p_clamped)


def _log_beta_pdf(x: float, alpha: float, beta: float) -> float:
    """log Beta(x | alpha, beta). x ∈ (0, 1)."""
    x_c = np.clip(x, 1e-6, 1 - 1e-6)
    a = max(alpha, 0.01)
    b = max(beta, 0.01)
    # log B(a,b) = log Γ(a) + log Γ(b) - log Γ(a+b)
    # Approximate log Γ via scipy if available, else Stirling
    try:
        from scipy.special import betaln
        log_B = betaln(a, b)
    except ImportError:
        # Stirling: log Γ(z) ≈ (z-0.5)log(z) - z + 0.5*log(2π)
        def _log_gamma(z):
            return (z - 0.5) * np.log(z) - z + 0.5 * np.log(2 * np.pi)
        log_B = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)

    return (a - 1) * np.log(x_c) + (b - 1) * np.log(1 - x_c) - log_B


# ---------- ContinuousEmissionConfig ----------

class ContinuousEmissionConfig:
    """Parameters for each dimension's continuous distribution, per state.

    All arrays are shape (3,) — one value per state {H, D, B}.
    """

    def __init__(
        self,
        tool_p: Optional[np.ndarray] = None,
        progress_mu: Optional[np.ndarray] = None,
        progress_sigma: Optional[np.ndarray] = None,
        user_p: Optional[np.ndarray] = None,
        error_lambda: Optional[np.ndarray] = None,
        length_mu: Optional[np.ndarray] = None,
        length_sigma: Optional[np.ndarray] = None,
        novelty_alpha: Optional[np.ndarray] = None,
        novelty_beta: Optional[np.ndarray] = None,
        negation_lambda: Optional[np.ndarray] = None,
    ):
        self.tool_p = tool_p if tool_p is not None else np.array([0.80, 0.50, 0.25])
        self.progress_mu = progress_mu if progress_mu is not None else np.array([0.15, 0.05, -0.02])
        self.progress_sigma = progress_sigma if progress_sigma is not None else np.array([0.08, 0.10, 0.12])
        self.user_p = user_p if user_p is not None else np.array([0.05, 0.15, 0.25])
        self.error_lambda = error_lambda if error_lambda is not None else np.array([0.10, 0.80, 2.50])
        self.length_mu = length_mu if length_mu is not None else np.array([0.0, 0.2, 0.5])
        self.length_sigma = length_sigma if length_sigma is not None else np.array([0.8, 1.0, 1.2])
        self.novelty_alpha = novelty_alpha if novelty_alpha is not None else np.array([8.0, 4.0, 2.0])
        self.novelty_beta = novelty_beta if novelty_beta is not None else np.array([2.0, 2.5, 3.0])
        self.negation_lambda = negation_lambda if negation_lambda is not None else np.array([0.5, 2.0, 5.0])


# ---------- ContinuousEmission ----------

class ContinuousEmission(EmissionModel):
    """
    Continuous emission model — Gaussian, Poisson, Bernoulli, Beta PDFs.

    Each observation dimension is modelled with a continuous distribution
    parameterised per hidden state.  No discretisation — the full signal
    strength is preserved.

    Parameters
    ----------
    config : ContinuousEmissionConfig or None
        Distribution parameters per state.  If None, defaults are used
        (derived from the discrete emission tables).
    """

    def __init__(self, config: Optional[ContinuousEmissionConfig] = None):
        self.cfg = config or ContinuousEmissionConfig()

    def log_prob(self, obs: Dict[int, float]) -> np.ndarray:
        """
        obs keys are dimension indices, values are floats:
          0: tool_ok (0.0 or 1.0, or soft probability in (0,1))
          1: progress_delta (continuous, typically [-0.3, 0.5])
          2: user_msg (0.0 or 1.0)
          3: error_count_delta (non-negative int or float)
          4: length_z (continuous z-score)
          5: token_novelty (0-1 continuous ratio)
          6: negation count (non-negative int or float)

        Returns (3,) log-likelihood per state.
        """
        log_lik = np.zeros(N_STATES, dtype=np.float64)

        # Dim 0: tool_ok — Bernoulli
        if DIM_TOOL in obs:
            x = float(obs[DIM_TOOL])
            for s in range(N_STATES):
                log_lik[s] += _log_bernoulli_pmf(x, self.cfg.tool_p[s])

        # Dim 1: progress_delta — Gaussian
        if DIM_PROGRESS in obs:
            x = float(obs[DIM_PROGRESS])
            for s in range(N_STATES):
                log_lik[s] += _log_gaussian_pdf(x, self.cfg.progress_mu[s], self.cfg.progress_sigma[s])

        # Dim 2: user_msg — Bernoulli
        if DIM_USER in obs:
            x = float(obs[DIM_USER])
            for s in range(N_STATES):
                log_lik[s] += _log_bernoulli_pmf(x, self.cfg.user_p[s])

        # Dim 3: error_count_delta — Poisson
        if DIM_ERROR in obs:
            x = float(obs[DIM_ERROR])
            for s in range(N_STATES):
                log_lik[s] += _log_poisson_pmf(x, self.cfg.error_lambda[s])

        # Dim 4: length_z — Gaussian
        if DIM_LENGTH in obs:
            x = float(obs[DIM_LENGTH])
            for s in range(N_STATES):
                log_lik[s] += _log_gaussian_pdf(x, self.cfg.length_mu[s], self.cfg.length_sigma[s])

        # Dim 5: token_novelty — Beta
        if DIM_NOVELTY in obs:
            x = float(obs[DIM_NOVELTY])
            for s in range(N_STATES):
                log_lik[s] += _log_beta_pdf(x, self.cfg.novelty_alpha[s], self.cfg.novelty_beta[s])

        # Dim 6: negation count — Poisson
        if DIM_NEGATION in obs:
            x = float(obs[DIM_NEGATION])
            for s in range(N_STATES):
                log_lik[s] += _log_poisson_pmf(x, self.cfg.negation_lambda[s])

        return log_lik


# ======================================================================
# Observation encoder for continuous mode
# ======================================================================
def encode_observation_continuous(
    tool_ok: bool,
    progress_delta: float,
    has_user_msg: bool,
    error_count_delta: int,
    content_signals: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """
    Encode raw observation into continuous values for ContinuousEmission.

    Unlike encode_observation() which returns discrete category indices,
    this returns the raw float values directly.

    Parameters
    ----------
    tool_ok : bool
    progress_delta : float — raw progress delta
    has_user_msg : bool
    error_count_delta : int — raw error count
    content_signals : dict or None
        {4: length_z, 5: novelty_ratio, 6: negation_count}

    Returns
    -------
    dict {dim_index: float}
    """
    result: Dict[int, float] = {
        DIM_TOOL: 1.0 if tool_ok else 0.0,
        DIM_PROGRESS: float(progress_delta),
        DIM_USER: 1.0 if has_user_msg else 0.0,
        DIM_ERROR: float(error_count_delta),
    }
    if content_signals:
        result.update(content_signals)
    return result
