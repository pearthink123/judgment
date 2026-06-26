"""
Content quality signals — lightweight, no-embedding text health metrics.

Three signals extracted from the LLM's text output per step:

  1. Length z-score   — relative to recent history; detects unusually
                        short/long responses (sign of derailment)
  2. Token novelty     — unique/total token ratio; low → verbatim repetition
  3. Negation surge    — count of negation words; spike → self-correction

All signals are discretised for HMM consumption.  No heavy dependencies
are required (pure token-splitting, no embeddings).

Usage:
    extractor = ContentSignalExtractor(window_size=10)
    signals = extractor.extract("The LLM output text for this step")
    # signals = {4: 1, 5: 1, 6: 0}   # dim → category index
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Extract alphanumeric tokens (ascii-safe, fast)."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Negation word list (English, extendable)
# ---------------------------------------------------------------------------
NEGATION_WORDS = frozenset({
    "no", "not", "never", "none", "neither", "nor",
    "don", "doesn", "didn", "won", "wouldn", "shouldn",
    "can", "cannot", "couldn", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "mustn",
    "wrong", "incorrect", "error", "mistake", "false", "invalid",
    "contrary", "however", "but", "although", "instead",
    "actually", "rather", "oops", "sorry", "apologize",
})


# ---------------------------------------------------------------------------
# Discretisation bins
# ---------------------------------------------------------------------------
LENGTH_BINS = {
    "low": 0,
    "normal": 1,
    "high": 2,
}

NOVELTY_BINS = {
    "repetitive": 0,
    "normal": 1,
    "fresh": 2,
}

NEGATION_BINS = {
    "normal": 0,
    "elevated": 1,
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class ContentSignalExtractor:
    """
    Lightweight content-quality signal extractor.

    Parameters
    ----------
    window_size : int
        Number of recent steps used for length z-score baseline (default 10).
    novelty_low_threshold : float
        Below this unique/token ratio → "repetitive" (default 0.40).
    novelty_high_threshold : float
        Above this ratio → "fresh" (default 0.85).
    negation_normal_max : int
        More negation tokens than this → "elevated" (default 3).
    length_z_threshold : float
        |z| above this → flagged as anomalous length (default 2.0).
    """

    # HMM dimension indices for content signals
    DIM_LENGTH = 4
    DIM_NOVELTY = 5
    DIM_NEGATION = 6

    def __init__(
        self,
        window_size: int = 10,
        novelty_low_threshold: float = 0.40,
        novelty_high_threshold: float = 0.85,
        negation_normal_max: int = 3,
        length_z_threshold: float = 2.0,
    ):
        self.window_size = int(window_size)
        self.novelty_low = float(novelty_low_threshold)
        self.novelty_high = float(novelty_high_threshold)
        self.negation_max = int(negation_normal_max)
        self.z_threshold = float(length_z_threshold)

        # Rolling length history
        self._lengths: deque[int] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def extract(self, text: Optional[str]) -> Dict[int, int]:
        """
        Compute content signal categories for one LLM output.

        Parameters
        ----------
        text : str or None
            The LLM's text output for this step.  If None or empty,
            all signals default to "normal" (1 or 0).

        Returns
        -------
        dict mapping HMM dimension index → category index:
            {4: length_cat, 5: novelty_cat, 6: negation_cat}
        """
        if not text:
            # No text — all normal
            self._lengths.append(0)
            return {
                self.DIM_LENGTH: LENGTH_BINS["normal"],
                self.DIM_NOVELTY: NOVELTY_BINS["normal"],
                self.DIM_NEGATION: NEGATION_BINS["normal"],
            }

        tokens = tokenize(text)
        n_tokens = len(tokens)

        # 1. Length z-score
        length_cat = self._compute_length(n_tokens)
        self._lengths.append(n_tokens)

        # 2. Token novelty
        novelty_cat = self._compute_novelty(tokens, n_tokens)

        # 3. Negation surge
        negation_cat = self._compute_negation(tokens)

        return {
            self.DIM_LENGTH: length_cat,
            self.DIM_NOVELTY: novelty_cat,
            self.DIM_NEGATION: negation_cat,
        }

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _compute_length(self, n_tokens: int) -> int:
        if len(self._lengths) < 2:
            return LENGTH_BINS["normal"]

        mean_len = sum(self._lengths) / len(self._lengths)
        variance = sum((l - mean_len) ** 2 for l in self._lengths) / len(self._lengths)
        std = max(math.sqrt(variance), 1.0)  # avoid division by zero

        z = (n_tokens - mean_len) / std

        if z > self.z_threshold:
            return LENGTH_BINS["high"]
        elif z < -self.z_threshold:
            return LENGTH_BINS["low"]
        else:
            return LENGTH_BINS["normal"]

    def _compute_novelty(self, tokens: List[str], n_tokens: int) -> int:
        if n_tokens < 3:
            return NOVELTY_BINS["normal"]

        unique_ratio = len(set(tokens)) / n_tokens

        if unique_ratio < self.novelty_low:
            return NOVELTY_BINS["repetitive"]
        elif unique_ratio > self.novelty_high:
            return NOVELTY_BINS["fresh"]
        else:
            return NOVELTY_BINS["normal"]

    def _compute_negation(self, tokens: List[str]) -> int:
        neg_count = sum(1 for t in tokens if t in NEGATION_WORDS)
        if neg_count > self.negation_max:
            return NEGATION_BINS["elevated"]
        return NEGATION_BINS["normal"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset(self):
        self._lengths.clear()

    @property
    def recent_mean_length(self) -> float:
        if not self._lengths:
            return 0.0
        return sum(self._lengths) / len(self._lengths)
