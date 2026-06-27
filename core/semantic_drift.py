"""
Semantic drift detection — embedding-based LLM output similarity.

Optional module. Detects subtle derailment that structural signals miss:
the agent is still calling tools successfully, still producing normal-length
output — but what it's saying is progressively less coherent.

Uses cosine similarity between consecutive outputs to flag "drift events":
a drop below the similarity threshold suggests the agent has gone off-topic.

Requires: pip install judgment[semantic]  or  pip install sentence-transformers

Usage:
    from core.semantic_drift import SemanticDriftDetector
    detector = SemanticDriftDetector()
    score = detector.check("previous output", "current output")
    # score ∈ [0, 1], < 0.5 → potential drift
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Deque
from collections import deque
import warnings


class SemanticDriftDetector:
    """
    Embedding-based semantic drift monitor.

    Caches the last N output embeddings and flags when cosine similarity
    between consecutive outputs drops below a threshold.  This catches
    "silent derailment" — structural signals look fine but the agent is
    saying progressively less relevant things.

    Parameters
    ----------
    model_name : str — sentence-transformers model. Default is lightweight
                       "all-MiniLM-L6-v2" (80MB, 384-dim).
    threshold : float — cosine similarity below which a drift event is
                        flagged (default 0.55).
    window_size : int — number of recent outputs to keep for baseline
                        similarity (default 5).
    min_tokens : int — skip embedding for very short outputs (default 10).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.55,
        window_size: int = 5,
        min_tokens: int = 10,
    ):
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.min_tokens = int(min_tokens)
        self.model_name = model_name

        self._model = None
        self._embeddings: Deque[np.ndarray] = deque(maxlen=window_size)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy init — sentence-transformers is optional
    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic drift detection. "
                    "Install: pip install sentence-transformers"
                )
        return self._model

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def check(self, previous_output: str, current_output: str) -> float:
        """
        Compute cosine similarity between two outputs.

        Returns
        -------
        float ∈ [0, 1] — similarity score. < threshold → potential drift.
        """
        if len(current_output.split()) < self.min_tokens:
            return 1.0  # too short to judge

        if len(previous_output.split()) < self.min_tokens:
            return 1.0

        try:
            emb1, emb2 = self.model.encode([previous_output, current_output])
            sim = float(_cosine_similarity(emb1, emb2))
            return max(0.0, min(1.0, sim))
        except Exception:
            return 1.0  # on error, assume no drift

    def step(self, output: str) -> Dict[str, float]:
        """
        Process one output. Returns drift metrics for the engine.

        Returns
        -------
        dict with keys:
            similarity  — most recent pairwise cosine similarity
            drift_flag  — 1.0 if below threshold, 0.0 otherwise
            mean_sim    — mean similarity over the window
            n_cached    — number of embeddings in the sliding window
        """
        tokens = output.split()
        if len(tokens) < self.min_tokens:
            return {"similarity": 1.0, "drift_flag": 0.0, "mean_sim": 1.0, "n_cached": len(self._embeddings)}

        try:
            emb = self.model.encode([output])[0]
        except Exception:
            return {"similarity": 1.0, "drift_flag": 0.0, "mean_sim": 1.0, "n_cached": len(self._embeddings)}

        similarity = 1.0
        if self._embeddings:
            prev = self._embeddings[-1]
            similarity = float(_cosine_similarity(prev, emb))
            similarity = max(0.0, min(1.0, similarity))

        self._embeddings.append(emb)

        mean_sim = 1.0
        if len(self._embeddings) >= 2:
            sims = []
            embs = list(self._embeddings)
            for i in range(1, len(embs)):
                sims.append(float(_cosine_similarity(embs[i - 1], embs[i])))
            mean_sim = float(np.mean(sims)) if sims else 1.0

        drift = 0.0 if similarity >= self.threshold else 1.0

        return {
            "similarity": round(similarity, 4),
            "drift_flag": drift,
            "mean_sim": round(mean_sim, 4),
            "n_cached": len(self._embeddings),
        }

    def reset(self):
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# ContentSignalExtractor integration — returns a dict ready for encode_observation
# ---------------------------------------------------------------------------
def semantic_signal_to_hmm(drift_result: Dict[str, float]) -> Dict[int, int]:
    """
    Convert a SemanticDriftDetector.step() result into HMM category indices.

    Uses the existing content signal dimension slots:
      DIM_LENGTH (4): semantic drift → low similarity maps to "low" bin
      DIM_NOVELTY (5): if drift, novelty is "repetitive" (the agent is looping)

    This means you can swap out the text-metrics-based ContentSignalExtractor
    for a semantic detector without changing the HMM.

    Returns {dim: category} suitable for encode_observation(content_signals=...).
    """
    result = {}
    sim = drift_result.get("similarity", 1.0)
    drift = drift_result.get("drift_flag", 0.0)

    # Map similarity to length bins: low sim → low bin (anomalous length)
    if sim < 0.40:
        result[4] = 0  # low
    elif sim < 0.65:
        result[4] = 1  # normal
    else:
        result[4] = 2  # high

    # Drift flag → novelty: drift = repetitive
    if drift > 0.5:
        result[5] = 0  # repetitive
    else:
        result[5] = 1  # normal

    # Negation: not directly detectable from embeddings
    result[6] = 0  # normal

    return result
