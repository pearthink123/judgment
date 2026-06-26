"""
MathHarness Judgment Engine

A mathematically rigorous decision core for AI Agent Harnesses.

Architecture:
  Layer 1: CUSUM anomaly detection (Hawkes-corrected surprisal)
  Layer 2: 3-state HMM latent-state inference (Healthy/Degraded/Broken)
  Layer 3: Threshold-gate decision (Continue/Correct/Escalate/Gather)

Each layer has a closed-form mathematical foundation with citable references.
"""

from .engine import (
    DecisionEngine,
    Decision,
    ACTION_CONTINUE,
    ACTION_CORRECT,
    ACTION_ESCALATE,
    ACTION_GATHER,
)
from .hawkes import (
    HawkesProcess,
    HawkesEvent,
    HawkesDiagnostics,
    EVENT_SUCCESS,
    EVENT_ERROR,
    EVENT_USER,
    EVENT_TOOL,
    EVENT_NAMES,
)
from .hmm import (
    HiddenMarkovModel,
    encode_observation,
    STATE_HEALTHY,
    STATE_DEGRADED,
    STATE_BROKEN,
    STATE_NAMES,
)
from .cusum import CUSUMDetector
from . import diagnostics

__all__ = [
    # Engine
    "DecisionEngine",
    "Decision",
    "ACTION_CONTINUE",
    "ACTION_CORRECT",
    "ACTION_ESCALATE",
    "ACTION_GATHER",
    # Hawkes
    "HawkesProcess",
    "HawkesEvent",
    "HawkesDiagnostics",
    "EVENT_SUCCESS",
    "EVENT_ERROR",
    "EVENT_USER",
    "EVENT_TOOL",
    "EVENT_NAMES",
    # HMM
    "HiddenMarkovModel",
    "encode_observation",
    "STATE_HEALTHY",
    "STATE_DEGRADED",
    "STATE_BROKEN",
    "STATE_NAMES",
    # CUSUM
    "CUSUMDetector",
    # Diagnostics
    "diagnostics",
]
