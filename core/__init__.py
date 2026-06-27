"""
Math-driven decision core — 3-layer architecture.

Layer 1: CUSUM anomaly detection (Hawkes-corrected surprisal)
Layer 2: 3-state HMM latent-state inference (structural + content signals)
Layer 3: POMDP action selection (POMCP online MCTS or grid value iteration)
"""

from .engine import (
    DecisionEngine,
    Decision,
    ACTION_CONTINUE,
    ACTION_CORRECT,
    ACTION_ESCALATE,
    ACTION_GATHER,
    ACTION_REPLAN,
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
from .pomdp import (
    POMDPPolicy,
    RewardConfig,
    solve_belief_mdp,
    get_policy,
)
from .pomcp import POMCPPlanner, POMCPSearchInfo
from .content_signals import ContentSignalExtractor
from .corrective import (
    CorrectiveRouter,
    CorrectiveAdvice,
    CORRECTIVE_VERIFY,
    CORRECTIVE_RETHINK,
    CORRECTIVE_RETRY,
    CORRECTIVE_ROLLBACK,
)
from .training import train_hmm, baum_welch
from .config import EngineConfig
from . import diagnostics

__all__ = [
    # Engine
    "DecisionEngine",
    "Decision",
    "ACTION_CONTINUE",
    "ACTION_CORRECT",
    "ACTION_ESCALATE",
    "ACTION_GATHER",
    "ACTION_REPLAN",
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
    # POMDP
    "POMDPPolicy",
    "RewardConfig",
    "solve_belief_mdp",
    "get_policy",
    # POMCP
    "POMCPPlanner",
    "POMCPSearchInfo",
    # Content signals
    "ContentSignalExtractor",
    # Corrective
    "CorrectiveRouter",
    "CorrectiveAdvice",
    "CORRECTIVE_VERIFY",
    "CORRECTIVE_RETHINK",
    "CORRECTIVE_RETRY",
    "CORRECTIVE_ROLLBACK",
    # Config
    "EngineConfig",
    # Training
    "train_hmm",
    "baum_welch",
    # Diagnostics
    "diagnostics",
]
