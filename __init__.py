"""
judgment — Math-Driven Agent Decision Engine + Harness.

Version: 0.2.0

Quick start:
    from judgment import JudgmentHarness

    harness = JudgmentHarness()
    result = harness.run("Implement an LRU cache in Python")
    print(result.summary)

CLI:
    pip install judgment
    judgment run "Your task here"
    judgment train ./logs/
    judgment dashboard
"""

from .core.version import __version__

from .harness import (
    JudgmentHarness,
    RunResult,
    SimulatedExecutor,
    LLMExecutor,
    AnthropicExecutor,
    ToolRegistry,
    default_registry,
)

from .core import (
    DecisionEngine,
    Decision,
    HiddenMarkovModel,
    HawkesProcess,
    CUSUMDetector,
    RewardConfig,
    EngineConfig,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER, ACTION_REPLAN,
    train_hmm,
)

__all__ = [
    # Harness (main API)
    "JudgmentHarness",
    "RunResult",
    "SimulatedExecutor",
    "LLMExecutor",
    "AnthropicExecutor",
    "ToolRegistry",
    "default_registry",
    # Core (power users)
    "DecisionEngine",
    "Decision",
    "HiddenMarkovModel",
    "HawkesProcess",
    "CUSUMDetector",
    "RewardConfig",
    "EngineConfig",
    "ACTION_CONTINUE", "ACTION_CORRECT", "ACTION_ESCALATE",
    "ACTION_GATHER", "ACTION_REPLAN",
    "train_hmm",
]
