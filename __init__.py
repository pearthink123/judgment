"""
judgment — Math-Driven Agent Decision Engine + Harness.

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

from .harness import (
    JudgmentHarness,
    RunResult,
    SimulatedExecutor,
    LLMExecutor,
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
    train_hmm,
)

__all__ = [
    # Harness (main API)
    "JudgmentHarness",
    "RunResult",
    "SimulatedExecutor",
    "LLMExecutor",
    "ToolRegistry",
    "default_registry",
    # Core (power users)
    "DecisionEngine",
    "Decision",
    "HiddenMarkovModel",
    "HawkesProcess",
    "CUSUMDetector",
    "RewardConfig",
    "train_hmm",
]
