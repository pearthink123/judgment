"""
MathHarness Judgment Engine

A mathematically rigorous decision core for AI Agent Harnesses.
Uses stochastic processes, Bayesian inference, information theory and control theory
to drive reliable, quantifiable agent behavior instead of pure prompt heuristics.
"""

from .judgment_engine import JudgmentEngine, Decision
from .hawkes import HawkesProcess
from .bayesian import BayesianStateEstimator
from .info_gain import ExpectedValueOfInformation
from .control import PIDController, StochasticController

__all__ = [
    "JudgmentEngine",
    "Decision",
    "HawkesProcess",
    "BayesianStateEstimator",
    "ExpectedValueOfInformation",
    "PIDController",
    "StochasticController",
]
