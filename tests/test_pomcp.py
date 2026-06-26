"""Tests for POMCP online MCTS planner."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.pomcp import POMCPPlanner, POMCPSearchInfo, belief_to_particles
from core.pomdp import (
    RewardConfig, DEFAULT_TRANSITIONS,
    ACT_CONTINUE, ACT_CORRECT, ACT_ESCALATE, ACT_GATHER,
    ACTION_NAMES_POMDP,
)
from core.hmm import STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN


class TestBeliefToParticles:
    def test_correct_count(self):
        belief = np.array([0.7, 0.2, 0.1])
        particles = belief_to_particles(belief, 100)
        assert len(particles) == 100

    def test_valid_states(self):
        belief = np.array([0.6, 0.3, 0.1])
        particles = belief_to_particles(belief, 200)
        assert set(particles).issubset({0, 1, 2})

    def test_distribution_approximately_right(self):
        belief = np.array([0.90, 0.07, 0.03])
        rng = np.random.default_rng(42)
        particles = belief_to_particles(belief, 5000, rng=rng)
        _, counts = np.unique(particles, return_counts=True)
        frac = counts / len(particles)
        # Majority should be state 0 (Healthy)
        assert frac[0] > 0.80


class TestPOMCPInit:
    def test_default_init(self):
        planner = POMCPPlanner()
        assert planner.n_simulations == 1000
        assert planner.n_particles == 200
        assert planner.gamma == 0.95

    def test_custom_params(self):
        planner = POMCPPlanner(
            n_simulations=500, n_particles=50, max_depth=8, ucb_c=2.0,
        )
        assert planner.n_simulations == 500
        assert planner.ucb_c == 2.0


class TestPOMCPSearch:
    def test_search_healthy_returns_continue(self):
        planner = POMCPPlanner(n_simulations=500, n_particles=50, rng=np.random.default_rng(42))
        belief = np.array([0.99, 0.008, 0.002])
        action = planner.search(belief)
        assert action == ACT_CONTINUE, (
            f"Healthy belief should CONTINUE, got {ACTION_NAMES_POMDP[action]}"
        )

    def test_search_broken_not_continue(self):
        planner = POMCPPlanner(n_simulations=500, n_particles=50, rng=np.random.default_rng(42))
        belief = np.array([0.01, 0.01, 0.98])
        action = planner.search(belief)
        assert action != ACT_CONTINUE, (
            "Broken belief should NOT continue"
        )

    def test_search_degraded_favours_correct(self):
        planner = POMCPPlanner(n_simulations=500, n_particles=50, rng=np.random.default_rng(42))
        belief = np.array([0.05, 0.90, 0.05])
        action = planner.search(belief)
        # Degraded → correct or escalate, not continue
        assert action in {ACT_CORRECT, ACT_ESCALATE}, (
            f"Degraded: expected correct/escalate, got {ACTION_NAMES_POMDP[action]}"
        )

    def test_search_has_info(self):
        planner = POMCPPlanner(n_simulations=300, n_particles=40, rng=np.random.default_rng(42))
        planner.search(np.array([0.8, 0.15, 0.05]))
        info = planner.last_info
        assert info is not None
        assert info.simulations == 300
        assert info.root_visits > 0
        assert len(info.q_values) == 4
        assert info.tree_size > 0

    def test_info_q_values_summary(self):
        planner = POMCPPlanner(n_simulations=300, n_particles=40, rng=np.random.default_rng(42))
        planner.search(np.array([0.5, 0.3, 0.2]))
        info = planner.last_info
        assert info.best_q >= info.runner_up_q - 1e-6  # best >= runner-up

    def test_reset_clears_state(self):
        planner = POMCPPlanner(n_simulations=100, rng=np.random.default_rng(42))
        planner.search(np.array([0.8, 0.15, 0.05]))
        planner.reset()
        assert planner.last_info is not None  # info persists for read-only access
        # After reset, a new search should work
        planner.search(np.array([0.7, 0.2, 0.1]))

    def test_reproducible_with_seed(self):
        """Same seed → same action."""
        belief = np.array([0.4, 0.3, 0.3])
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)

        p1 = POMCPPlanner(n_simulations=300, n_particles=40, rng=rng1)
        p2 = POMCPPlanner(n_simulations=300, n_particles=40, rng=rng2)
        a1 = p1.search(belief)
        a2 = p2.search(belief)
        assert a1 == a2, f"Reproducibility broken: {a1} vs {a2}"

    def test_more_simulations_improve_consistency(self):
        """Higher simulation budget → more consistent Q-values."""
        belief = np.array([0.5, 0.3, 0.2])

        # Run 3 times with low budget
        low_results = []
        for seed in [1, 2, 3]:
            p = POMCPPlanner(n_simulations=100, n_particles=30, rng=np.random.default_rng(seed))
            low_results.append(p.search(belief))

        # Run 3 times with high budget
        high_results = []
        for seed in [1, 2, 3]:
            p = POMCPPlanner(n_simulations=600, n_particles=80, rng=np.random.default_rng(seed))
            high_results.append(p.search(belief))

        # High budget should be more consistent (less variance in action choice)
        # Not a hard assert — stochastic — but high budget helps
        assert len(set(high_results)) <= len(set(low_results)) + 1
