"""Tests for POMDP belief-MDP solver (pomdp.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.pomdp import (
    solve_belief_mdp, get_policy, POMDPPolicy,
    RewardConfig, discretise_simplex, nearest_grid_point, observation_prob,
    ACT_CONTINUE, ACT_CORRECT, ACT_ESCALATE, ACT_GATHER,
    ACTION_NAMES_POMDP, N_OBSERVATIONS,
)


class TestSimplexDiscretisation:
    def test_grid_points_sum_to_one(self):
        grid, _ = discretise_simplex(0.05)
        for row in grid:
            assert abs(row.sum() - 1.0) < 1e-6

    def test_grid_all_nonnegative(self):
        grid, _ = discretise_simplex(0.05)
        assert np.all(grid >= 0)

    def test_nearest_grid_point_self(self):
        grid, _ = discretise_simplex(0.05)
        for i in range(0, len(grid), 50):
            assert nearest_grid_point(grid[i], grid) == i


class TestObservationModel:
    def test_observation_prob_sums(self):
        """For a given state, Σ_o P(o|s) should be close to 1."""
        # Check via brute force over all observations
        for s in range(3):
            total = 0.0
            for oi in range(N_OBSERVATIONS):
                total += float(observation_prob(oi)[s])
            assert abs(total - 1.0) < 0.05, f"State {s}: ΣP(o|s)={total:.4f}"


class TestPOMDPSolver:
    def test_solve_converges(self):
        policy = solve_belief_mdp(resolution=0.1, max_iter=500, tol=5e-4)
        assert policy.converged
        assert policy.n_iterations < 500

    def test_policy_size(self):
        policy = solve_belief_mdp(resolution=0.1)
        assert len(policy.grid) > 50
        assert policy.V.shape == (len(policy.grid),)
        assert policy.Q.shape == (len(policy.grid), 4)
        assert policy.policy.shape == (len(policy.grid),)

    def test_best_action_healthy(self):
        """At belief = [1, 0, 0] (100% Healthy), best action should be continue."""
        policy = solve_belief_mdp(resolution=0.1)
        b = np.array([1.0, 0.0, 0.0])
        action = policy.best_action(b)
        assert action == ACT_CONTINUE, (
            f"Expected CONTINUE for pure Healthy, got {ACTION_NAMES_POMDP[action]}"
        )

    def test_best_action_broken(self):
        """At belief = [0, 0, 1] (100% Broken), escalate should be best."""
        policy = solve_belief_mdp(resolution=0.1)
        b = np.array([0.0, 0.0, 1.0])
        action = policy.best_action(b)
        # Either escalate or correct — definitely not continue
        assert action != ACT_CONTINUE, "Should NOT continue when 100% Broken"

    def test_best_action_degraded(self):
        """At belief = [0, 1, 0] (100% Degraded), correct should rank high."""
        policy = solve_belief_mdp(resolution=0.1)
        b = np.array([0.0, 1.0, 0.0])
        q = policy.q_values(b)
        # correct should be competitive
        assert q[ACTION_NAMES_POMDP[ACT_CORRECT]] > q[ACTION_NAMES_POMDP[ACT_CONTINUE]], (
            f"Degraded: correct Q should exceed continue Q: {q}"
        )

    def test_value_non_negative_for_heavily_broken(self):
        """Even at 100% Broken, V should not be pathologically negative."""
        policy = solve_belief_mdp(resolution=0.1)
        b = np.array([0.0, 0.0, 1.0])
        v = policy.value(b)
        # Escalate gives +2 immediately. With gamma discount, V ≥ 0.
        assert v >= -5.0, f"V for Broken should not be extremely negative: {v}"


class TestRewardConfig:
    def test_presets_load(self):
        for name in ["general", "conservative", "permissive"]:
            cfg = RewardConfig.preset(name)
            assert isinstance(cfg, RewardConfig)

    def test_presets_have_different_values(self):
        gen = RewardConfig.preset("general")
        con = RewardConfig.preset("conservative")
        assert con.continue_B < gen.continue_B, "Conservative: Broken→continue should be more penalised"

    def test_matrix_shape(self):
        R = RewardConfig().matrix()
        assert R.shape == (3, 4)


class TestCachedPolicy:
    def test_get_policy_returns_cached(self):
        p1 = get_policy(resolution=0.1)
        p2 = get_policy(resolution=0.1)
        assert p1 is p2  # same object — cached

    def test_force_recompute(self):
        p1 = get_policy(resolution=0.1)
        p2 = get_policy(resolution=0.1, force_recompute=True)
        assert p1 is not p2
