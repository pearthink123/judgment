"""
Tests for Layer 2: Hidden Markov Model (hmm.py)

Validates:
  - Emission tables: correct shapes, row-sums valid
  - Forward algorithm: belief normalises, monotonic response to evidence
  - encode_observation: correct category mapping
  - Viterbi: plausible state sequences
  - Transition matrix: row-stochastic
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.hmm import (
    HiddenMarkovModel,
    encode_observation,
    STATE_HEALTHY,
    STATE_DEGRADED,
    STATE_BROKEN,
    STATE_NAMES,
    N_STATES,
    DEFAULT_PRIOR,
    DEFAULT_TRANSITION,
    EMISSION_TABLES,
)


class TestEmissionTables:
    """Verify emission tables are well-formed."""

    def test_table_shapes(self):
        for dim, tbl in EMISSION_TABLES.items():
            assert tbl.shape[0] == N_STATES, f"Dim {dim}: wrong n_states"
            assert tbl.shape[1] >= 2, f"Dim {dim}: too few categories"
            assert np.all(tbl >= 0), f"Dim {dim}: negative probabilities"

    def test_emission_rows_sum_to_one(self):
        for dim, tbl in EMISSION_TABLES.items():
            for s in range(N_STATES):
                row_sum = tbl[s].sum()
                assert abs(row_sum - 1.0) < 0.01, (
                    f"Dim {dim}, state {s}: row sums to {row_sum}"
                )

    def test_healthy_better_than_broken(self):
        """P(tool_ok | H) > P(tool_ok | B)."""
        tbl = EMISSION_TABLES[0]  # tool_ok dimension
        assert tbl[STATE_HEALTHY, 1] > tbl[STATE_BROKEN, 1]

    def test_degraded_between_healthy_and_broken(self):
        """H > D > B for good outcomes, reversed for bad."""
        tbl_ok = EMISSION_TABLES[0]  # tool_ok
        assert (
            tbl_ok[STATE_HEALTHY, 1]
            > tbl_ok[STATE_DEGRADED, 1]
            > tbl_ok[STATE_BROKEN, 1]
        )


class TestTransitionMatrix:
    def test_row_stochastic(self):
        for r in range(N_STATES):
            assert abs(DEFAULT_TRANSITION[r].sum() - 1.0) < 0.01

    def test_diagonal_dominant(self):
        """Markov inertia: staying is more likely than moving."""
        for s in range(N_STATES):
            diag = DEFAULT_TRANSITION[s, s]
            for s2 in range(N_STATES):
                if s2 != s:
                    assert diag > DEFAULT_TRANSITION[s, s2], (
                        f"State {s}: diag {diag} not dominant over {DEFAULT_TRANSITION[s, s2]}"
                    )


class TestEncodeObservation:
    def test_healthy_step(self):
        cats = encode_observation(
            tool_ok=True, progress_delta=0.15, has_user_msg=False, error_count_delta=0
        )
        assert cats[0] == 1  # tool_ok
        assert cats[1] == 2  # progress pos
        assert cats[2] == 0  # user silent
        assert cats[3] == 0  # error stable

    def test_error_step(self):
        cats = encode_observation(
            tool_ok=False, progress_delta=-0.05, has_user_msg=False, error_count_delta=2
        )
        assert cats[0] == 0  # tool fail
        assert cats[1] == 0  # progress neg
        assert cats[2] == 0  # user silent
        assert cats[3] == 1  # error rising

    def test_zero_progress(self):
        cats = encode_observation(
            tool_ok=True, progress_delta=0.01, has_user_msg=False, error_count_delta=0
        )
        assert cats[1] == 1  # progress zero


class TestForwardAlgorithm:
    def test_belief_normalises(self):
        hmm = HiddenMarkovModel()
        obs = encode_observation(True, 0.10, False, 0)
        b = hmm.forward_step(obs)
        assert abs(b.sum() - 1.0) < 1e-6
        assert np.all(b >= 0) and np.all(b <= 1)

    def test_multiple_steps_normalise(self):
        hmm = HiddenMarkovModel()
        for _ in range(10):
            obs = encode_observation(True, 0.10, False, 0)
            b = hmm.forward_step(obs)
            assert abs(b.sum() - 1.0) < 1e-6

    def test_healthy_sequence_stays_healthy(self):
        """Consecutive healthy observations → belief converges to H."""
        hmm = HiddenMarkovModel()
        for _ in range(10):
            obs = encode_observation(
                tool_ok=True, progress_delta=0.20, has_user_msg=False, error_count_delta=0
            )
            b = hmm.forward_step(obs)
        assert b[STATE_HEALTHY] > b[STATE_DEGRADED]
        assert b[STATE_HEALTHY] > 0.90

    def test_error_sequence_trends_degraded(self):
        """Repeated errors → belief shifts toward Degraded/Broken."""
        hmm = HiddenMarkovModel()
        for step_i in range(15):
            obs = encode_observation(
                tool_ok=False,
                progress_delta=-0.02,
                has_user_msg=False,
                error_count_delta=1,
            )
            b = hmm.forward_step(obs)
        # After 15 error steps, Healthy should be low
        assert b[STATE_HEALTHY] < 0.15
        # Broken should dominate
        assert b[STATE_BROKEN] > 0.3

    def test_recovery_from_degraded(self):
        """Degraded → Healthy transitions are possible."""
        hmm = HiddenMarkovModel()
        # 5 mild error steps: tool fail but zero progress (not negative),
        # stable error count (no cascading errors).
        # This favours Degraded over Broken.
        for _ in range(5):
            obs = encode_observation(
                False, progress_delta=0.0, has_user_msg=False, error_count_delta=0
            )
            hmm.forward_step(obs)
        mid_belief = hmm.belief()
        # Degraded should have significant mass
        assert mid_belief[STATE_DEGRADED] > 0.10, (
            f"Should be at least somewhat degraded, got D={mid_belief[STATE_DEGRADED]:.3f}"
        )

        # 10 recovery steps
        for _ in range(10):
            obs = encode_observation(True, 0.15, False, 0)
            hmm.forward_step(obs)
        final_belief = hmm.belief()
        assert final_belief[STATE_HEALTHY] > final_belief[STATE_DEGRADED], (
            "Should recover to Healthy after good steps"
        )

    def test_user_interaction_dampens_broken(self):
        """User message → P(B) should not increase monotonically."""
        hmm = HiddenMarkovModel()
        # Step into degraded first
        for _ in range(5):
            hmm.forward_step(encode_observation(False, -0.02, False, 1))
        pre_user = hmm.belief()

        # Now add a user message with positive progress
        hmm.forward_step(encode_observation(True, 0.10, True, 0))
        post_user = hmm.belief()

        # Broken should not spike up from a user message with success
        assert post_user[STATE_BROKEN] <= pre_user[STATE_BROKEN] + 0.05


class TestViterbi:
    def test_single_observation(self):
        hmm = HiddenMarkovModel()
        obs = [encode_observation(True, 0.20, False, 0)]
        path = hmm.viterbi(obs)
        assert len(path) == 1
        assert path[0] == STATE_HEALTHY

    def test_healthy_sequence(self):
        hmm = HiddenMarkovModel()
        obs = [encode_observation(True, 0.15, False, 0) for _ in range(10)]
        path = hmm.viterbi(obs)
        assert all(s == STATE_HEALTHY for s in path)

    def test_error_sequence(self):
        hmm = HiddenMarkovModel()
        obs = [encode_observation(False, -0.05, False, 1) for _ in range(15)]
        path = hmm.viterbi(obs)
        # Should eventually end in Broken
        assert path[-1] == STATE_BROKEN

    def test_mixed_sequence(self):
        """Recovery pattern: good → mild errors → recovery."""
        hmm = HiddenMarkovModel()
        obs = []
        # 5 good steps
        obs += [encode_observation(True, 0.15, False, 0) for _ in range(5)]
        # 5 mild error steps (tool fail, zero progress, stable errors)
        # → should enter Degraded before progressing to Broken
        obs += [encode_observation(False, 0.0, False, 0) for _ in range(5)]
        # 5 recovery steps
        obs += [encode_observation(True, 0.10, False, 0) for _ in range(5)]

        path = hmm.viterbi(obs)
        # Start healthy
        assert path[0] == STATE_HEALTHY
        # End healthy after recovery
        assert path[-1] == STATE_HEALTHY
        # Some non-Healthy in the middle section
        middle_states = set(path[5:10])
        assert middle_states != {STATE_HEALTHY}, (
            "Should have non-Healthy states in the error phase"
        )


class TestExpectedLogLik:
    def test_healthy_obs_has_reasonable_loglik(self):
        hmm = HiddenMarkovModel()
        obs = encode_observation(True, 0.15, False, 0)
        ell = hmm.expected_log_lik(obs)
        # Should not be extremely low (e.g. < -10 would be pathological)
        assert ell > -10.0

    def test_per_state_lik_ordered(self):
        """log P(o | H) > log P(o | B) for a healthy observation."""
        hmm = HiddenMarkovModel()
        obs = encode_observation(True, 0.15, False, 0)
        log_lik = hmm.log_obs_likelihood(obs)
        assert log_lik[STATE_HEALTHY] > log_lik[STATE_BROKEN]
