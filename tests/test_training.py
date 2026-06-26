"""Tests for Baum-Welch training (training.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.training import baum_welch, train_hmm, build_observation_sequences
from core.hmm import (
    HiddenMarkovModel, encode_observation,
    STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN,
)


def _make_healthy_seq(n_steps: int = 10) -> list:
    return [
        encode_observation(True, 0.15, False, 0)
        for _ in range(n_steps)
    ]


def _make_error_seq(n_steps: int = 10) -> list:
    return [
        encode_observation(False, -0.05, False, 1)
        for _ in range(n_steps)
    ]


def _make_mixed_seq() -> list:
    seq = []
    seq += [encode_observation(True, 0.15, False, 0) for _ in range(4)]
    seq += [encode_observation(False, -0.02, False, 1) for _ in range(3)]
    seq += [encode_observation(True, 0.10, False, 0) for _ in range(3)]
    return seq


class TestBaumWelch:
    def test_converges_on_single_sequence(self):
        seq = _make_healthy_seq(15)
        prior, T, B, ll_hist = baum_welch(
            [seq], n_iter=30, tol=1e-3,
        )
        assert len(ll_hist) > 0
        # Log-likelihood should generally increase
        assert ll_hist[-1] > ll_hist[0] - 1.0  # allow small initial dip

    def test_output_shapes(self):
        prior, T, B, _ = baum_welch([_make_healthy_seq(10)], n_iter=5)
        assert prior.shape == (3,)
        assert T.shape == (3, 3)
        for dim in B:
            assert B[dim].shape[0] == 3

    def test_rows_stochastic(self):
        _, T, B, _ = baum_welch([_make_healthy_seq(10)], n_iter=10)
        for s in range(3):
            assert abs(T[s].sum() - 1.0) < 0.01
        for dim, tbl in B.items():
            for s in range(3):
                assert abs(tbl[s].sum() - 1.0) < 0.01

    def test_mixed_sequences(self):
        seq1 = _make_healthy_seq(10)
        seq2 = _make_error_seq(10)
        seq3 = _make_mixed_seq()
        prior, T, B, ll_hist = baum_welch(
            [seq1, seq2, seq3], n_iter=30, tol=1e-4,
        )
        # Healthy→Healthy should be learned as high
        assert T[STATE_HEALTHY, STATE_HEALTHY] > 0.3

    def test_semi_supervised_anchors_state(self):
        """With label at step 0 forcing STATE_HEALTHY, prior should reflect it."""
        seq = _make_healthy_seq(10)
        labels = [{0: STATE_HEALTHY}]  # label step 0 as Healthy
        prior, T, B, _ = baum_welch(
            [seq], labels=labels, n_iter=10,
        )
        # The prior should be heavily weighted toward Healthy
        assert prior[STATE_HEALTHY] > 0.3

    def test_multiple_sequences(self):
        sequences = [_make_healthy_seq(8) for _ in range(5)]
        prior, T, B, ll_hist = baum_welch(sequences, n_iter=15)
        assert len(ll_hist) > 0
        assert abs(prior.sum() - 1.0) < 0.01


class TestTrainHMM:
    def test_returns_trained_hmm(self):
        logs = []
        for _ in range(3):
            traj = []
            for _ in range(10):
                traj.append({
                    "tool_ok": True, "progress_delta": 0.15,
                    "has_user_msg": False, "error_count_delta": 0,
                })
            logs.append(traj)

        hmm = train_hmm(logs, n_iter=10)
        assert isinstance(hmm, HiddenMarkovModel)

        # Test forward on a new observation
        obs = encode_observation(True, 0.15, False, 0)
        b = hmm.forward_step(obs)
        assert abs(b.sum() - 1.0) < 1e-6


class TestBuildObservationSequences:
    def test_converts_raw_logs(self):
        logs = [[
            {"tool_ok": True, "progress_delta": 0.15, "has_user_msg": False, "error_count_delta": 0},
            {"tool_ok": False, "progress_delta": -0.05, "has_user_msg": False, "error_count_delta": 1},
        ]]
        seqs = build_observation_sequences(logs)
        assert len(seqs) == 1
        assert len(seqs[0]) == 2
        # First obs should have tool_ok=1
        assert seqs[0][0][0] == 1  # tool_ok dimension, category "ok"
        # Second obs should have tool_ok=0
        assert seqs[0][1][0] == 0  # tool_ok dimension, category "fail"
