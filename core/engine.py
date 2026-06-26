"""
DecisionEngine — 3-layer stack for Agent Harness action decisions.

Layer 1: CUSUM anomaly detection (Hawkes-corrected surprisal)
Layer 2: HMM latent-state belief update (Healthy / Degraded / Broken)
Layer 3: Threshold-gate decision (Continue / Correct / Escalate / Gather)

Usage inside an agent loop:

    engine = DecisionEngine(seed=42)

    for step in range(max_steps):
        # Collect raw observation from harness
        obs = {
            "tool_ok": True,
            "progress_delta": 0.15,
            "has_user_msg": False,
            "error_count_delta": 0,
        }

        decision = engine.step(obs)
        # decision.action ∈ {"continue", "correct", "escalate", "gather"}
        # decision.belief → {"healthy": 0.82, "degraded": 0.15, "broken": 0.03}

        if decision.action == "escalate":
            break

        # Execute in harness, get next observation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .hawkes import (
    HawkesProcess, HawkesDiagnostics,
    EVENT_SUCCESS, EVENT_ERROR, EVENT_USER, EVENT_TOOL,
)
from .hmm import (
    HiddenMarkovModel, encode_observation,
    STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN, STATE_NAMES,
)
from .cusum import CUSUMDetector


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------
# Canonical action set
ACTION_CONTINUE = "continue"     # next planned tool-call
ACTION_CORRECT = "correct"       # verify / rethink / repair
ACTION_ESCALATE = "escalate"     # ask user / rollback / restart
ACTION_GATHER = "gather"         # low-cost info collection

ACTION_SET = {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}


@dataclass
class Decision:
    """Structured output of one engine.step() call."""

    action: str                    # one of ACTION_SET
    belief: Dict[str, float]       # {"healthy": p, "degraded": p, "broken": p}
    confidence: float              # belief mass on the dominant state
    anomaly: bool                  # did CUSUM fire this step?
    drift: float                   # current CUSUM statistic S_t
    layer_diagnostics: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    3-layer math-driven decision engine.

    Parameters
    ----------
    hmm : HiddenMarkovModel or None
    hawkes : HawkesProcess or None
    cusum : CUSUMDetector or None
    theta_broken : float
        P(Broken) threshold for escalation (default 0.45).
    theta_degraded : float
        P(Degraded) threshold for corrective action (default 0.35).
    theta_healthy : float
        P(Healthy) threshold for confident continue (default 0.60).
    hysteresis_margin : float
        Extra margin required to change decision when oscillating (default 0.08).
    seed : int or None
    """

    def __init__(
        self,
        hmm: Optional[HiddenMarkovModel] = None,
        hawkes: Optional[HawkesProcess] = None,
        cusum: Optional[CUSUMDetector] = None,
        theta_broken: float = 0.45,
        theta_degraded: float = 0.35,
        theta_healthy: float = 0.60,
        hysteresis_margin: float = 0.08,
        seed: Optional[int] = None,
    ):
        self.hmm = hmm or HiddenMarkovModel()
        self.hawkes = hawkes or HawkesProcess()
        self.cusum = cusum or CUSUMDetector()
        self.rng = np.random.default_rng(seed)

        self.theta_broken = float(theta_broken)
        self.theta_degraded = float(theta_degraded)
        self.theta_healthy = float(theta_healthy)
        self.hysteresis_margin = float(hysteresis_margin)

        # Running state
        self.step_count: int = 0
        self.prev_action: Optional[str] = None
        self.prev_belief: Optional[np.ndarray] = None
        self.decision_log: List[Decision] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def step(self, observation: Dict[str, Any]) -> Decision:
        """
        Process one observation and return a Decision.

        Parameters
        ----------
        observation : dict with keys:
            tool_ok          : bool — did the tool call succeed?
            progress_delta   : float — Δ in task progress (0–1 scale)
            has_user_msg     : bool — did the user send a message this step?
            error_count_delta: int — new errors this step

        Returns
        -------
        Decision
        """
        self.step_count += 1
        t = float(self.step_count)

        # --- Extract observation fields ---
        tool_ok = bool(observation.get("tool_ok", True))
        progress_delta = float(observation.get("progress_delta", 0.0))
        has_user_msg = bool(observation.get("has_user_msg", False))
        error_count_delta = int(observation.get("error_count_delta", 0))

        # ==================================================================
        # LAYER 1: Feed Hawkes + CUSUM
        # ==================================================================
        self.hawkes.add_observation(t, tool_ok, has_user_msg, progress_delta, error_count_delta)

        # Encode for HMM
        obs_cats = encode_observation(tool_ok, progress_delta, has_user_msg, error_count_delta)

        # Per-state log-likelihoods (before HMM update)
        log_lik_per_state = self.hmm.log_obs_likelihood(obs_cats)  # shape (3,)

        # Surprisals: -log P(o | state)
        healthy_surprisal = -float(log_lik_per_state[STATE_HEALTHY])
        degraded_surprisal = -float(log_lik_per_state[STATE_DEGRADED])

        # Hawkes intensity for the primary event type of this observation
        dominant_type = EVENT_ERROR if (not tool_ok or error_count_delta > 0) else EVENT_TOOL
        hawkes_lam = self.hawkes.intensity_for(dominant_type, t)

        # CUSUM update
        cusum_result = self.cusum.update(
            surprisal_healthy=healthy_surprisal,
            hawkes_intensity=hawkes_lam,
            surprisal_degraded=degraded_surprisal,
        )
        anomaly = bool(cusum_result["alarm"])

        # ==================================================================
        # LAYER 2: HMM forward update
        # ==================================================================
        belief = self.hmm.forward_step(obs_cats)

        # If CUSUM alarmed, nudge belief toward Degraded
        # (this is a soft prior, not a hard override)
        if anomaly:
            belief_perturbed = belief.copy()
            belief_perturbed[STATE_HEALTHY] *= 0.7
            belief_perturbed[STATE_DEGRADED] = min(
                belief_perturbed[STATE_DEGRADED] + 0.15, 1.0
            )
            belief_perturbed = belief_perturbed / belief_perturbed.sum()
        else:
            belief_perturbed = belief

        # ==================================================================
        # LAYER 3: Threshold gate
        # ==================================================================
        p_healthy = float(belief_perturbed[STATE_HEALTHY])
        p_degraded = float(belief_perturbed[STATE_DEGRADED])
        p_broken = float(belief_perturbed[STATE_BROKEN])

        # Candidate action before hysteresis
        raw_action = self._gate(p_healthy, p_degraded, p_broken)

        # Hysteresis: require extra margin to flip from previous action
        action = self._apply_hysteresis(raw_action, p_healthy, p_degraded, p_broken)

        # Confidence: probability mass on the state driving the decision
        if action == ACTION_ESCALATE:
            confidence = p_broken
        elif action == ACTION_CORRECT:
            confidence = p_degraded
        elif action == ACTION_CONTINUE:
            confidence = p_healthy
        else:
            confidence = 1.0 - max(p_healthy, p_degraded, p_broken)

        # Rationale
        rationale = self._build_rationale(action, p_healthy, p_degraded, p_broken, anomaly, cusum_result)

        # Assemble
        belief_dict = {
            "healthy": round(p_healthy, 4),
            "degraded": round(p_degraded, 4),
            "broken": round(p_broken, 4),
        }

        diag = {
            "anomaly": anomaly,
            "drift": cusum_result["S"],
            "drift_contrib": cusum_result["L"],
            "hawkes_correction": cusum_result["hawkes_correction"],
            "hawkes_intensities": self.hawkes.intensity(t).tolist(),
            "cusum_alarm_count": self.cusum.alarm_count,
            "most_likely_state": STATE_NAMES[self.hmm.most_likely_state()],
        }

        decision = Decision(
            action=action,
            belief=belief_dict,
            confidence=round(confidence, 4),
            anomaly=anomaly,
            drift=round(cusum_result["S"], 4),
            layer_diagnostics=diag,
            rationale=rationale,
        )

        # Persist
        self.prev_action = action
        self.prev_belief = belief_perturbed
        self.decision_log.append(decision)

        return decision

    # ------------------------------------------------------------------
    # Gating logic
    # ------------------------------------------------------------------
    def _gate(
        self, p_healthy: float, p_degraded: float, p_broken: float
    ) -> str:
        """
        Threshold gate mapping belief → action.

        Priority order: escalate > correct > continue > gather.
        """
        if p_broken >= self.theta_broken:
            return ACTION_ESCALATE
        if p_degraded >= self.theta_degraded:
            return ACTION_CORRECT
        if p_healthy >= self.theta_healthy:
            return ACTION_CONTINUE
        return ACTION_GATHER

    def _apply_hysteresis(
        self,
        raw_action: str,
        p_healthy: float,
        p_degraded: float,
        p_broken: float,
    ) -> str:
        """Prevent boundary oscillation by requiring extra margin to change."""
        if self.prev_action is None or raw_action == self.prev_action:
            return raw_action

        # Action wants to change — require extra margin
        if raw_action == ACTION_ESCALATE and self.prev_action != ACTION_ESCALATE:
            if p_broken < self.theta_broken + self.hysteresis_margin:
                return self.prev_action

        if raw_action == ACTION_CORRECT and self.prev_action == ACTION_CONTINUE:
            if p_degraded < self.theta_degraded + self.hysteresis_margin:
                return self.prev_action

        if raw_action == ACTION_CONTINUE and self.prev_action == ACTION_CORRECT:
            if p_healthy < self.theta_healthy + self.hysteresis_margin:
                return self.prev_action

        return raw_action

    # ------------------------------------------------------------------
    # Rationale
    # ------------------------------------------------------------------
    def _build_rationale(
        self,
        action: str,
        p_h: float,
        p_d: float,
        p_b: float,
        anomaly: bool,
        cusum: Dict[str, float],
    ) -> str:
        parts: list[str] = []

        if action == ACTION_ESCALATE:
            parts.append(f"P(Broken)={p_b:.2f} ≥ θ_B={self.theta_broken}")
        elif action == ACTION_CORRECT:
            parts.append(f"P(Degraded)={p_d:.2f} ≥ θ_D={self.theta_degraded}")
        elif action == ACTION_CONTINUE:
            parts.append(f"P(Healthy)={p_h:.2f} ≥ θ_H={self.theta_healthy}")
        else:
            parts.append("belief ambiguous — gathering information")

        if anomaly:
            parts.append("CUSUM alarm fired")

        if cusum.get("S", 0) > self.cusum.h * 0.5:
            parts.append(f"drift elevated (S={cusum['S']:.2f})")

        return " | ".join(parts) if parts else "default"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_diagnostics_dataframe(self):
        """Diagnostics DataFrame for dashboards."""
        import pandas as pd
        if not self.decision_log:
            return pd.DataFrame()
        rows = []
        for i, d in enumerate(self.decision_log):
            rows.append({
                "step": i + 1,
                "action": d.action,
                "confidence": d.confidence,
                "P(H)": d.belief["healthy"],
                "P(D)": d.belief["degraded"],
                "P(B)": d.belief["broken"],
                "drift": d.drift,
                "anomaly": d.anomaly,
            })
        return pd.DataFrame(rows)

    def reset(self):
        self.hmm.reset()
        self.hawkes.reset()
        self.cusum.reset()
        self.step_count = 0
        self.prev_action = None
        self.prev_belief = None
        self.decision_log = []
