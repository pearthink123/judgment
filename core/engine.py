"""
DecisionEngine — 3-layer stack for Agent Harness action decisions.

Layer 1: CUSUM anomaly detection (Hawkes-corrected surprisal)
Layer 2: HMM latent-state belief update (Healthy / Degraded / Broken)
Layer 3: POMDP policy lookup (optimal action under uncertainty)
         Falls back to threshold-gate if POMDP solver unavailable.

Usage inside an agent loop:

    engine = DecisionEngine(seed=42)

    for step in range(max_steps):
        obs = {
            "tool_ok": True,
            "progress_delta": 0.15,
            "has_user_msg": False,
            "error_count_delta": 0,
        }

        decision = engine.step(obs)
        # decision.action ∈ {"continue","correct","escalate","gather"}
        # decision.corrective_advice → CorrectiveAdvice or None

        if decision.action == "escalate":
            break
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .hawkes import (
    HawkesProcess,
    EVENT_ERROR, EVENT_TOOL,
)
from .hmm import (
    HiddenMarkovModel, encode_observation,
    STATE_HEALTHY, STATE_DEGRADED, STATE_BROKEN, STATE_NAMES,
)
from .cusum import CUSUMDetector
from .corrective import CorrectiveRouter, CorrectiveAdvice
from .pomdp import (
    POMDPPolicy, RewardConfig, get_policy,
    ACT_CONTINUE, ACT_CORRECT, ACT_ESCALATE, ACT_GATHER,
    ACTION_NAMES_POMDP,
)


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------
ACTION_CONTINUE = "continue"
ACTION_CORRECT = "correct"
ACTION_ESCALATE = "escalate"
ACTION_GATHER = "gather"

ACTION_SET = {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}

# Map POMDP action indices → string
_POMDP_TO_ACTION = {
    ACT_CONTINUE: ACTION_CONTINUE,
    ACT_CORRECT: ACTION_CORRECT,
    ACT_ESCALATE: ACTION_ESCALATE,
    ACT_GATHER: ACTION_GATHER,
}


@dataclass
class Decision:
    """Structured output of one engine.step() call."""

    action: str
    belief: Dict[str, float]
    confidence: float
    anomaly: bool
    drift: float
    layer_diagnostics: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    corrective_advice: Optional[CorrectiveAdvice] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    3-layer math-driven decision engine.

    Layer 3 defaults to POMDP policy lookup; falls back to threshold-gate
    if ``use_pomdp=False`` or the POMDP solver fails.

    Parameters
    ----------
    hmm, hawkes, cusum : optional component overrides.
    reward : RewardConfig or str preset name.
    use_pomdp : bool — default True.
    use_corrective : bool — default True (attach advice on CORRECT).
    pomdp_resolution : float — belief grid step (default 0.05).
    seed : int or None.
    """

    def __init__(
        self,
        hmm: Optional[HiddenMarkovModel] = None,
        hawkes: Optional[HawkesProcess] = None,
        cusum: Optional[CUSUMDetector] = None,
        reward: Optional[RewardConfig] = None,
        use_pomdp: bool = True,
        use_corrective: bool = True,
        pomdp_resolution: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.hmm = hmm or HiddenMarkovModel()
        self.hawkes = hawkes or HawkesProcess()
        self.cusum = cusum or CUSUMDetector()
        self.rng = np.random.default_rng(seed)

        self.use_pomdp = use_pomdp
        self.use_corrective = use_corrective

        # ---- POMDP policy ----
        self._policy: Optional[POMDPPolicy] = None
        self._pomdp_resolution = pomdp_resolution
        if use_pomdp:
            try:
                self._policy = get_policy(reward=reward, resolution=pomdp_resolution)
            except Exception:
                self._policy = None

        # ---- Corrective router ----
        self.corrective = CorrectiveRouter() if use_corrective else None

        # ---- Threshold fallback ----
        self.theta_broken = 0.45
        self.theta_degraded = 0.35
        self.theta_healthy = 0.60
        self.hysteresis_margin = 0.08

        # ---- State ----
        self.step_count: int = 0
        self.prev_action: Optional[str] = None
        self.prev_belief: Optional[np.ndarray] = None
        self.decision_log: List[Decision] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def step(self, observation: Dict[str, Any]) -> Decision:
        self.step_count += 1
        t = float(self.step_count)

        # --- Extract observation ---
        tool_ok = bool(observation.get("tool_ok", True))
        progress_delta = float(observation.get("progress_delta", 0.0))
        has_user_msg = bool(observation.get("has_user_msg", False))
        error_count_delta = int(observation.get("error_count_delta", 0))

        # ==================================================================
        # LAYER 1: Hawkes + CUSUM
        # ==================================================================
        self.hawkes.add_observation(t, tool_ok, has_user_msg, progress_delta, error_count_delta)

        obs_cats = encode_observation(tool_ok, progress_delta, has_user_msg, error_count_delta)

        log_lik_per_state = self.hmm.log_obs_likelihood(obs_cats)
        healthy_surprisal = -float(log_lik_per_state[STATE_HEALTHY])
        degraded_surprisal = -float(log_lik_per_state[STATE_DEGRADED])

        dominant_type = EVENT_ERROR if (not tool_ok or error_count_delta > 0) else EVENT_TOOL
        hawkes_lam = self.hawkes.intensity_for(dominant_type, t)

        cusum_result = self.cusum.update(
            surprisal_healthy=healthy_surprisal,
            hawkes_intensity=hawkes_lam,
            surprisal_degraded=degraded_surprisal,
        )
        anomaly = bool(cusum_result["alarm"])

        # ==================================================================
        # LAYER 2: HMM forward
        # ==================================================================
        belief = self.hmm.forward_step(obs_cats)

        if anomaly:
            belief_perturbed = belief.copy()
            belief_perturbed[STATE_HEALTHY] *= 0.7
            belief_perturbed[STATE_DEGRADED] = min(belief_perturbed[STATE_DEGRADED] + 0.15, 1.0)
            belief_perturbed = belief_perturbed / belief_perturbed.sum()
        else:
            belief_perturbed = belief

        # ==================================================================
        # LAYER 3: POMDP policy (or threshold fallback)
        # ==================================================================
        if self._policy is not None:
            pomdp_action_idx = self._policy.best_action(belief_perturbed)
            raw_action = _POMDP_TO_ACTION[pomdp_action_idx]
            action = self._hysteresis(raw_action, belief_perturbed)
            q_vals = self._policy.q_values(belief_perturbed)
        else:
            p_h = float(belief_perturbed[STATE_HEALTHY])
            p_d = float(belief_perturbed[STATE_DEGRADED])
            p_b = float(belief_perturbed[STATE_BROKEN])
            raw_action = self._gate(p_h, p_d, p_b)
            action = self._hysteresis(raw_action, belief_perturbed)
            q_vals = {}

        # Confidence: belief mass on the dominant state
        if action == ACTION_ESCALATE:
            confidence = float(belief_perturbed[STATE_BROKEN])
        elif action == ACTION_CORRECT:
            confidence = float(belief_perturbed[STATE_DEGRADED])
        elif action == ACTION_CONTINUE:
            confidence = float(belief_perturbed[STATE_HEALTHY])
        else:
            confidence = 1.0 - float(belief_perturbed.max())

        # Corrective advice
        corrective_advice = None
        if action == ACTION_CORRECT and self.corrective is not None:
            corrective_advice = self.corrective.analyse(self.decision_log)

        # Rationale
        rationale = self._build_rationale(action, belief_perturbed, anomaly, cusum_result, q_vals)

        # Assemble
        belief_dict = {
            "healthy": round(float(belief_perturbed[STATE_HEALTHY]), 4),
            "degraded": round(float(belief_perturbed[STATE_DEGRADED]), 4),
            "broken": round(float(belief_perturbed[STATE_BROKEN]), 4),
        }

        diag = {
            "anomaly": anomaly,
            "drift": cusum_result["S"],
            "drift_contrib": cusum_result["L"],
            "hawkes_correction": cusum_result["hawkes_correction"],
            "hawkes_intensities": self.hawkes.intensity(t).tolist(),
            "cusum_alarm_count": self.cusum.alarm_count,
            "most_likely_state": STATE_NAMES[self.hmm.most_likely_state()],
            "pomdp_q_values": q_vals,
        }

        decision = Decision(
            action=action,
            belief=belief_dict,
            confidence=round(confidence, 4),
            anomaly=anomaly,
            drift=round(cusum_result["S"], 4),
            layer_diagnostics=diag,
            rationale=rationale,
            corrective_advice=corrective_advice if corrective_advice else None,
        )

        self.prev_action = action
        self.prev_belief = belief_perturbed
        self.decision_log.append(decision)
        return decision

    # ------------------------------------------------------------------
    # Fallback: threshold gate (used when POMDP unavailable)
    # ------------------------------------------------------------------
    def _gate(self, p_h: float, p_d: float, p_b: float) -> str:
        if p_b >= self.theta_broken:
            return ACTION_ESCALATE
        if p_d >= self.theta_degraded:
            return ACTION_CORRECT
        if p_h >= self.theta_healthy:
            return ACTION_CONTINUE
        return ACTION_GATHER

    def _hysteresis(self, raw_action: str, belief: np.ndarray) -> str:
        """Prevent oscillation by requiring extra margin to change action."""
        if self.prev_action is None or raw_action == self.prev_action:
            return raw_action

        p_h = float(belief[STATE_HEALTHY])
        p_d = float(belief[STATE_DEGRADED])
        p_b = float(belief[STATE_BROKEN])
        m = self.hysteresis_margin

        if raw_action == ACTION_ESCALATE and self.prev_action != ACTION_ESCALATE:
            if p_b < self.theta_broken + m:
                return self.prev_action
        if raw_action == ACTION_CORRECT and self.prev_action == ACTION_CONTINUE:
            if p_d < self.theta_degraded + m:
                return self.prev_action
        if raw_action == ACTION_CONTINUE and self.prev_action == ACTION_CORRECT:
            if p_h < self.theta_healthy + m:
                return self.prev_action
        return raw_action

    # ------------------------------------------------------------------
    # Rationale
    # ------------------------------------------------------------------
    def _build_rationale(
        self,
        action: str,
        belief: np.ndarray,
        anomaly: bool,
        cusum: Dict[str, float],
        q_vals: Dict[str, float],
    ) -> str:
        parts: list[str] = []
        p_h = float(belief[STATE_HEALTHY])
        p_d = float(belief[STATE_DEGRADED])
        p_b = float(belief[STATE_BROKEN])

        if self._policy is not None and q_vals:
            best_q = max(q_vals.values())
            runner_up = sorted(q_vals.values(), reverse=True)[1] if len(q_vals) > 1 else best_q
            margin = best_q - runner_up
            parts.append(
                f"POMDP: {action} (Q*={best_q:.2f}, margin={margin:.2f})"
            )
        else:
            if action == ACTION_ESCALATE:
                parts.append(f"P(B)={p_b:.2f} ≥ θ_B={self.theta_broken}")
            elif action == ACTION_CORRECT:
                parts.append(f"P(D)={p_d:.2f} ≥ θ_D={self.theta_degraded}")
            elif action == ACTION_CONTINUE:
                parts.append(f"P(H)={p_h:.2f} ≥ θ_H={self.theta_healthy}")
            else:
                parts.append("belief ambiguous")

        if anomaly:
            parts.append("CUSUM alarm")
        if cusum.get("S", 0) > self.cusum.h * 0.5:
            parts.append(f"S={cusum['S']:.2f}")

        return " | ".join(parts) if parts else "default"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_diagnostics_dataframe(self):
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
