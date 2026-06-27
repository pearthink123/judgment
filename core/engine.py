"""
DecisionEngine — 3-layer stack for Agent Harness action decisions.

Layer 1: CUSUM anomaly detection (Hawkes-corrected surprisal)
Layer 2: HMM latent-state belief update (Healthy / Degraded / Broken)
         Optional: content-quality signals for richer belief inference
Layer 3: POMDP action selection:
           - POMCP (online particle MCTS) → scales to larger state spaces
           - Grid value iteration → fast, 231-point exact solve
           - Threshold gate → fallback if both solvers unavailable

Usage inside an agent loop:

    engine = DecisionEngine(seed=42)

    for step in range(max_steps):
        obs = {
            "tool_ok": True,
            "progress_delta": 0.15,
            "has_user_msg": False,
            "error_count_delta": 0,
            "llm_text": "The agent's output text...",   # optional
        }
        decision = engine.step(obs)
        # decision.action ∈ {"continue","correct","escalate","gather"}
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
from .pomcp import POMCPPlanner, POMCPSearchInfo
from .pomcp_fast import FastPOMCPPlanner, FastSearchInfo
from .content_signals import ContentSignalExtractor


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------
ACTION_CONTINUE = "continue"
ACTION_CORRECT = "correct"
ACTION_ESCALATE = "escalate"
ACTION_GATHER = "gather"

ACTION_SET = {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER}

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
    # Content signals (present when use_content_signals=True)
    content_signals: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    3-layer math-driven decision engine.

    Layer 3 has four modes (auto-selected):
      1. FastPOMCP — batch-pre-sampled MCTS, ~10x faster (use_fast_pomcp=True)
      2. POMCP — recursive MCTS (use_pomcp=True)
      3. Grid value iteration — 231-point exact solve (use_pomdp=True, default)
      4. Threshold gate — fallback

    Parameters
    ----------
    use_pomcp : bool — use online MCTS instead of grid lookup.
    use_fast_pomcp : bool — use batch-optimised MCTS (~10x faster than POMCP).
    use_content_signals : bool — extract content-quality metrics from llm_text.
    pomcp_n_simulations, pomcp_n_particles : int — POMCP budget.
    """

    def __init__(
        self,
        hmm: Optional[HiddenMarkovModel] = None,
        hawkes: Optional[HawkesProcess] = None,
        cusum: Optional[CUSUMDetector] = None,
        reward: Optional[RewardConfig] = None,
        use_pomdp: bool = False,
        use_pomcp: bool = False,
        use_fast_pomcp: bool = True,
        use_corrective: bool = True,
        use_content_signals: bool = False,
        pomdp_resolution: float = 0.05,
        pomcp_n_simulations: int = 1000,
        pomcp_n_particles: int = 200,
        seed: Optional[int] = None,
    ):
        self.hmm = hmm or HiddenMarkovModel()
        self.hawkes = hawkes or HawkesProcess()
        self.cusum = cusum or CUSUMDetector()
        self.rng = np.random.default_rng(seed)

        self.use_pomdp = use_pomdp
        self.use_pomcp = use_pomcp
        self.use_fast_pomcp = use_fast_pomcp
        self.use_corrective = use_corrective
        self.use_content_signals = use_content_signals

        # ---- Content signal extractor ----
        self.content_extractor: Optional[ContentSignalExtractor] = None
        if use_content_signals:
            self.content_extractor = ContentSignalExtractor()

        # ---- POMDP solver (grid) ----
        self._policy: Optional[POMDPPolicy] = None
        self._pomdp_resolution = pomdp_resolution
        if use_pomdp and not use_pomcp and not use_fast_pomcp:
            try:
                self._policy = get_policy(reward=reward, resolution=pomdp_resolution)
            except Exception:
                self._policy = None

        # ---- POMCP solver (online MCTS) ----
        self._pomcp: Any = None  # POMCPPlanner or FastPOMCPPlanner
        self._pomcp_simulations = pomcp_n_simulations
        self._pomcp_particles = pomcp_n_particles
        if use_fast_pomcp:
            self._pomcp = FastPOMCPPlanner(
                reward_config=reward,
                n_simulations=pomcp_n_simulations,
                n_particles=pomcp_n_particles,
                rng=self.rng,
            )
        elif use_pomcp:
            self._pomcp = POMCPPlanner(
                reward_config=reward,
                n_simulations=pomcp_n_simulations,
                n_particles=pomcp_n_particles,
                rng=self.rng,
            )

        # ---- Corrective router ----
        self.corrective = CorrectiveRouter() if use_corrective else None

        # ---- Entropy threshold for solver fast-path ----
        self._entropy_threshold = 0.40  # skip solver when belief is sharp (~80%+ in one state)

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
        """
        Process one observation and return a Decision.

        Parameters
        ----------
        observation : dict with keys:
            tool_ok           : bool
            progress_delta    : float
            has_user_msg      : bool
            error_count_delta : int
            llm_text          : str or None  — LLM output text (for content signals)

        Returns
        -------
        Decision
        """
        self.step_count += 1
        t = float(self.step_count)

        # --- Extract structural observation ---
        tool_ok = bool(observation.get("tool_ok", True))
        progress_delta = float(observation.get("progress_delta", 0.0))
        has_user_msg = bool(observation.get("has_user_msg", False))
        error_count_delta = int(observation.get("error_count_delta", 0))

        # --- Extract content signals (optional) ---
        content_cats: Optional[Dict[int, int]] = None
        content_info: Optional[Dict[str, Any]] = None
        if self.content_extractor is not None:
            llm_text = observation.get("llm_text", None)
            content_cats = self.content_extractor.extract(llm_text)
            content_info = {
                "length_z_cat": content_cats.get(self.content_extractor.DIM_LENGTH, 1),
                "novelty_cat": content_cats.get(self.content_extractor.DIM_NOVELTY, 1),
                "negation_cat": content_cats.get(self.content_extractor.DIM_NEGATION, 0),
                "recent_mean_length": round(self.content_extractor.recent_mean_length, 1),
            }

        # ==================================================================
        # LAYER 1: Hawkes + CUSUM
        # ==================================================================
        self.hawkes.add_observation(t, tool_ok, has_user_msg, progress_delta, error_count_delta)

        obs_cats = encode_observation(
            tool_ok, progress_delta, has_user_msg, error_count_delta,
            content_signals=content_cats,
        )

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
        # LAYER 2: HMM forward (with content signals if available)
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
        # LAYER 3: Action selection
        #   If belief is sharply peaked (low entropy) and no anomaly:
        #     → fast path: threshold gate (<100us)
        #   Otherwise:
        #     → POMDP solver (~25ms for FastPOMCP, <1ms for grid)
        # ==================================================================
        pomcp_info: Optional[POMCPSearchInfo] = None
        q_vals: Dict[str, float] = {}

        # --- Compute belief entropy: -Σ b·log(b) ---
        b_clipped = np.clip(belief_perturbed, 1e-8, 1.0)
        belief_entropy = -float(np.sum(b_clipped * np.log(b_clipped)))

        # Max entropy for 3 states = log(3) ≈ 1.099
        # Skip solver when belief is confident (entropy < 0.40, ~80%+ mass on one state)
        # and there's no anomaly signal.
        skip_solver = (
            belief_entropy < self._entropy_threshold
            and not anomaly
            and self.prev_action is not None  # only skip after first step
        )

        if skip_solver:
            # Fast path — threshold gate from confident belief
            p_h = float(belief_perturbed[STATE_HEALTHY])
            p_d = float(belief_perturbed[STATE_DEGRADED])
            p_b = float(belief_perturbed[STATE_BROKEN])
            raw_action = self._gate(p_h, p_d, p_b)
            action = self._hysteresis(raw_action, belief_perturbed)
            q_vals = {}
        elif self._pomcp is not None:
            # --- POMCP: online particle MCTS ---
            pomdp_action_idx = self._pomcp.search(belief_perturbed)
            raw_action = _POMDP_TO_ACTION[pomdp_action_idx]
            action = self._hysteresis(raw_action, belief_perturbed)
            if self._pomcp.last_info:
                pomcp_info = self._pomcp.last_info
                q_vals = self._pomcp.last_info.q_values

        elif self._policy is not None:
            # --- Grid value iteration ---
            pomdp_action_idx = self._policy.best_action(belief_perturbed)
            raw_action = _POMDP_TO_ACTION[pomdp_action_idx]
            action = self._hysteresis(raw_action, belief_perturbed)
            q_vals = self._policy.q_values(belief_perturbed)

        else:
            # --- Threshold gate ---
            p_h = float(belief_perturbed[STATE_HEALTHY])
            p_d = float(belief_perturbed[STATE_DEGRADED])
            p_b = float(belief_perturbed[STATE_BROKEN])
            raw_action = self._gate(p_h, p_d, p_b)
            action = self._hysteresis(raw_action, belief_perturbed)

        # Confidence
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
        rationale = self._build_rationale(
            action, belief_perturbed, anomaly, cusum_result, q_vals, pomcp_info,
        )

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
            "solver": (
                "threshold" if skip_solver
                else "fast_pomcp" if (self._pomcp is not None and self.use_fast_pomcp)
                else "pomcp" if self._pomcp is not None
                else "grid" if self._policy is not None
                else "threshold"
            ),
        }

        if pomcp_info:
            diag["pomcp_simulations"] = pomcp_info.simulations
            diag["pomcp_tree_size"] = pomcp_info.tree_size
            diag["pomcp_max_depth"] = pomcp_info.max_depth_reached

        decision = Decision(
            action=action,
            belief=belief_dict,
            confidence=round(confidence, 4),
            anomaly=anomaly,
            drift=round(cusum_result["S"], 4),
            layer_diagnostics=diag,
            rationale=rationale,
            corrective_advice=corrective_advice if corrective_advice else None,
            content_signals=content_info,
        )

        self.prev_action = action
        self.prev_belief = belief_perturbed
        self.decision_log.append(decision)
        return decision

    # ------------------------------------------------------------------
    # Fallback: threshold gate
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
        pomcp_info: Optional[POMCPSearchInfo] = None,
    ) -> str:
        parts: list[str] = []
        p_h = float(belief[STATE_HEALTHY])
        p_d = float(belief[STATE_DEGRADED])
        p_b = float(belief[STATE_BROKEN])

        if pomcp_info is not None:
            parts.append(
                f"POMCP({pomcp_info.simulations} sims): {action} "
                f"(Q*={pomcp_info.best_q:.2f}, margin="
                f"{pomcp_info.best_q - pomcp_info.runner_up_q:.2f}, "
                f"tree={pomcp_info.tree_size} nodes)"
            )
        elif q_vals:
            best_q = max(q_vals.values())
            runner_up = sorted(q_vals.values(), reverse=True)[1] if len(q_vals) > 1 else best_q
            parts.append(
                f"Grid POMDP: {action} (Q*={best_q:.2f}, margin={best_q - runner_up:.2f})"
            )
        else:
            if action == ACTION_ESCALATE:
                parts.append(f"P(B)={p_b:.2f} >= theta_B={self.theta_broken}")
            elif action == ACTION_CORRECT:
                parts.append(f"P(D)={p_d:.2f} >= theta_D={self.theta_degraded}")
            elif action == ACTION_CONTINUE:
                parts.append(f"P(H)={p_h:.2f} >= theta_H={self.theta_healthy}")
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
        if self.content_extractor:
            self.content_extractor.reset()
        if self._pomcp:
            self._pomcp.reset()
        self.step_count = 0
        self.prev_action = None
        self.prev_belief = None
        self.decision_log = []
