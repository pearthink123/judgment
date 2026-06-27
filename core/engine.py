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
        # decision.action ∈ {"continue","correct","escalate","gather","replan"}
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
from .config import EngineConfig


# ---------------------------------------------------------------------------
# Decision output — action constants
# ---------------------------------------------------------------------------
ACTION_CONTINUE = "continue"
ACTION_CORRECT  = "correct"
ACTION_ESCALATE = "escalate"
ACTION_GATHER   = "gather"
ACTION_REPLAN   = "replan"     # plan-level: step back, reconsider strategy

ACTION_SET = {ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER, ACTION_REPLAN}

_POMDP_TO_ACTION = {
    ACT_CONTINUE: ACTION_CONTINUE,
    ACT_CORRECT:  ACTION_CORRECT,
    ACT_ESCALATE:  ACTION_ESCALATE,
    ACT_GATHER:   ACTION_GATHER,
}


# ---------------------------------------------------------------------------
# Decision output dataclass
# ---------------------------------------------------------------------------
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

    # --- Anthropic-aligned structured fields ---
    structured_rationale: Dict[str, Any] = field(default_factory=dict)
    # Keys: "thinking", "detected_issue", "recommended_action", "confidence",
    #        "evidence", "monitoring_level"
    handover_report: Optional[Dict[str, Any]] = None
    # Present when action=="escalate" — rich context for human / next agent


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

    Configuration: use ``EngineConfig.preset("conservative")`` or
    ``DecisionEngine.from_config(cfg)``.  Entropy fast-path skips
    the expensive solver when belief is sharply peaked.

    Anthropic alignment mode: ``EngineConfig.preset("anthropic")`` enables
    structured_rationale, cost-aware tracking, plan-adherence signals,
    replan action, and rich handover reports on escalation.
    """

    @classmethod
    def from_config(cls, cfg: EngineConfig, **overrides) -> "DecisionEngine":
        """Create an engine from an EngineConfig object. Accepts keyword overrides."""
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cls(
            reward=cfg.reward,
            use_pomdp=cfg.use_pomdp,
            use_pomcp=cfg.use_pomcp,
            use_fast_pomcp=cfg.use_fast_pomcp,
            use_corrective=cfg.use_corrective,
            use_content_signals=cfg.use_content_signals,
            pomdp_resolution=cfg.pomdp_resolution,
            pomcp_n_simulations=cfg.pomcp_n_simulations,
            pomcp_n_particles=cfg.pomcp_n_particles,
            seed=cfg.seed,
            anthropic_mode=getattr(cfg, "anthropic_mode", False),
            cost_aware=getattr(cfg, "cost_aware", False),
            enable_replan=getattr(cfg, "enable_replan", False),
        )

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
        # --- Anthropic-alignment flags ---
        anthropic_mode: bool = False,
        cost_aware: bool = False,
        enable_replan: bool = False,
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
        self._reward_config = reward  # retained for clone()

        # --- Anthropic-alignment ---
        self.anthropic_mode = anthropic_mode
        self.cost_aware = cost_aware or anthropic_mode
        self.enable_replan = enable_replan or anthropic_mode

        # --- Cost tracking ---
        self.cumulative_tokens_est: int = 0
        self.cumulative_cost_est: float = 0.0
        self._token_cost_per_k: float = 0.003   # ~$3/M input tokens (rough)
        self._token_cost_per_k_output: float = 0.015  # ~$15/M output tokens (rough)

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
        self._entropy_threshold = 0.40  # skip solver when belief is sharp

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
            tool_ok                  : bool
            progress_delta           : float
            has_user_msg             : bool
            error_count_delta        : int
            llm_text                 : str or None  — LLM output text (for content signals)
            plan_adherence           : float or None — how well output follows the plan (-1..1)
            subtask_completion_rate  : float or None — fraction of subtasks completed (0..1)
            token_estimate           : int or None   — estimated tokens consumed this step
            thinking_quality         : float or None — heuristic quality of model reasoning (0..1)

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

        # --- Extended process signals (anthropic_mode) ---
        plan_adherence: Optional[float] = None
        subtask_completion_rate: Optional[float] = None
        if self.anthropic_mode:
            pa = observation.get("plan_adherence")
            if pa is not None:
                plan_adherence = float(pa)
            scr = observation.get("subtask_completion_rate")
            if scr is not None:
                subtask_completion_rate = float(scr)

        # --- Cost tracking ---
        token_estimate = observation.get("token_estimate")
        if token_estimate is not None and self.cost_aware:
            tok = int(token_estimate)
            self.cumulative_tokens_est += tok
            # Rough split: 60% input, 40% output
            self.cumulative_cost_est += (
                tok * 0.60 / 1000 * self._token_cost_per_k
                + tok * 0.40 / 1000 * self._token_cost_per_k_output
            )

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

        # --- Inject plan deviation into CUSUM drift (anthropic_mode) ---
        plan_deviation_penalty = 0.0
        if plan_adherence is not None and plan_adherence < -0.3:
            plan_deviation_penalty = abs(plan_adherence) * 1.5

        cusum_result = self.cusum.update(
            surprisal_healthy=healthy_surprisal + plan_deviation_penalty,
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
        #   Graduated monitoring intensity:
        #     entropy low + healthy → threshold gate (<0.3ms)
        #     entropy medium         → grid POMDP (<1ms)
        #     entropy high + anomaly → FastPOMCP (25ms)
        #     (full POMCP at 200ms is the last-resort debug solver)
        # ==================================================================
        pomcp_info: Optional[POMCPSearchInfo] = None
        q_vals: Dict[str, float] = {}

        # --- Compute belief entropy: -Σ b·log(b) ---
        b_clipped = np.clip(belief_perturbed, 1e-8, 1.0)
        belief_entropy = -float(np.sum(b_clipped * np.log(b_clipped)))

        skip_solver = (
            belief_entropy < self._entropy_threshold
            and not anomaly
            and self.prev_action is not None
        )

        # --- Determine monitoring level (graduated, per Anthropic's compaction philosophy) ---
        if skip_solver:
            monitoring_level = "level_1_threshold"  # cheapest
        elif self._pomcp is not None and self.use_fast_pomcp:
            monitoring_level = "level_3_fast_pomcp"  # standard
        elif self._policy is not None:
            monitoring_level = "level_2_grid"         # mid-cost
        elif self._pomcp is not None:
            monitoring_level = "level_4_full_pomcp"   # most expensive
        else:
            monitoring_level = "level_1_threshold"

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

        # --- REPLAN override (engine-level, not in POMDP action space) ---
        # Triggered when plan_adherence is very low but agent isn't yet broken
        if self.enable_replan and action != ACTION_ESCALATE:
            if plan_adherence is not None and plan_adherence < -0.5:
                p_d = float(belief_perturbed[STATE_DEGRADED])
                p_b = float(belief_perturbed[STATE_BROKEN])
                if p_d >= 0.25 or p_b >= 0.15:
                    action = self._hysteresis(ACTION_REPLAN, belief_perturbed)
                    monitoring_level = "level_3_replan"

        # Confidence
        if action == ACTION_ESCALATE:
            confidence = float(belief_perturbed[STATE_BROKEN])
        elif action == ACTION_REPLAN:
            confidence = float(belief_perturbed[STATE_DEGRADED])
        elif action == ACTION_CORRECT:
            confidence = float(belief_perturbed[STATE_DEGRADED])
        elif action == ACTION_CONTINUE:
            confidence = float(belief_perturbed[STATE_HEALTHY])
        else:
            confidence = 1.0 - float(belief_perturbed.max())

        # Corrective advice
        corrective_advice = None
        if action in (ACTION_CORRECT, ACTION_REPLAN) and self.corrective is not None:
            corrective_advice = self.corrective.analyse(
                self.decision_log,
                anthropic_tone=self.anthropic_mode,
            )

        # Rationale
        rationale = self._build_rationale(
            action, belief_perturbed, anomaly, cusum_result, q_vals, pomcp_info,
            plan_adherence=plan_adherence,
        )

        # Structured rationale (Anthropic XML-style)
        structured = self._build_structured_rationale(
            action, belief_perturbed, anomaly, cusum_result,
            plan_adherence, monitoring_level,
        )

        # Handover report (when escalating)
        handover = None
        if action == ACTION_ESCALATE and self.anthropic_mode:
            handover = self._build_handover_report(
                belief_perturbed, cusum_result, plan_adherence,
            )

        # Assemble belief dict
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
            "monitoring_level": monitoring_level,
            "belief_entropy": round(belief_entropy, 4),
        }

        if pomcp_info:
            diag["pomcp_simulations"] = pomcp_info.simulations
            diag["pomcp_tree_size"] = pomcp_info.tree_size
            diag["pomcp_max_depth"] = pomcp_info.max_depth_reached

        # Cost diagnostics
        if self.cost_aware:
            diag["cumulative_tokens_est"] = self.cumulative_tokens_est
            diag["cumulative_cost_est"] = round(self.cumulative_cost_est, 6)
            diag["est_cost_remaining"] = round(
                self.cumulative_cost_est * max(0, (50 - self.step_count) / max(self.step_count, 1)), 4
            )

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
            structured_rationale=structured,
            handover_report=handover,
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
        # REPLAN hysteresis: don't flip to replan unless we're confident
        if raw_action == ACTION_REPLAN and self.prev_action not in (ACTION_REPLAN, ACTION_CORRECT, ACTION_GATHER):
            if p_d < self.theta_degraded + m:
                return self.prev_action
        return raw_action

    # ------------------------------------------------------------------
    # Rationale — compact text version
    # ------------------------------------------------------------------
    def _build_rationale(
        self,
        action: str,
        belief: np.ndarray,
        anomaly: bool,
        cusum: Dict[str, float],
        q_vals: Dict[str, float],
        pomcp_info: Optional[POMCPSearchInfo] = None,
        plan_adherence: Optional[float] = None,
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
            elif action == ACTION_REPLAN:
                parts.append(
                    f"Plan deviation detected "
                    f"(adherence={plan_adherence:.2f}, P(D)={p_d:.2f})"
                )
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

        if plan_adherence is not None and plan_adherence < -0.3:
            parts.append(f"off-plan={plan_adherence:.2f}")

        return " | ".join(parts) if parts else "default"

    # ------------------------------------------------------------------
    # Structured rationale — Anthropic XML-style
    # ------------------------------------------------------------------
    def _build_structured_rationale(
        self,
        action: str,
        belief: np.ndarray,
        anomaly: bool,
        cusum: Dict[str, float],
        plan_adherence: Optional[float],
        monitoring_level: str,
    ) -> Dict[str, Any]:
        p_h = float(belief[STATE_HEALTHY])
        p_d = float(belief[STATE_DEGRADED])
        p_b = float(belief[STATE_BROKEN])

        # --- Thinking: a 1-2 sentence internal thought ---
        thinking_parts = [f"P(H)={p_h:.2f} P(D)={p_d:.2f} P(B)={p_b:.2f}"]
        if anomaly:
            thinking_parts.append("CUSUM alarm active")
        if cusum.get("S", 0) > self.cusum.h * 0.5:
            thinking_parts.append(f"drift S={cusum['S']:.2f} approaching threshold h={self.cusum.h}")
        if plan_adherence is not None:
            thinking_parts.append(f"plan adherence={plan_adherence:.2f}")
        thinking = "; ".join(thinking_parts) + "."

        # --- Detected issue ---
        if action == ACTION_CONTINUE:
            detected = "none — agent appears healthy"
        elif action == ACTION_REPLAN:
            detected = (
                f"plan drift (adherence={plan_adherence:.2f}) with "
                f"elevated degraded belief (P(D)={p_d:.2f})"
            )
        elif action == ACTION_CORRECT:
            detected = (
                f"tool failures or progress stall; "
                f"P(D)={p_d:.2f} above correction threshold"
            )
        elif action == ACTION_ESCALATE:
            detected = (
                f"agent likely broken; P(B)={p_b:.2f}, "
                f"CUSUM S={cusum.get('S', 0):.2f}"
            )
        elif action == ACTION_GATHER:
            detected = (
                f"ambiguous belief — insufficient evidence to decide; "
                f"entropy high"
            )
        else:
            detected = "unknown"

        # --- Evidence ---
        evidence = {
            "belief": {"healthy": round(p_h, 3), "degraded": round(p_d, 3), "broken": round(p_b, 3)},
            "anomaly": anomaly,
            "cusum_drift": round(cusum.get("S", 0), 3),
            "cusum_h": self.cusum.h,
            "step_count": self.step_count,
        }
        if plan_adherence is not None:
            evidence["plan_adherence"] = round(plan_adherence, 3)

        return {
            "thinking": thinking,
            "detected_issue": detected,
            "recommended_action": action,
            "confidence": round(
                p_h if action == ACTION_CONTINUE
                else p_d if action in (ACTION_CORRECT, ACTION_REPLAN)
                else p_b,
                3,
            ),
            "evidence": evidence,
            "monitoring_level": monitoring_level,
        }

    # ------------------------------------------------------------------
    # Handover report — rich context for escalation
    # ------------------------------------------------------------------
    def _build_handover_report(
        self,
        belief: np.ndarray,
        cusum: Dict[str, float],
        plan_adherence: Optional[float],
    ) -> Dict[str, Any]:
        """Generate a structured handover report for human / next agent on escalation."""
        recent_actions = [d.action for d in self.decision_log[-10:]]
        anomaly_steps = [
            i + 1 for i, d in enumerate(self.decision_log) if d.anomaly
        ]

        summary = (
            f"Agent escalated at step {self.step_count}. "
            f"Belief: P(Healthy)={belief[STATE_HEALTHY]:.2f}, "
            f"P(Degraded)={belief[STATE_DEGRADED]:.2f}, "
            f"P(Broken)={belief[STATE_BROKEN]:.2f}. "
            f"CUSUM drift={cusum.get('S', 0):.2f} (threshold h={self.cusum.h}). "
            f"Anomalies detected at steps: {anomaly_steps[-5:] if anomaly_steps else 'none'}. "
            f"Recent action sequence: {recent_actions[-5:]}. "
            + (
                f"Plan adherence was {plan_adherence:.2f} (significantly off-track). "
                if plan_adherence is not None and plan_adherence < -0.3 else ""
            )
        )

        return {
            "escalation_step": self.step_count,
            "belief_snapshot": {
                "healthy": round(float(belief[STATE_HEALTHY]), 4),
                "degraded": round(float(belief[STATE_DEGRADED]), 4),
                "broken": round(float(belief[STATE_BROKEN]), 4),
            },
            "cusum_drift": round(cusum.get("S", 0), 4),
            "cusum_threshold": self.cusum.h,
            "recent_actions": recent_actions[-10:],
            "anomaly_steps": anomaly_steps,
            "plan_adherence": round(plan_adherence, 3) if plan_adherence is not None else None,
            "summary": summary,
            "recommendation": (
                "Review the last few tool outputs. Identify whether this is a "
                "genuine failure or a false alarm. If genuine, consider restarting "
                "from the last known-good checkpoint with a revised plan. "
                "If false alarm, the CUSUM threshold h may need tuning for this domain."
            ),
        }

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
        self.cumulative_tokens_est = 0
        self.cumulative_cost_est = 0.0

    def clone(self) -> "DecisionEngine":
        """
        Create a fresh engine with the same *configuration and trained parameters*
        but zero runtime state (step_count, belief, event history, decision_log).

        Use this for per-task isolation — one clone per concurrent agent run,
        so belief state never bleeds across tasks.

        Returns
        -------
        DecisionEngine
            New engine sharing the configuration of ``self``, ready for fresh use.
        """
        import copy

        # --- Build a new HMM with current (possibly trained) parameters ---
        hmm_clone = HiddenMarkovModel(
            prior=self.hmm.prior.copy(),
            transition=self.hmm.T.copy(),
            emission_tables={k: v.copy() for k, v in self.hmm.B.items()},
        )
        # If the source HMM uses a non-default emission model, clone its config
        if self.hmm._emission_model is not None:
            em = self.hmm._emission_model
            if hasattr(em, "cfg"):
                hmm_clone._emission_model = type(em)(config=copy.copy(em.cfg))
            else:
                hmm_clone._emission_model = type(em)(
                    tables={k: v.copy() for k, v in em.tables.items()}
                )
            hmm_clone.log_B = {
                dim: np.log(tbl + 1e-12) for dim, tbl in hmm_clone.B.items()
            }

        # --- Build a new Hawkes with same parameters ---
        hawkes_clone = HawkesProcess(
            mu=self.hawkes.mu.copy(),
            alpha=self.hawkes.alpha.copy(),
            beta=self.hawkes.beta,
            max_history=self.hawkes.max_history,
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        # --- Build a new CUSUM with same thresholds ---
        cusum_clone = CUSUMDetector(
            h=self.cusum.h,
            gamma=self.cusum.gamma,
            drift_floor=self.cusum.drift_floor,
        )

        # --- Build the new engine ---
        cloned = DecisionEngine(
            hmm=hmm_clone,
            hawkes=hawkes_clone,
            cusum=cusum_clone,
            reward=copy.deepcopy(self._reward_config),
            use_pomdp=self.use_pomdp,
            use_pomcp=self.use_pomcp,
            use_fast_pomcp=self.use_fast_pomcp,
            use_corrective=self.use_corrective,
            use_content_signals=self.use_content_signals,
            pomdp_resolution=self._pomdp_resolution,
            pomcp_n_simulations=self._pomcp_simulations,
            pomcp_n_particles=self._pomcp_particles,
            seed=int(self.rng.integers(0, 2**31)),
            anthropic_mode=self.anthropic_mode,
            cost_aware=self.cost_aware,
            enable_replan=self.enable_replan,
        )

        cloned._entropy_threshold = self._entropy_threshold
        cloned.theta_broken = self.theta_broken
        cloned.theta_degraded = self.theta_degraded
        cloned.theta_healthy = self.theta_healthy
        cloned.hysteresis_margin = self.hysteresis_margin
        cloned._token_cost_per_k = self._token_cost_per_k
        cloned._token_cost_per_k_output = self._token_cost_per_k_output

        return cloned

    # ------------------------------------------------------------------
    # State persistence — save/load for long-running agents
    # ------------------------------------------------------------------
    def save_state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for warm restart."""
        import json
        events = []
        for ev in self.hawkes.events:
            events.append({
                "time": ev.time,
                "event_type": ev.event_type,
                "mark": ev.mark,
            })

        log_alpha = None
        if self.hmm.log_alpha is not None:
            log_alpha = [float(x) for x in self.hmm.log_alpha]

        lengths = None
        if self.content_extractor:
            lengths = list(self.content_extractor._lengths)

        return {
            "version": "0.3.0",
            "step_count": self.step_count,
            "prev_action": self.prev_action,
            "prev_belief": (
                [float(x) for x in self.prev_belief]
                if self.prev_belief is not None else None
            ),
            "hmm": {
                "log_alpha": log_alpha,
                "t": self.hmm.t,
            },
            "hawkes": {
                "events": events,
                "current_time": self.hawkes.current_time,
            },
            "cusum": {
                "S": self.cusum.S,
                "t": self.cusum.t,
            },
            "content_lengths": lengths,
            "cumulative_tokens_est": self.cumulative_tokens_est,
            "cumulative_cost_est": self.cumulative_cost_est,
        }

    def load_state(self, snapshot: Dict[str, Any]):
        """Restore engine from a snapshot produced by save_state()."""
        self.reset()

        self.step_count = snapshot.get("step_count", 0)
        self.prev_action = snapshot.get("prev_action")
        prev = snapshot.get("prev_belief")
        if prev is not None:
            self.prev_belief = np.array(prev, dtype=np.float64)

        # HMM
        hmm_s = snapshot.get("hmm", {})
        self.hmm.t = hmm_s.get("t", 0)
        la = hmm_s.get("log_alpha")
        if la is not None:
            self.hmm.log_alpha = np.array(la, dtype=np.float64)

        # Hawkes
        hw_s = snapshot.get("hawkes", {})
        self.hawkes.reset()
        self.hawkes.current_time = hw_s.get("current_time", 0.0)
        for ev in hw_s.get("events", []):
            self.hawkes.add_event(
                ev["time"], ev["event_type"], mark=ev["mark"],
            )

        # CUSUM
        cu_s = snapshot.get("cusum", {})
        self.cusum.S = cu_s.get("S", 0.0)
        self.cusum.t = cu_s.get("t", 0)

        # Content extractor lengths
        if self.content_extractor:
            self.content_extractor._lengths.clear()
            for v in snapshot.get("content_lengths", []) or []:
                self.content_extractor._lengths.append(v)

        # Cost tracking
        self.cumulative_tokens_est = snapshot.get("cumulative_tokens_est", 0)
        self.cumulative_cost_est = snapshot.get("cumulative_cost_est", 0.0)
