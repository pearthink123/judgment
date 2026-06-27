"""
Single-point configuration for the DecisionEngine.

Instead of passing 20 constructor params, create a config and pass it:

    from core.config import EngineConfig
    cfg = EngineConfig.preset("conservative", use_content_signals=True)
    engine = DecisionEngine.from_config(cfg)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .pomdp import RewardConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_type(name: str, value, expected) -> None:
    """Raise TypeError if value is not an instance of expected."""
    if not isinstance(value, expected):
        type_name = getattr(expected, "__name__", str(expected))
        raise TypeError(
            f"{name} must be {type_name}, got {type(value).__name__} ({value!r})"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    # ---- Solver selection ----
    use_cusum: bool = True
    use_hawkes: bool = True
    use_hmm: bool = True
    use_pomdp: bool = False          # grid value iteration (3-state only)
    use_pomcp: bool = False          # recursive MCTS
    use_fast_pomcp: bool = True      # batch-optimised MCTS (default)

    use_corrective: bool = True      # heuristic corrective router
    use_content_signals: bool = False

    # ---- POMDP / POMCP ----
    reward: Optional[RewardConfig] = None
    pomdp_resolution: float = 0.05
    pomcp_n_simulations: int = 1000
    pomcp_n_particles: int = 200

    # ---- CUSUM ----
    cusum_h: float = 4.0
    cusum_gamma: float = 0.35

    # ---- Entropy fast-path ----
    entropy_threshold: float = 0.40  # skip solver when belief is peaked

    # ---- Hysteresis ----
    hysteresis_margin: float = 0.08

    # ---- Threshold fallback ----
    theta_healthy: float = 0.60
    theta_degraded: float = 0.35
    theta_broken: float = 0.45

    # ---- Anthropic alignment ----
    anthropic_mode: bool = False      # enable all Anthropic-aligned features
    cost_aware: bool = False          # track cumulative token cost estimates
    enable_replan: bool = False       # allow "replan" action on plan deviation

    # ---- Misc ----
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate field types and ranges.  Raises TypeError / ValueError."""
        _MAX_ENTROPY = math.log(3)  # ~1.099 for uniform 3-state belief

        # --- bool fields ---
        for name in ("use_cusum", "use_hawkes", "use_hmm",
                     "use_pomdp", "use_pomcp", "use_fast_pomcp",
                     "use_corrective", "use_content_signals",
                     "anthropic_mode", "cost_aware", "enable_replan"):
            _check_type(name, getattr(self, name), bool)

        # --- int fields ---
        for name in ("pomcp_n_simulations", "pomcp_n_particles"):
            _check_type(name, getattr(self, name), int)
        if self.pomcp_n_simulations < 1:
            raise ValueError(
                f"pomcp_n_simulations must be >= 1, got {self.pomcp_n_simulations}"
            )
        if self.pomcp_n_particles < 1:
            raise ValueError(
                f"pomcp_n_particles must be >= 1, got {self.pomcp_n_particles}"
            )

        # --- float fields ---
        for name in ("pomdp_resolution", "cusum_h", "cusum_gamma",
                     "entropy_threshold", "hysteresis_margin",
                     "theta_healthy", "theta_degraded", "theta_broken"):
            _check_type(name, getattr(self, name), (int, float))

        if not (0.0 < self.pomdp_resolution <= 1.0):
            raise ValueError(
                f"pomdp_resolution must be in (0, 1], got {self.pomdp_resolution}"
            )

        if self.cusum_h <= 0:
            raise ValueError(f"cusum_h must be > 0, got {self.cusum_h}")

        if not (0.0 <= self.cusum_gamma <= 1.0):
            raise ValueError(
                f"cusum_gamma must be in [0, 1], got {self.cusum_gamma}"
            )

        _max_entropy = math.log(3)
        if not (0.0 <= self.entropy_threshold <= _max_entropy):
            raise ValueError(
                f"entropy_threshold must be in [0, {_max_entropy:.3f}], "
                f"got {self.entropy_threshold}"
            )

        if not (0.0 <= self.hysteresis_margin <= 0.5):
            raise ValueError(
                f"hysteresis_margin must be in [0, 0.5], got {self.hysteresis_margin}"
            )

        # --- theta probabilities (must be 0-1) ---
        for name in ("theta_healthy", "theta_degraded", "theta_broken"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        # --- seed ---
        if self.seed is not None:
            _check_type("seed", self.seed, int)

        # --- reward ---
        if self.reward is not None and not isinstance(self.reward, RewardConfig):
            raise TypeError(
                f"reward must be RewardConfig or None, got {type(self.reward).__name__}"
            )

    @classmethod
    def preset(cls, name: str, **overrides) -> "EngineConfig":
        """Create a config from a named preset, with optional field overrides."""
        presets = {
            "default": cls(),
            "fast": cls(use_fast_pomcp=True),
            "conservative": cls(
                use_fast_pomcp=True,
                cusum_h=3.5,
                theta_broken=0.35,
                theta_degraded=0.28,
                reward=RewardConfig.preset("conservative"),
            ),
            "permissive": cls(
                use_fast_pomcp=True,
                cusum_h=5.5,
                theta_broken=0.55,
                theta_degraded=0.45,
                reward=RewardConfig.preset("permissive"),
            ),
            "lightweight": cls(
                use_cusum=True, use_hawkes=True, use_hmm=True,
                use_pomdp=False, use_pomcp=False, use_fast_pomcp=False,
                use_corrective=False, use_content_signals=False,
            ),
            "anthropic": cls(
                use_fast_pomcp=True,
                use_content_signals=True,
                use_corrective=True,
                cusum_h=4.0,
                anthropic_mode=True,
                cost_aware=True,
                enable_replan=True,
                reward=RewardConfig.preset("conservative"),
            ),
        }
        cfg = presets.get(name, cls())
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(
                    f"Unknown config field '{k}'. Available: {list(cls.__dataclass_fields__)}"
                )
            setattr(cfg, k, v)
        # Re-validate after overrides
        cfg.__post_init__()
        return cfg
