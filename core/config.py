"""
Single-point configuration for the DecisionEngine.

Instead of passing 20 constructor params, create a config and pass it:

    from core.config import EngineConfig
    cfg = EngineConfig.preset("conservative", use_content_signals=True)
    engine = DecisionEngine.from_config(cfg)
"""

from dataclasses import dataclass, field
from typing import Optional

from .pomdp import RewardConfig


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

    # ---- Misc ----
    seed: Optional[int] = None

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
        }
        cfg = presets.get(name, cls())
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
