"""
Realistic fault models for Agent evaluation.

Four patterns that simulate how real LLM agents degrade:

  1. Context Drift  — output gradually drifts off-track (hardest to detect)
  2. Tool Degradation — tool calls fail more and more over time
  3. Loop Trap — agent gets stuck repeating the same tool
  4. Catastrophic Cascade — one failure poisons everything downstream
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Fault model type
# ---------------------------------------------------------------------------
FaultGenerator = Callable[[int, np.random.Generator], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Model 1: Context Drift (hardest)
# ---------------------------------------------------------------------------
def context_drift_generator(
    step: int,
    rng: np.random.Generator,
    drift_start: int = 6,
    drift_rate: float = 0.03,
) -> Dict[str, Any]:
    """
    Progress gradually decays without any explicit tool failure.

    The agent thinks it's doing fine (tool_ok=True) but progress is eroding.
    This is the hardest pattern — no clear error signal, just entropy.
    """
    if step < drift_start:
        return {
            "tool_ok": True,
            "progress_delta": 0.12 + 0.04 * rng.random(),
            "error_count_delta": 0,
            "has_user_msg": False,
            "llm_text": "Step executed successfully. Output looks correct.",
        }
    else:
        decay = drift_rate * (step - drift_start)  # 0.03 → 0.06 → 0.09 ...
        progress = max(0.0, 0.10 - decay + 0.02 * rng.random())
        # After severe drift, occasional errors appear
        tool_ok = rng.random() > 0.15
        return {
            "tool_ok": tool_ok,
            "progress_delta": progress,
            "error_count_delta": 0 if tool_ok else 1,
            "has_user_msg": False,
            "llm_text": "Step completed." if tool_ok else "Unexpected result from tool.",
        }


# ---------------------------------------------------------------------------
# Model 2: Tool Degradation (moderate)
# ---------------------------------------------------------------------------
def tool_degradation_generator(
    step: int,
    rng: np.random.Generator,
    healthy_steps: int = 5,
    degrade_over: int = 15,
) -> Dict[str, Any]:
    """
    Tool success rate linearly declines from 0.95 to 0.20 over degrade_over steps.
    """
    if step <= healthy_steps:
        fail_prob = 0.05
    else:
        t = min((step - healthy_steps) / degrade_over, 1.0)
        fail_prob = 0.05 + 0.75 * t  # 0.05 → 0.80

    tool_ok = rng.random() > fail_prob
    return {
        "tool_ok": tool_ok,
        "progress_delta": 0.14 if tool_ok else -0.04,
        "error_count_delta": 0 if tool_ok else 1,
        "has_user_msg": False,
        "llm_text": (
            "Operation successful." if tool_ok
            else "Error: tool returned unexpected status."
        ),
    }


# ---------------------------------------------------------------------------
# Model 3: Loop Trap (moderate)
# ---------------------------------------------------------------------------
def loop_trap_generator(
    step: int,
    rng: np.random.Generator,
    trap_start: int = 6,
    trap_duration: int = 7,
) -> Dict[str, Any]:
    """
    Agent gets stuck calling the same tool with zero progress.
    After trap_duration, recovery is possible but unlikely.
    """
    in_trap = trap_start <= step < trap_start + trap_duration

    if in_trap:
        # Stuck — same tool, no progress
        return {
            "tool_ok": True,
            "progress_delta": 0.0,
            "error_count_delta": 0,
            "has_user_msg": False,
            "llm_text": "Running check again... same result as before.",
        }
    elif step < trap_start:
        return {
            "tool_ok": True,
            "progress_delta": 0.15,
            "error_count_delta": 0,
            "has_user_msg": False,
            "llm_text": "Step executed normally.",
        }
    else:
        # Post-trap: may recover (30%) or stay broken
        if rng.random() < 0.30:
            return {
                "tool_ok": True,
                "progress_delta": 0.10,
                "error_count_delta": 0,
                "has_user_msg": False,
                "llm_text": "Recovered — found alternative approach.",
            }
        else:
            return {
                "tool_ok": False,
                "progress_delta": -0.02,
                "error_count_delta": 1,
                "has_user_msg": False,
                "llm_text": "Still stuck. Same error as before.",
            }


# ---------------------------------------------------------------------------
# Model 4: Catastrophic Cascade (easiest)
# ---------------------------------------------------------------------------
def catastrophic_cascade_generator(
    step: int,
    rng: np.random.Generator,
    trigger_step: int = 5,
) -> Dict[str, Any]:
    """
    One catastrophic failure at trigger_step poisons the rest of the run.
    High error rate and negative progress after the trigger.
    """
    if step < trigger_step:
        return {
            "tool_ok": True,
            "progress_delta": 0.14 + 0.04 * rng.random(),
            "error_count_delta": 0,
            "has_user_msg": False,
            "llm_text": "All systems nominal.",
        }
    elif step == trigger_step:
        # The catastrophic event
        return {
            "tool_ok": False,
            "progress_delta": -0.30,
            "error_count_delta": 3,
            "has_user_msg": False,
            "llm_text": "CRITICAL ERROR: state corruption detected. Halting.",
        }
    else:
        # Cascading failures
        tool_ok = rng.random() > 0.80  # 80% failure
        return {
            "tool_ok": tool_ok,
            "progress_delta": -0.06 if not tool_ok else 0.02,
            "error_count_delta": 2 if not tool_ok else 0,
            "has_user_msg": False,
            "llm_text": (
                "error" if not tool_ok else "attempting recovery"
            ),
        }


# ---------------------------------------------------------------------------
# Healthy generator
# ---------------------------------------------------------------------------
def healthy_generator(
    step: int,
    rng: np.random.Generator,
    base_fail_rate: float = 0.08,
) -> Dict[str, Any]:
    """Normal trajectory — mostly success, rare transient failures."""
    tool_ok = rng.random() > base_fail_rate
    return {
        "tool_ok": tool_ok,
        "progress_delta": 0.14 + 0.04 * rng.random() if tool_ok else -0.01,
        "error_count_delta": 0 if tool_ok else 1,
        "has_user_msg": False,
        "llm_text": "Step complete." if tool_ok else "Minor issue, retrying.",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
FAULT_MODELS: Dict[str, FaultGenerator] = {
    "context_drift": context_drift_generator,
    "tool_degradation": tool_degradation_generator,
    "loop_trap": loop_trap_generator,
    "catastrophic_cascade": catastrophic_cascade_generator,
    "healthy": healthy_generator,
}
