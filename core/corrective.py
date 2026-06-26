"""
Corrective action router (Layer 3 post-decision).

When the DecisionEngine decides CORRECT, this module analyses the
observation history and belief trajectory to produce:

  1. An evidence summary (for the Planner to consume)
  2. A corrective action type suggestion (verify / rethink / retry / rollback)

These are **explicitly heuristic** — no mathematical pretense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Decision


# ---------------------------------------------------------------------------
# Corrective action types
# ---------------------------------------------------------------------------
CORRECTIVE_VERIFY = "verify"
CORRECTIVE_RETHINK = "rethink"
CORRECTIVE_RETRY = "retry"
CORRECTIVE_ROLLBACK = "rollback"


@dataclass
class CorrectiveAdvice:
    """Structured output for the Planner when CORRECT is signalled."""

    action_type: str
    evidence: Dict[str, Any]
    summary: str

    recent_failures: int = 0
    consecutive_same_error: bool = False
    progress_stalled_steps: int = 0
    avg_tool_success_rate: float = 1.0
    dominant_state: str = "healthy"
    cusum_alarmed: bool = False


class CorrectiveRouter:
    """
    Rule-based corrective action selector.

    Rules (priority order):
      1. Same error repeated ≥3 times + negative progress → ROLLBACK
      2. ≥2 tool failures in last 3 steps → VERIFY
      3. Progress stalled ≥5 steps → RETHINK
      4. Isolated single failure → RETRY
    """

    def __init__(self):
        pass

    def analyse(self, decision_log: list) -> CorrectiveAdvice:
        """Examine decision history and produce corrective advice."""
        n = len(decision_log)
        if n == 0:
            return CorrectiveAdvice(
                action_type=CORRECTIVE_VERIFY,
                evidence={},
                summary="No history available — verify current state.",
            )

        recent = decision_log[-10:]

        recent_failures = sum(
            1 for d in recent[-5:]
            if d.anomaly
        )

        consecutive_same = (
            sum(1 for d in recent[-3:] if d.anomaly) >= 3
        )

        stalled_steps = 0
        for i in range(1, min(n, 10)):
            d_prev = decision_log[-i - 1]
            d_curr = decision_log[-i]
            p_prev = d_prev.belief.get("healthy", 0.5)
            p_curr = d_curr.belief.get("healthy", 0.5)
            if p_curr <= p_prev + 0.02:
                stalled_steps += 1
            else:
                break

        avg_confidence = (
            sum(d.confidence for d in recent) / len(recent) if recent else 1.0
        )

        current_belief = decision_log[-1].belief
        dominant = max(current_belief, key=current_belief.get)  # type: ignore[arg-type]

        alarmed = decision_log[-1].anomaly

        evidence: Dict[str, Any] = {
            "recent_failures": recent_failures,
            "consecutive_same_error": consecutive_same,
            "stalled_steps": stalled_steps,
        }

        if consecutive_same and stalled_steps >= 2:
            action_type = CORRECTIVE_ROLLBACK
            summary = (
                f"Repeated failures ({recent_failures} in last 5 steps) with "
                f"stalled progress ({stalled_steps} steps). Consider rollback."
            )
        elif recent_failures >= 2:
            action_type = CORRECTIVE_VERIFY
            summary = (
                f"{recent_failures} tool failures in last 5 steps. "
                f"Verify current state before continuing."
            )
        elif stalled_steps >= 5:
            action_type = CORRECTIVE_RETHINK
            summary = (
                f"Progress stalled for {stalled_steps} steps. "
                f"Current approach may not be working — reconsider strategy."
            )
        elif recent_failures == 1:
            action_type = CORRECTIVE_RETRY
            summary = (
                "Single isolated failure — retry the last action "
                "before escalating."
            )
        else:
            action_type = CORRECTIVE_VERIFY
            summary = (
                f"Belief in {dominant} state elevated "
                f"(H={current_belief.get('healthy', 0):.2f}, "
                f"D={current_belief.get('degraded', 0):.2f}, "
                f"B={current_belief.get('broken', 0):.2f}). Verify."
            )

        return CorrectiveAdvice(
            action_type=action_type,
            evidence=evidence,
            summary=summary,
            recent_failures=recent_failures,
            consecutive_same_error=consecutive_same,
            progress_stalled_steps=stalled_steps,
            avg_tool_success_rate=avg_confidence,
            dominant_state=dominant,
            cusum_alarmed=alarmed,
        )
