"""
Diagnostics helpers — structured data for visualisation and logging.

All functions here produce plain dicts / lists that are safe to serialise
(JSON, DataFrame rows, etc.).  No matplotlib/streamlit dependencies in core.
"""

from typing import Dict, Any, List
import numpy as np

from .engine import DecisionEngine, Decision
from .hmm import HiddenMarkovModel, STATE_NAMES
from .cusum import CUSUMDetector
from .hawkes import HawkesProcess


def engine_snapshot(engine: DecisionEngine) -> Dict[str, Any]:
    """Full diagnostic snapshot of a running engine."""
    h = engine.hmm
    c = engine.cusum
    hw = engine.hawkes

    belief = h.belief_dict() if h is not None else {}

    return {
        "step": engine.step_count,
        "belief": belief,
        "cusum_S": round(c.S, 4) if c else None,
        "cusum_alarms": c.alarm_count if c else None,
        "hawkes_intensity": (
            hw.intensity().tolist() if hw else None
        ),
        "hawkes_n_events": len(hw.events) if hw else None,
        "prev_action": engine.prev_action,
    }


def decision_trace(engine: DecisionEngine) -> List[Dict[str, Any]]:
    """Return all decisions as a list of dicts (for DataFrame conversion)."""
    return [
        {
            "step": i + 1,
            "action": d.action,
            "confidence": d.confidence,
            "P_H": d.belief.get("healthy", 0),
            "P_D": d.belief.get("degraded", 0),
            "P_B": d.belief.get("broken", 0),
            "drift": d.drift,
            "anomaly": d.anomaly,
        }
        for i, d in enumerate(engine.decision_log)
    ]


def hmm_state_report(hmm: HiddenMarkovModel) -> Dict[str, Any]:
    """Read-only report of HMM internal state."""
    b = hmm.belief()
    return {
        "belief": {STATE_NAMES[i]: round(float(b[i]), 4) for i in range(3)},
        "most_likely": STATE_NAMES[hmm.most_likely_state()],
        "steps_processed": hmm.t,
        "prior": {STATE_NAMES[i]: round(float(hmm.prior[i]), 4) for i in range(3)},
        "transition_matrix": {
            f"{STATE_NAMES[r]}→{STATE_NAMES[c]}": round(float(hmm.T[r, c]), 4)
            for r in range(3)
            for c in range(3)
        },
    }


def cusum_report(cusum: CUSUMDetector) -> Dict[str, Any]:
    """Read-only report of CUSUM detector state."""
    return {
        "S_current": round(cusum.S, 4),
        "threshold_h": cusum.h,
        "gamma": cusum.gamma,
        "alarms_fired": cusum.alarm_count,
        "alarm_steps": list(cusum.alarm_history),
        "S_trace": [round(s, 4) for s in cusum.S_history[-30:]],
    }


def hawkes_report(hawkes: HawkesProcess) -> Dict[str, Any]:
    """Read-only report of Hawkes process state."""
    diag = hawkes.get_diagnostics()
    return {
        "intensities": [round(float(x), 4) for x in diag.intensities],
        "baseline": [round(float(x), 4) for x in diag.baseline],
        "n_events": diag.n_events,
        "spectral_radius": round(hawkes.spectral_radius, 4),
        "stationary": hawkes.check_stationarity(),
        "recent_event_types": [
            {"type": ev.event_type, "mark": round(ev.mark, 3)}
            for ev in diag.recent_events[-5:]
        ],
    }
