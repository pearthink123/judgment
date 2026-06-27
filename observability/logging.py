"""
Structured JSON logging for the Judgment Engine.

Usage:
    from observability.logging import JudgmentLogger

    logger = JudgmentLogger()
    logger.step(step=5, decision=decision, latency_ms=3.2)
    # → {"ts":"2026-06-27T...","event":"step","step":5,"action":"continue",...}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from core.engine import Decision

# ---------------------------------------------------------------------------
# JSON log handler
# ---------------------------------------------------------------------------
class _JsonHandler(logging.Handler):
    """Handler that formats log records as JSON lines."""

    def emit(self, record: logging.LogRecord):
        payload = getattr(record, "json_payload", None)
        if payload is None:
            # Fallback: plain text
            payload = {"msg": record.getMessage()}
        payload["level"] = record.levelname.lower()
        payload["ts"] = record.created
        print(json.dumps(payload, ensure_ascii=False))
        self.flush()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class JudgmentLogger:
    """
    Structured logger for engine events.

    Parameters
    ----------
    output : str — "stdout" (default) or file path.
    level : str — "info" (default), "debug", "warning".
    """

    def __init__(self, output: str = "stdout", level: str = "info"):
        self._logger = logging.getLogger(f"judgment.{id(self)}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        if output == "stdout":
            handler = _JsonHandler()
        else:
            handler = logging.FileHandler(output, encoding="utf-8")
        self._logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Event methods
    # ------------------------------------------------------------------
    def step(
        self,
        step: int,
        decision: Decision,
        latency_ms: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ):
        payload = {
            "event": "step",
            "step": step,
            "action": decision.action,
            "confidence": decision.confidence,
            "belief": decision.belief,
            "anomaly": decision.anomaly,
            "drift": decision.drift,
            "latency_ms": round(latency_ms, 1),
        }
        if extra:
            payload.update(extra)
        record = logging.LogRecord(
            name=self._logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )
        record.created = time.time()
        record.json_payload = payload
        self._logger.handle(record)

    def escalate(self, step: int, decision: Decision):
        payload = {
            "event": "escalate",
            "step": step,
            "belief": decision.belief,
            "rationale": decision.rationale,
        }
        record = logging.LogRecord(
            name=self._logger.name, level=logging.WARNING,
            pathname="", lineno=0, msg="", args=(), exc_info=None,
        )
        record.created = time.time()
        record.json_payload = payload
        self._logger.handle(record)

    def train_complete(self, n_trajectories: int, log_lik: float, duration_s: float):
        payload = {
            "event": "train_complete",
            "n_trajectories": n_trajectories,
            "final_log_lik": round(log_lik, 2),
            "duration_s": round(duration_s, 1),
        }
        record = logging.LogRecord(
            name=self._logger.name, level=logging.INFO,
            pathname="", lineno=0, msg="", args=(), exc_info=None,
        )
        record.created = time.time()
        record.json_payload = payload
        self._logger.handle(record)
