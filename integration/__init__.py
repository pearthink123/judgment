"""
Harness integration adapters — drop judgment into existing agent frameworks.

- base.py       : Abstract adapter protocol (framework-agnostic)
- langgraph.py  : LangGraph node + conditional-edge router
- crewai.py     : CrewAI step callback + health-check tool + tool wrapper
- custom.py     : Generic adapter — wrap_step decorator + judgment_guard context
"""

from .base import BaseAdapter, ObservationExtractor, RouteTargets, default_extractor

__all__ = [
    "BaseAdapter",
    "ObservationExtractor",
    "RouteTargets",
    "default_extractor",
]
