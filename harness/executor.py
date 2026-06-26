"""
LLM Executor — model-agnostic tool-calling interface.

Supports:
  - SimulatedExecutor: for testing / demo (no API key needed)
  - LLMExecutor: real API calls via OpenAI-compatible endpoint
    (DeepSeek, Anthropic via proxy, OpenAI, local models)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import json
import re

from .tools import ToolRegistry


@dataclass
class ExecutorOutput:
    """What the executor returns after one step."""

    text: str                          # LLM text response (if any)
    tool_ok: bool                      # did the primary tool succeed?
    tool_name: Optional[str] = None    # which tool was called
    tool_result: Optional[str] = None  # tool output
    error_count_delta: int = 0         # new errors this step
    raw_response: Optional[Any] = None # for debugging


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseExecutor(ABC):
    """Abstract executor — implement for each LLM backend."""

    @abstractmethod
    def run_step(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: ToolRegistry,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutorOutput:
        """One step: call LLM, execute tools, return structured output."""
        ...


# ---------------------------------------------------------------------------
# Simulated executor — for testing / demo
# ---------------------------------------------------------------------------
class SimulatedExecutor(BaseExecutor):
    """
    Fake executor that follows a script or random pattern.
    No API key needed.  Useful for testing the decision engine in isolation.
    """

    def __init__(self, script: Optional[List[str]] = None, seed: int = 42):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self._script = script or []
        self._script_pos = 0

    def run_step(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: ToolRegistry,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutorOutput:
        # If a script is provided, follow it
        if self._script_pos < len(self._script):
            action = self._script[self._script_pos]
            self._script_pos += 1
        else:
            # Default: mostly succeed
            if self.rng.random() < 0.85:
                action = "success"
            else:
                action = "error"

        if action == "success":
            return ExecutorOutput(
                text="Task progressing normally.",
                tool_ok=True,
                tool_name="simulated_tool",
                tool_result="Simulated success output.",
                error_count_delta=0,
            )
        elif action == "error":
            return ExecutorOutput(
                text="Encountered an issue.",
                tool_ok=False,
                tool_name="simulated_tool",
                tool_result="Simulated error: something went wrong.",
                error_count_delta=1,
            )
        else:
            return ExecutorOutput(
                text=f"Simulated: {action}",
                tool_ok=True,
                tool_name="simulated_tool",
                tool_result=f"Simulated output for {action}.",
                error_count_delta=0,
            )


# ---------------------------------------------------------------------------
# Real LLM executor — OpenAI-compatible API
# ---------------------------------------------------------------------------
class LLMExecutor(BaseExecutor):
    """
    Executor that calls a real LLM via OpenAI-compatible API.

    Works with DeepSeek, OpenAI, Anthropic (via proxy), and any
    openai-compatible endpoint.

    Parameters
    ----------
    model : str — model name (e.g. "deepseek-chat").
    api_key : str or None — if None, reads from env var.
    base_url : str or None — override API base URL.
    temperature : float.
    max_tokens : int.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if api_key is None:
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_key = api_key

        self.base_url = base_url

    def run_step(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: ToolRegistry,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutorOutput:
        if self.api_key is None:
            return ExecutorOutput(
                text="",
                tool_ok=False,
                error_count_delta=1,
                tool_result="[error] No API key configured. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.",
            )

        try:
            from openai import OpenAI
        except ImportError:
            return ExecutorOutput(
                text="",
                tool_ok=False,
                error_count_delta=1,
                tool_result="[error] openai package not installed. Run: pip install openai",
            )

        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        client = OpenAI(**kwargs)

        # Build full message list
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        tool_schemas = tools.openai_schemas() if tools.list_names() else None

        try:
            if tool_schemas:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        except Exception as e:
            return ExecutorOutput(
                text="",
                tool_ok=False,
                error_count_delta=1,
                tool_result=f"[error] LLM API call failed: {e}",
            )

        choice = response.choices[0]
        msg = choice.message

        text = msg.content or ""

        # Tool calls?
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_result = tools.execute(tool_name, tool_args)
            tool_ok = not tool_result.startswith("[error]") and not tool_result.startswith("[tool error]")
            error_delta = 0 if tool_ok else 1

            return ExecutorOutput(
                text=text,
                tool_ok=tool_ok,
                tool_name=tool_name,
                tool_result=tool_result,
                error_count_delta=error_delta,
                raw_response=msg,
            )

        # No tool calls — just text
        return ExecutorOutput(
            text=text,
            tool_ok=True,  # text generation itself didn't error
            tool_name=None,
            tool_result=text,
            error_count_delta=0,
            raw_response=msg,
        )


# ---------------------------------------------------------------------------
# Progress estimator (pluggable)
# ---------------------------------------------------------------------------
def default_progress_estimator(output: ExecutorOutput) -> float:
    """
    Heuristic progress delta from executor output.

    Users should replace this with a task-specific estimator.
    """
    if output.tool_ok:
        if output.tool_result and len(output.tool_result) > 100:
            return 0.08   # substantial output → some progress
        return 0.04        # minimal output
    else:
        return -0.03        # failure → slight regression
