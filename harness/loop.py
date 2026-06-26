"""
JudgmentHarness — unified agent execution loop with math-driven oversight.

The harness wraps an LLM executor, tool registry, and DecisionEngine
into a single ReAct-style loop.  The engine watches every step and can:

  continue  — let the LLM proceed with its next planned action
  correct   — inject corrective advice into the LLM's context
  escalate  — stop and return control to the user
  gather    — prompt the LLM to collect more info before deciding

Usage:
    from judgment import JudgmentHarness

    harness = JudgmentHarness()
    result = harness.run("Write a Python function that finds prime numbers.")
    print(result.summary)
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import time

from core.engine import (
    DecisionEngine, Decision,
    ACTION_CONTINUE, ACTION_CORRECT, ACTION_ESCALATE, ACTION_GATHER,
)
from core.pomdp import RewardConfig
from core.corrective import CorrectiveAdvice
from core.training import train_hmm

from .tools import ToolRegistry, default_registry
from .executor import (
    BaseExecutor, SimulatedExecutor, LLMExecutor,
    ExecutorOutput, default_progress_estimator,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """What you get back from harness.run()."""

    status: str                      # "success", "escalated", "max_steps", "error"
    steps: int
    final_belief: Dict[str, float]
    decision_log: List[Decision]
    messages: List[Dict[str, Any]]   # full conversation (for inspection)
    summary: str                     # human-readable one-liner
    duration_seconds: float


# ---------------------------------------------------------------------------
# System prompt template (pluggable)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are a task-completing agent with access to tools.
Work through the task step by step.

At each step:
1. Think about what you need to do next.
2. Call the appropriate tool to make progress.
3. Observe the result and plan the next step.

When you are done, state the final answer clearly.

The system will occasionally inject advisory messages — pay attention to them.
If you see [CORRECTIVE ADVICE], adjust your approach accordingly.
If you see [GATHERING], focus on collecting information before acting."""


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class JudgmentHarness:
    """
    Complete Agent harness with math-driven decision oversight.

    Parameters
    ----------
    executor : BaseExecutor
        The LLM backend.  If None, uses SimulatedExecutor (for testing).
    tools : ToolRegistry
        Available tools.  If None, uses default (read_file, write_file, etc.).
    engine : DecisionEngine
        The math oversight engine.  If None, creates one with POMDP.
    reward : RewardConfig or str
        Reward preset for the POMDP solver.
    system_prompt : str
        Override the default system prompt.
    progress_fn : callable
        (ExecutorOutput) -> float.  Estimates progress delta from tool output.
    max_steps : int
        Safety limit.
    """

    def __init__(
        self,
        executor: Optional[BaseExecutor] = None,
        tools: Optional[ToolRegistry] = None,
        engine: Optional[DecisionEngine] = None,
        reward: Optional[RewardConfig] = None,
        system_prompt: Optional[str] = None,
        progress_fn: Optional[Callable[[ExecutorOutput], float]] = None,
        max_steps: int = 40,
        seed: Optional[int] = None,
    ):
        self.executor = executor or SimulatedExecutor(seed=seed or 42)
        self.tools = tools or default_registry()
        self.engine = engine or DecisionEngine(
            reward=reward, use_pomdp=True, use_corrective=True, seed=seed,
        )
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.progress_fn = progress_fn or default_progress_estimator
        self.max_steps = max_steps

        self._messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """
        Execute a task with math-driven oversight.

        Parameters
        ----------
        task : str
            Task description (e.g. "Write a function to merge two sorted lists").
        initial_context : dict or None
            Extra context passed to the executor each step.

        Returns
        -------
        RunResult
        """
        t_start = time.time()

        self.engine.reset()
        self._messages = [{"role": "user", "content": task}]

        obs: Dict[str, Any] = {
            "tool_ok": True,
            "progress_delta": 0.0,
            "has_user_msg": False,
            "error_count_delta": 0,
        }

        # ---- Main loop ----
        for step_idx in range(1, self.max_steps + 1):
            # 1. Engine decides
            decision = self.engine.step(obs)

            if decision.action == ACTION_ESCALATE:
                return RunResult(
                    status="escalated",
                    steps=step_idx,
                    final_belief=decision.belief,
                    decision_log=list(self.engine.decision_log),
                    messages=list(self._messages),
                    summary=f"Escalated at step {step_idx}: {decision.rationale}",
                    duration_seconds=round(time.time() - t_start, 2),
                )

            # 2. Build executor input based on decision
            ctx = dict(initial_context or {})

            if decision.action == ACTION_CORRECT and decision.corrective_advice:
                advice = decision.corrective_advice
                corrective_msg = (
                    f"[CORRECTIVE ADVICE] {advice.summary}\n"
                    f"  Suggested action: {advice.action_type}\n"
                    f"  Evidence: failures={advice.recent_failures}, "
                    f"stalled={advice.progress_stalled_steps} steps, "
                    f"dominant state={advice.dominant_state}"
                )
                self._messages.append({"role": "user", "content": corrective_msg})
                ctx["corrective"] = True

            elif decision.action == ACTION_GATHER:
                gather_msg = (
                    "[GATHERING] The system is uncertain about the current state. "
                    "Focus on collecting information (read files, check outputs, verify "
                    "assumptions) before taking action."
                )
                self._messages.append({"role": "user", "content": gather_msg})
                ctx["gathering"] = True

            # 3. Execute
            try:
                output = self.executor.run_step(
                    system_prompt=self.system_prompt,
                    messages=self._messages,
                    tools=self.tools,
                    context=ctx,
                )
            except Exception as e:
                obs = {
                    "tool_ok": False,
                    "progress_delta": -0.05,
                    "has_user_msg": False,
                    "error_count_delta": 1,
                }
                self._messages.append({
                    "role": "assistant",
                    "content": f"[executor error] {e}",
                })
                continue

            # 4. Record in conversation
            assistant_content = output.text or ""
            if output.tool_name:
                assistant_content += (
                    f"\n\n[Tool: {output.tool_name}]\n{output.tool_result or ''}"
                )
            self._messages.append({"role": "assistant", "content": assistant_content})

            # 5. Build next observation
            progress_delta = self.progress_fn(output)
            has_user_msg = output.tool_result and "user" in output.tool_result.lower()

            obs = {
                "tool_ok": output.tool_ok,
                "progress_delta": progress_delta,
                "has_user_msg": has_user_msg,
                "error_count_delta": output.error_count_delta,
            }

            # 6. Check task completion (simple heuristic)
            if output.text and any(
                kw in output.text.lower()
                for kw in ["task complete", "done.", "final answer", "finished"]
            ):
                return RunResult(
                    status="success",
                    steps=step_idx,
                    final_belief=decision.belief,
                    decision_log=list(self.engine.decision_log),
                    messages=list(self._messages),
                    summary=f"Task completed in {step_idx} steps.",
                    duration_seconds=round(time.time() - t_start, 2),
                )

        # Exhausted max steps
        final_belief = (
            self.engine.decision_log[-1].belief
            if self.engine.decision_log
            else {"healthy": 1.0, "degraded": 0.0, "broken": 0.0}
        )
        return RunResult(
            status="max_steps",
            steps=self.max_steps,
            final_belief=final_belief,
            decision_log=list(self.engine.decision_log),
            messages=list(self._messages),
            summary=f"Reached max steps ({self.max_steps}) without completing.",
            duration_seconds=round(time.time() - t_start, 2),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def learn(
        self,
        logs: List[List[Dict[str, Any]]],
        labels: Optional[List[Dict[int, int]]] = None,
        n_iter: int = 50,
    ):
        """
        Update the HMM parameters from Agent run logs.

        Parameters
        ----------
        logs : list of trajectories
            Each is a list of observation dicts (one per step).
        labels : optional semi-supervised state labels.
        n_iter : EM iterations.
        """
        trained_hmm = train_hmm(logs, labels=labels, n_iter=n_iter)
        self.engine.hmm = trained_hmm
        # Recompute POMDP policy with new HMM (emission tables changed)
        from core.pomdp import get_policy
        self.engine._policy = get_policy(
            reward=self.engine._policy.reward if self.engine._policy else None,
            resolution=self.engine._pomdp_resolution,
            force_recompute=True,
        )

    @property
    def hmm(self):
        return self.engine.hmm

    @property
    def decision_log(self) -> List[Decision]:
        return list(self.engine.decision_log)
