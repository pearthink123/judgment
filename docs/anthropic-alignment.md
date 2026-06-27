# Anthropic Alignment

How judgment's design aligns with Anthropic's official agent architecture principles
(2024–2026), and how to use the `anthropic_mode` preset to get the best of both worlds.

---

## The Anthropic Agent Philosophy

Anthropic's agent architecture has a clear through-line from their 2024
"Building Effective Agents" guide through Claude Code (2025) to Claude
Managed Agents & Dreams (2026).  Four principles stand out:

| Principle | Anthropic Says | Judgment Does |
|---|---|---|
| **Radical simplicity** | "The core agent is just environment + tools + system prompt in a loop" | judgment is a **4th piece** — a math-driven step-level process monitor that wraps the loop, not rewrites it |
| **Close the loop** | "Give the model a way to verify its own output" | CorrectiveRouter produces actionable advice (verify / rethink / retry / rollback) |
| **Process over outcome** | "Grade outcomes, not paths" | CUSUM + HMM monitors step-level signals (tool health, progress, plan drift), not final correctness |
| **Shrink scaffolding as models improve** | "After each model release, comment out harness pieces to see what's load-bearing" | Graduated monitoring intensity: threshold gate → grid POMDP → FastPOMCP. Low-cost paths activate first. |

---

## What `anthropic_mode` Enables

```python
from judgment import DecisionEngine
from judgment.core.config import EngineConfig

engine = DecisionEngine.from_config(EngineConfig.preset("anthropic"))
```

| Feature | Default | `anthropic_mode` | Reference |
|---|---|---|---|
| `structured_rationale` | `{}` | XML-style thinking / detected_issue / evidence | Anthropic's Model Spec `<thinking>` pattern |
| Handover report | `None` | Rich escalation context for human / next agent | Computer Use human-in-the-loop |
| REPLAN action | absent | Triggered on plan_adherence < -0.5 | "Close the loop" replanning |
| Cost-aware tracking | off | Cumulative token/cost estimates in diagnostics | Token budget discipline |
| Corrective tone | direct | Constructive, non-judgmental ("what can improve") | Model Spec honesty & helpfulness |
| Plan adherence signals | ignored | CUSUM penalises plan deviation | Process supervision |
| Content signals | off | Enabled | Richer observation for HMM |

---

## Architectural Parallels

### 1. Graduated Monitoring = 5-Layer Compaction

Anthropic's 5-layer compaction pipeline runs the cheapest compression first.
judgment mirrors this with graduated monitoring intensity:

```
belief entropy low + healthy → threshold gate (<0.3ms)   ← c.f. Budget Reduction
belief entropy medium         → grid POMDP (<1ms)        ← c.f. Snip
belief entropy high + anomaly → FastPOMCP (25ms)         ← c.f. Microcompact
extreme uncertainty           → full POMCP (200ms)        ← c.f. Auto-compact
```

The `monitoring_level` field in `layer_diagnostics` tells you which level was used.

### 2. per-task Isolation = Sub-agent Isolation

Anthropic isolates sub-agents in fresh context windows. judgment's `clone()` provides
the same guarantee for the health monitor:

```python
base = DecisionEngine.from_config(EngineConfig.preset("anthropic"))

for task in batch:
    engine = base.clone()   # fresh Belief, Hawkes history, CUSUM state
    run_agent(task, engine)
```

### 3. Dreams Reversibility = Shadow HMM (roadmap)

Anthropic's Dreams writes to a **parallel** memory store, never mutating the input store.
This makes the dream reversible and evaluable. judgment's planned online learning
will follow the same pattern — a shadow HMM that can be promoted or discarded after
eval comparison.

### 4. SHADE-Arena Monitoring = CUSUM + HMM Persistence

Anthropic's sabotage-detection research found that revealing chain-of-thought to a
monitor dramatically improves detection. judgment's `save_state()` / `load_state()`
and structured diagnostics make the engine's internal reasoning transparent to
higher-level monitoring systems.

---

## Using AnthropicExecutor with judgment

```python
from judgment import AnthropicExecutor, JudgmentHarness

executor = AnthropicExecutor(
    model="claude-sonnet-4-20250514",
    api_key="...",
)

harness = JudgmentHarness(executor=executor)
# The harness automatically wraps each step with judgment health checks.
# In anthropic_mode, escalation produces a rich handover report that
# another Claude instance or a human can pick up.
```

When Anomaly fires and the engine escalates, the `handover_report` contains:

- Belief snapshot at escalation time
- Recent action sequence
- CUSUM drift trajectory
- Plan adherence (if tracked)
- Human-readable summary + recommendation

This follows the Anthropic "human-in-the-loop" pattern from Computer Use.

---

## What judgment Adds That Anthropic Doesn't (Yet) Ship

| Capability | Status in Anthropic Stack |
|---|---|
| CUSUM sequential change detection | Not available — Anthropic uses rule-based / classifier monitoring |
| Bayesian health inference (HMM) | Not available — Anthropic uses hierarchical summarization |
| POMDP optimal stopping policy | Not available — Anthropic uses fixed thresholds + human gating |
| FastPOMCP (25ms) | Not available — Anthropic uses LLM-based evaluation (slower) |

judgment is **complementary** to the Anthropic stack. It fills a gap between
"no monitoring" and "another full LLM call for judging" — a lightweight,
math-driven process supervisor that catches doomed runs before they burn tokens.

---

## References

- Anthropic. "Building Effective Agents." Dec 2024.
  https://www.anthropic.com/research/building-effective-agents
- Anthropic. "Steering Claude Code: Skills, Hooks, Subagents." 2025.
  https://claude.com/blog/steering-claude-code-skills-hooks-rules-subagents-and-more
- Anthropic. "Monitoring Computer Use via Hierarchical Summarization." Feb 2025.
  https://alignment.anthropic.com/2025/summarization-for-monitoring/
- Anthropic. "Dreams: Self-Improving Memory for Agents." May 2026.
  https://platform.claude.com/docs/en/managed-agents/dreams
- Zhang, D. "Useful Memories Become Faulty When Continuously Updated by LLMs."
  arXiv 2605.12978, May 2026.
- Anthropic Alignment. "SHADE-Arena: Evaluating Sabotage and Monitoring." Jun 2025.
  https://alignment.anthropic.com/2025/sabotage-risk-report/
