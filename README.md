# judgment

**Stop wasting tokens on doomed agent runs. 61% less wasted steps.**

A health monitor for LLM agents. Tells your agent loop when to continue,
when to fix something, and when to give up and ask for help — using math,
not vibes.

```bash
pip install judgment
```

```python
from judgment import quick_check, DecisionEngine

engine = DecisionEngine()  # FastPOMCP by default, 25ms per step

for step in agent_loop:
    action = quick_check(engine, tool_ok=True, progress_delta=0.1)
    if action == "escalate":
        break  # engine says: stop, this isn't going anywhere
```

## Why

LLM agents fail silently. A ReAct loop will happily run 50 steps on a
broken trajectory, burning tokens on tools that keep failing, before
anyone notices. Judgment watches the execution and says "stop" early.

| Metric | Without judgment | With judgment |
|---|---|---|
| Steps wasted on failed tasks | **100%** of max | **39%** of max |
| Catastrophic failures caught | 0% | **100%** (1 step delay) |
| False interruption of healthy runs | — | 8% (tunable) |

## 30 seconds to running

```bash
git clone https://github.com/pearthink123/judgment.git && cd judgment
pip install -e ".[dashboard]"

python examples/coding_agent_demo.py      # simulated agent, no API key
streamlit run dashboard/app.py            # live dashboard on localhost:8501
judgment run "Build an LRU cache"         # CLI
```

## Add to your agent

**LangGraph** — one node + conditional edge:
```python
from judgment.integration.langgraph import create_judgment_node, create_judgment_router

graph.add_node("judgment", create_judgment_node(engine))
graph.add_edge("tools", "judgment")
graph.add_conditional_edges("judgment", create_judgment_router(engine), {...})
```

**CrewAI** — step callback:
```python
from judgment.integration.crewai import create_judgment_callback

agent = Agent(step_callback=create_judgment_callback(engine), ...)
```

**Any loop** — one decorator:
```python
from judgment.integration.custom import wrap_step

@wrap_step(engine)
def my_step(state): ...

result, decision = my_step(state)
if decision.action == "escalate": break
```

## LLM providers

```python
from judgment import JudgmentHarness, LLMExecutor, AnthropicExecutor

# OpenAI / DeepSeek / Groq / vLLM (OpenAI-compatible API)
harness = JudgmentHarness(
    executor=LLMExecutor(
        model="groq/llama-4",     # or "deepseek-chat", "gpt-4o"
        base_url="https://api.groq.com/openai/v1",  # ← set for non-OpenAI
        api_key="...",
    ),
)

# Anthropic (native SDK)
harness = JudgmentHarness(
    executor=AnthropicExecutor(
        model="claude-sonnet-4-20250514",
        api_key="...",
    ),
)
```

## How it works

```
Agent step completes
        │
        ▼
┌───────────────────┐
│ Layer 1: CUSUM    │  "Is this normal fluctuation or a real anomaly?"
│ + Hawkes baseline │  Math: sequential change detection (Page 1954)
└────────┬──────────┘
         ▼
┌───────────────────┐
│ Layer 2: HMM      │  "How healthy is the agent right now?"
│ Healthy/Degraded/ │  Math: Bayesian filtering (Rabiner 1989)
│ Broken            │  + optional content signals (heuristic)
└────────┬──────────┘
         ▼
┌───────────────────┐
│ Layer 3: POMDP    │  "Continue, correct, escalate, or gather?"
│ (FastPOMCP, 25ms) │  Math: online MCTS (Silver & Veness 2010)
└────────┬──────────┘
         ▼
┌───────────────────┐
│ Corrective advice  │  "What specifically should we do?"
│ (heuristic)       │  Rules: verify / rethink / retry / rollback
└───────────────────┘
```

## What's math and what's not

| Component | Foundation | Status |
|---|---|---|
| CUSUM drift detector | Page (1954) | **Rigorous** — sequential hypothesis testing |
| Hawkes baseline | Hawkes (1971) | **Rigorous** — self-exciting point process |
| HMM Forward filter | Rabiner (1989) | **Rigorous** — Bayesian state inference |
| POMDP policy (grid) | Kaelbling et al. (1998) | **Rigorous** — exact value iteration |
| FastPOMCP | Silver & Veness (2010) | **Rigorous** — particle MCTS |
| Corrective action router | – | **Heuristic** (explicitly labelled, 4 rules) |
| Content signals (text metrics) | – | **Heuristic** (length, novelty, negation) |
| Threshold hysteresis | – | **Heuristic** (prevents oscillation)

## Ablation — does each layer actually help?

Marginal contribution of each component (synthetic benchmark, 5 fault models × 15 trajectories):

| Config | Detection recall | Precision |
|---|---|---|
| HMM only (no CUSUM, no POMDP) | 0% | — |
| +CUSUM | 33% | 67% |
| +POMDP grid | 47% | 82% |
| +FastPOMCP | 42% | 89% |
| Full stack (+content signals, +corrective) | 42% | **97%** |

**CUSUM is essential** — HMM alone can't detect faults. POMDP adds 14pp recall.
FastPOMCP trades 5pp for 8× speed. Content signals nearly eliminate false alarms.

```bash
python scripts/ablation.py            # reproduce these numbers
python scripts/benchmark.py --compare  # latency benchmark
```

## Real benchmark adapter

```bash
python scripts/benchmark_adapter.py --generate 50
```
Plug into any JSONL trace stream. On 50 synthetic traces: 60% detection, 0% FPR,
~910K tokens saved via early escalation.

## Run the benchmarks

```bash
python scripts/eval_runner.py --trajectories-per-model 25
```
Runs 125 baseline vs 125 judgment trajectories across 5 realistic fault
patterns (context drift, tool degradation, loop trap, catastrophic
cascade, healthy). Prints per-model breakdown and aggregate comparison.

```bash
python scripts/benchmark.py --trajectories 100 --compare
```
Side-by-side: grid POMDP vs FastPOMCP. Includes detection rate, false
positive rate, median delay, and latency.

## How is this different from X?

| Approach | Fixes bad agent runs by... | Needs config? | Math-based? |
|---|---|---|---|
| LangGraph retry | Retrying failed tool calls N times | No | No — fixed count |
| LLM-as-judge | Prompting another LLM to critique | Prompt engineering | No — vibes |
| Reflexion | LLM reflects on mistakes, stores in memory | Prompt + memory setup | No |
| Simple timeout | Killing the loop after N seconds | No | No |
| **judgment** | Statistical anomaly detection + Bayesian health inference | Yes — obs dict | Yes — CUSUM+HMM+POMDP |

**When to use judgment**: Long-running agent tasks (>10 steps), multi-tool workflows,
cost-sensitive deployments where wasted tokens = real money.
**When NOT to use it**: Simple single-call agents, already-short timeouts,
prototypes where you just want to see if it works.

## Production gotchas

See [`docs/production-gotchas.md`](docs/production-gotchas.md) — covers engine thread-safety,
CUSUM threshold tuning per domain, `progress_delta` calibration, warm-start behavior,
POMCP particle/simulation budget, content signal limitations, and monitoring checklist.

## Project scope

Judgment is a **critic** — it watches the execution loop and decides when
to intervene. It is not:

- A replacement for LangGraph / CrewAI / ReAct
- A semantic correctness judge — it detects structural degradation (tool
  failures, progress stalls, error cascades). Content signals (repetition,
  contradiction) are lightweight heuristics, not factuality checks.
  Semantic drift detection via embeddings is on the roadmap.
- A learning system — Baum-Welch learns HMM params from logs, but the
  POMDP reward function is configured manually, not learned.
- Pure math — corrective actions and content signals are explicitly
  labelled as heuristics. See `docs/limitations.md` for details.

## Roadmap

- [x] LangGraph, CrewAI, and custom adapters
- [x] Content-quality signals (repetition, contradiction detection)
- [x] FastPOMCP — online MCTS, 25ms per decision
- [x] Baum-Welch HMM parameter learning from run logs
- [x] Evaluation benchmark suite (5 fault models, 250 trajectories)
- [x] Benchmark: 61% waste reduction with 74.6% detection precision
- [ ] Reward learning from annotated trajectories
- [ ] Real benchmark integration (GAIA / SWE-bench / WebArena)
- [ ] Streaming observation model

## References

The math, for the curious:

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models." *Proc. IEEE*.
- Hawkes, A. G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*.
- Kaelbling et al. (1998). "Planning and Acting in POMDPs." *AIJ*.
- Silver & Veness (2010). "Monte-Carlo Planning in Large POMDPs." *NeurIPS*.

## License

MIT
