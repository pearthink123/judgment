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
│ + Hawkes baseline │  Catches drift in the observation stream.
└────────┬──────────┘
         ▼
┌───────────────────┐
│ Layer 2: HMM      │  "How healthy is the agent right now?"
│ Healthy/Degraded/ │  Bayesian inference from noisy signals.
│ Broken            │
└────────┬──────────┘
         ▼
┌───────────────────┐
│ Layer 3: POMDP    │  "Continue, correct, escalate, or gather info?"
│ (FastPOMCP, 25ms) │  Optimal action under uncertainty.
└───────────────────┘
```

Each layer uses math with a real paper behind it — but you don't need to
read those papers to use the library.

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

## Project scope

Judgment is a **critic** — it watches the execution loop and decides when
to intervene. It is not:

- A replacement for LangGraph / CrewAI / ReAct
- A content-quality judge (it checks structural health, not correctness)
- A learning system (POMDP reward function is configured, not learned)

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
