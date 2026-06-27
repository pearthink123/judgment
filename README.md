# judgment

**Math-driven Agent Harness — CUSUM + HMM + POMDP decision engine with built-in execution loop.**

Instead of relying on prompt heuristics, this module uses **sequential change-point detection, discrete Hidden Markov Models, and exact POMDP value iteration** to make quantifiable, auditable decisions about:

- Whether the agent is healthy, degraded, or broken (latent state inference)
- When to continue, correct, escalate, or gather information (optimal action under uncertainty)
- Whether the current observation stream has drifted from normal (anomaly detection)

This targets the core engineering challenge: knowing when an LLM agent is going off the rails *before* it wastes context or produces bad output.

## Architecture

```
┌─────────────────────────────┐
│ Layer 1: CUSUM + Hawkes     │  Page (1954) change-point detection
│   Anomaly detection         │  Hawkes (1971) baseline likelihood
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Layer 2: 3-State HMM        │  Rabiner (1989) Forward algorithm
│   Healthy/Degraded/Broken   │  Log-space filtering
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Layer 3: POMDP Policy       │  Kaelbling et al. (1998)
│   Exact value iteration     │  Belief-MDP on 231-point simplex
│   → continue/correct/       │
│     escalate/gather         │
└─────────────────────────────┘
```

Each layer has a citable mathematical foundation. No heuristic masquerading as math.

## Project Structure

```
judgment/
├── core/
│   ├── engine.py          # DecisionEngine: 3-layer integration
│   ├── hawkes.py          # Multivariate marked Hawkes (likelihood provider)
│   ├── hmm.py             # 3-state discrete HMM + Forward/Viterbi (7 obs dims)
│   ├── cusum.py           # CUSUM anomaly detector
│   ├── pomdp.py           # Exact belief-MDP value iteration + RewardConfig
│   ├── pomcp.py           # POMCP — online particle MCTS (no grid, scalable)
│   ├── content_signals.py # Content-quality metrics (length, novelty, negation)
│   ├── corrective.py      # Heuristic corrective action router (4 rules)
│   ├── training.py        # Baum-Welch EM for HMM parameter learning
│   └── diagnostics.py     # Structured diagnostic outputs
├── integration/
│   ├── base.py            # Abstract adapter protocol
│   └── langgraph.py       # LangGraph node + conditional-edge router
├── harness/
│   ├── loop.py            # JudgmentHarness: unified execution loop
│   ├── executor.py        # LLMExecutor (OpenAI-compat) + SimulatedExecutor
│   └── tools.py           # ToolRegistry with built-in tools
├── examples/
│   ├── coding_agent_demo.py
│   └── langgraph_agent.py # LangGraph-style agent with judgment oversight
├── cli/
│   └── main.py            # CLI: judgment run / train / dashboard
├── dashboard/
│   └── app.py             # Streamlit live visualizer
├── docs/
│   ├── architecture-redesign.md
│   ├── hawkes-redesign.md
│   └── three-gaps-design.md
├── tests/                 # 139 tests, all passing
├── scripts/
│   ├── benchmark.py        # Detection performance benchmark
│   ├── eval_runner.py      # Head-to-head: baseline vs judgment
│   └── fault_models.py     # 4 realistic degradation patterns
├── pyproject.toml
└── README.md
```

## Math → Harness Problems

| Component | Math | Problem Solved |
|---|---|---|
| CUSUM + Hawkes | Page (1954) + Hawkes (1971) | Detects observation drift; Hawkes corrects for expected event clustering |
| HMM Forward | Rabiner (1989) | Infers hidden health state (H/D/B) from noisy structural + content signals |
| POMCP (online MCTS) | Silver & Veness (2010) | Scalable online POMDP solving — particle-based UCT, no grid discretisation |
| Grid POMDP | Kaelbling et al. (1998) | Exact value iteration on 231-point simplex (fast fallback for 3-state case) |
| Content Signals | Heuristic (lightweight) | Detects LLM derailment from text output: length anomaly, repetition, self-contradiction |
| Corrective Router | Heuristic (explicitly labelled) | Maps CORRECT signal to concrete advice (verify/rethink/retry/rollback) |
| Baum-Welch (EM) | Rabiner (1989) §III-C | Learns HMM parameters from agent run logs; semi-supervised mode |

## Quick Start

```bash
git clone https://github.com/pearthink123/judgment.git
cd judgment
pip install -e ".[dashboard]"   # base + Streamlit + pandas + matplotlib

# Run demo (no API key needed)
python examples/coding_agent_demo.py

# Launch dashboard
streamlit run dashboard/app.py

# Run CLI
judgment run "Implement an LRU cache in Python" --max-steps 10
```

## Using as a Library

```python
from judgment import JudgmentHarness

# One line to create a harness
harness = JudgmentHarness(max_steps=30, seed=42)

# Run a task
result = harness.run("Write a function that merges two sorted lists.")

print(f"Status: {result.status}")
print(f"Steps:  {result.steps}")
print(f"Belief: {result.final_belief}")
print(f"Summary: {result.summary}")

# Inspect the full decision log
for d in result.decision_log:
    print(f"  {d.action:10s}  H={d.belief['healthy']:.3f}  "
          f"D={d.belief['degraded']:.3f}  B={d.belief['broken']:.3f}")
```

Power users can access individual layers directly:

```python
from judgment import DecisionEngine, RewardConfig, train_hmm

# Custom reward function
reward = RewardConfig.preset("conservative")
engine = DecisionEngine(reward=reward)

# Process observations step by step
decision = engine.step({
    "tool_ok": True,
    "progress_delta": 0.15,
    "has_user_msg": False,
    "error_count_delta": 0,
})
print(decision.action)  # "continue"

# Learn from logs
hmm = train_hmm(logs)

# POMCP mode — scalable online MCTS (no grid)
engine = DecisionEngine(use_pomcp=True, pomcp_n_simulations=2000)

# Content-quality monitoring
engine = DecisionEngine(use_content_signals=True)
decision = engine.step({
    "tool_ok": True,
    "progress_delta": 0.12,
    "has_user_msg": False,
    "error_count_delta": 0,
    "llm_text": "The agent's output text for this step...",
})
# decision.content_signals → length_z_cat, novelty_cat, negation_cat
```

## LangGraph Integration

Drop judgment oversight into an existing LangGraph agent without restructuring your graph:

```python
from judgment.integration.langgraph import (
    create_judgment_node, create_judgment_router,
)
from judgment import DecisionEngine

engine = DecisionEngine()

graph = StateGraph(MyState)
graph.add_node("agent", my_agent_node)
graph.add_node("tools", my_tool_node)
graph.add_node("human", my_human_node)
graph.add_node("judgment", create_judgment_node(engine))

# After each tool execution, check health
graph.add_edge("tools", "judgment")

# judgment routes back to agent, tools, or human based on engine output
graph.add_conditional_edges(
    "judgment",
    create_judgment_router(engine),
    {"agent": "agent", "tools": "tools", "human": "human"},
)
```

See `examples/langgraph_agent.py` for a complete runnable example (no LangGraph install required).

## Benchmarks

Head-to-head evaluation on 250 synthetic trajectories across 5 fault models (context drift, tool degradation, loop trap, catastrophic cascade, healthy baseline). Full results: `scripts/eval_results.json`.

| Metric | Baseline | Judgment | Meaning |
|---|---|---|---|
| **Waste ratio** (failure tasks) | 100% | **39%** | Judgment stops broken runs 61% earlier |
| Catastrophic cascade recall | — | **100%** | Instant detection (1.1 step delay) |
| Detection precision | — | **74.6%** | 3/4 alarms are real faults |
| False escalation rate | — | 8.0% | Healthy runs incorrectly interrupted |
| Mean progress (all tasks) | 0.60 | **0.77** | +28% improvement |

**Key takeaway**: judgment's purpose is not to increase success rate — it's to *stop wasting steps on doomed trajectories*. On that metric it delivers a 61% reduction. The 8% false escalation rate can be tuned by adjusting the CUSUM threshold `h`.

```bash
# Run the evaluation yourself
python scripts/eval_runner.py --trajectories-per-model 25 --max-steps 30

# Compare grid vs POMCP solver
python scripts/benchmark.py --trajectories 100 --compare
```

## What This Is Not

- **Not a replacement for ReAct / LangGraph / CrewAI.** The engine is a *critic* that watches the execution loop and decides when to intervene. The harness loop (`JudgmentHarness`) wraps an LLM executor with math-driven oversight.
- **Not a learning system (yet).** Baum-Welch can learn HMM parameters from logs, but the POMDP reward function must be configured, not learned.
- **Not a content-quality judge.** The HMM includes structural health signals and lightweight content metrics (length anomaly, repetition), but it does not evaluate the semantic correctness or factuality of LLM outputs.

## Roadmap

- [x] LangGraph adapter (drop-in node + conditional-edge router)
- [x] Content-quality signals (length, novelty, negation — no embedding deps)
- [x] POMCP online MCTS (scalable alternative to grid value iteration)
- [x] Baum-Welch HMM parameter learning from agent run logs
- [x] Evaluation benchmark suite (5 fault models, head-to-head vs baseline)
- [ ] CrewAI / AutoGen adapters
- [ ] POMDP reward learning from annotated trajectories
- [ ] Multi-agent health monitoring (one engine per agent, shared HMM)
- [ ] Real benchmark integration (GAIA / SWE-bench / WebArena)
- [ ] Streaming observation model (continuous, not discrete categories)

## License

MIT

## References

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100–115.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models." *Proceedings of the IEEE*, 77(2), 257–286.
- Hawkes, A. G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*, 58(1), 83–90.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). "Planning and Acting in Partially Observable Stochastic Domains." *Artificial Intelligence*, 101(1-2), 99–134.
- Silver, D. & Veness, J. (2010). "Monte-Carlo Planning in Large POMDPs." *NeurIPS*, 23.
