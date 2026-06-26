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
│   ├── hmm.py             # 3-state discrete HMM + Forward/Viterbi
│   ├── cusum.py           # CUSUM anomaly detector
│   ├── pomdp.py           # Exact belief-MDP value iteration + RewardConfig
│   ├── corrective.py      # Heuristic corrective action router (4 rules)
│   ├── training.py        # Baum-Welch EM for HMM parameter learning
│   └── diagnostics.py     # Structured diagnostic outputs
├── harness/
│   ├── loop.py            # JudgmentHarness: unified execution loop
│   ├── executor.py        # LLMExecutor (OpenAI-compat) + SimulatedExecutor
│   └── tools.py           # ToolRegistry with built-in tools
├── cli/
│   └── main.py            # CLI: judgment run / train / dashboard
├── dashboard/
│   └── app.py             # Streamlit live visualizer
├── docs/
│   ├── architecture-redesign.md
│   ├── hawkes-redesign.md
│   └── three-gaps-design.md
├── tests/                 # 89 tests, all passing
├── pyproject.toml
└── README.md
```

## Math → Harness Problems

| Component | Math | Problem Solved |
|---|---|---|
| CUSUM + Hawkes | Page (1954) sequential detection + Hawkes (1971) | Detects observation drift; distinguishes noise from real anomalies; Hawkes corrects for expected event clustering |
| HMM Forward | Rabiner (1989) | Infers hidden health state (H/D/B) from noisy observations; each state has concrete operational meaning |
| POMDP Value Iteration | Kaelbling et al. (1998) | Optimal action selection under partial observability; exact solve on discretised belief simplex |
| Corrective Router | Heuristic (explicitly labelled) | Maps engine's CORRECT signal to concrete advice (verify/rethink/retry/rollback) |
| Baum-Welch (EM) | Rabiner (1989) §III-C | Learns HMM parameters from agent run logs; semi-supervised mode anchors state semantics |

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
from judgment import train_hmm
hmm = train_hmm(logs)  # logs: list of trajectories
```

## What This Is Not

- **Not a replacement for ReAct / LangGraph / CrewAI.** The engine is a *critic* that watches the execution loop and decides when to intervene. The harness loop (`JudgmentHarness`) wraps an LLM executor with math-driven oversight.
- **Not a learning system (yet).** Baum-Welch can learn HMM parameters from logs, but the POMDP reward function must be configured, not learned.
- **Not a content-quality judge.** The HMM cares about structural health signals (tool success rate, progress, error trend), not whether the LLM's output text is correct.

## Roadmap

- POMDP reward learning from annotated trajectories
- LangGraph adapter (drop-in decision node)
- Multi-agent health monitoring (one engine per agent, shared HMM)
- Streaming observation model (replace discrete categories with continuous soft signals)

## License

MIT

## References

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100–115.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models." *Proceedings of the IEEE*, 77(2), 257–286.
- Hawkes, A. G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting Point Processes." *Biometrika*, 58(1), 83–90.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). "Planning and Acting in Partially Observable Stochastic Domains." *Artificial Intelligence*, 101(1-2), 99–134.
