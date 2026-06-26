# math_harness_judgment

**A mathematically rigorous Judgment Engine for Production AI Agent Harnesses.**

Instead of relying on brittle prompt heuristics or ad-hoc ReAct loops, this module uses **stochastic processes, Bayesian inference, information theory, and control theory** to make quantifiable, auditable decisions about:

- When the agent should act (proactive timing)
- What is the hidden state of the task (beliefs over success, error risk, stuck probability)
- Which action has the highest expected value right now (avoid context rot and low-value tool calls)
- How aggressively to behave and how to correct when things go wrong

This directly addresses the core engineering challenges in building reliable long-horizon Agents (the "Harness" layer on top of raw LLMs).

## Why this exists

DeepSeek Harness (and similar efforts at Anthropic, OpenAI, Cursor, etc.) are trying to turn powerful models into dependable systems that can execute 20-100+ step workflows with high success rate.

Common failure modes:
- Context rot from noisy tool outputs
- Compound error explosion (99% per step → ~60% after 50 steps)
- Unstable timing (when to call tools vs verify vs ask human)
- Lack of principled exploration/exploitation and error recovery

Most open-source Agent projects attack this with more prompt engineering, better memory, or graph orchestration.

This project attacks it with **applied math** that has proven itself in high-stakes domains (quant trading, control systems, operations research).

## Math → Harness Pain Points

| Math Primitive              | Harness Problem it Solves                          | Quantified Benefit                     |
|-----------------------------|----------------------------------------------------|----------------------------------------|
| Hawkes / Poisson Process    | Decision timing & proactive triggering             | Dynamic urge instead of fixed polling or LLM whim |
| Bayesian State Estimation   | Hidden state inference (progress, risk, stuck)     | Actual probability distributions instead of vibes |
| Expected Value of Information (EVOI) | Action selection & context bloat control     | Only call tools when the expected information/reward justifies it |
| PID + Stochastic Control    | Behavior regulation, recovery, exploration bias    | Automatic gain scheduling based on real error signals |
| Approximate POMDP value     | Long-horizon compound errors                       | Looks ahead a few steps instead of greedy |

This is particularly resonant with Tianyi Cui's background (Jane Street quant → AI infra).

## Project Structure

```
math_harness_judgment/
├── core/
│   ├── judgment_engine.py      # The main plug-in class
│   ├── hawkes.py               # Self-exciting point process
│   ├── bayesian.py             # Belief state (task success, error risk, etc.)
│   ├── info_gain.py            # EVOI action valuation
│   ├── control.py              # PID + stochastic controller
│   ├── mdp_pomdp.py            # Lightweight multi-step value estimator
│   └── ...
├── harness_integration/
│   └── decision_loop.py        # Example math-augmented ReAct-style loop
├── examples/
│   └── coding_agent_demo.py    # Runnable demo (simulated coding task)
├── dashboard/
│   └── app.py                  # Streamlit visualizer
├── README.md
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/YOURNAME/math_harness_judgment.git
cd math_harness_judgment
pip install -r requirements.txt

# Run the demo
python examples/coding_agent_demo.py

# Launch the dashboard
streamlit run dashboard/app.py
```

## Using the Judgment Engine

```python
from core.judgment_engine import JudgmentEngine

engine = JudgmentEngine(seed=42)

for turn in range(20):
    # Your harness gathers real observation
    observation = {
        "tool_success": True,
        "progress_delta": 0.12,
        "error_count_delta": 0,
        "steps_taken": turn,
        # ... more signals from your execution environment
    }

    decision = engine.decide(observation)

    print(f"Action: {decision.action} (conf={decision.confidence:.2f})")
    print(f"  Trigger: {decision.trigger_intensity:.2f}  EVOI: {decision.evoi:.2f}")
    print(f"  Belief: {decision.belief}")
    print(f"  Rationale: {decision.rationale}")

    # Execute the chosen action in your real harness/tooling
    outcome = your_harness.execute(decision.action)

    engine.record_outcome(decision.action, outcome)
```

The `Decision` object carries rich diagnostics that you can log, surface to humans, or feed back into model training.

## Positioning for DeepSeek Harness roles

If you're looking at roles on the Harness / Agent Engineering side (especially anything involving reliability, scheduling, control plane, or multi-agent coordination), this project demonstrates:

- Deep understanding of the actual hard problems (not just "I built another LangChain wrapper")
- Ability to bring rigorous quantitative methods from other domains (quant finance, control theory)
- Production thinking: plug-in architecture, diagnostics, separation of judgment vs execution
- End-to-end runnable artifact with visualization (rare in pure researchy math projects)

## Roadmap / Extensions

- Real POMDP / MCTS rollouts for better long-horizon planning
- Learnable parameters (α, β of Hawkes, PID gains) from failure trajectories
- Integration adapters for LangGraph, LlamaIndex workflows, or custom DeepSeek harness
- Multi-agent game-theoretic coordination layer
- Uncertainty-aware tool schema selection

## License

MIT

## Contact / Credit

Built as a focused demonstration of applying mathematical decision theory to Agent Harness engineering.
