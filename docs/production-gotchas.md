# Production Gotchas

## Don't share engines across threads

`DecisionEngine` is stateful (`step_count`, `decision_log`, `prev_belief`, `prev_action`).
Share it across concurrent Agent runs and you'll get corrupted belief states.

```python
# Wrong:
engine = DecisionEngine()
for task in batch:
    run_agent(task, engine)  # engine state bleeds across tasks

# Right:
for task in batch:
    engine = DecisionEngine()
    run_agent(task, engine)
```

## CUSUM threshold h is your most important knob

Default `h=4.0` is tuned for synthetic 30-step tasks where false alarms are slightly
annoying but missing a failure is worse. Real harnesses vary:

| Domain | Suggested h | Why |
|---|---|---|
| Code generation (≤30 steps) | 4.0 | Failures are visible quickly |
| Long-running research (50+ steps) | 5.5 | Phase transitions are normal |
| Real-time / streaming | 3.0 | Every second matters |
| Batch / offline | 6.0 | False alarms waste no one's time |

Too low → false alarms annoy users. Too high → you miss real failures.
**Tune this per domain.** Run `scripts/trace_analyzer.py --input your_logs.jsonl`
on real data to find the right value.

## progress_delta is a fake signal unless you define it

The engine expects `progress_delta` as a float in roughly \[-0.3, 0.5\]. But "progress"
is task-specific. Give it real numbers:

```python
# Too vague:
engine.step({"tool_ok": True, "progress_delta": 0.1, ...})

# Better — anchored to your domain:
passed = run_tests()
engine.step({
    "tool_ok": passed,
    "progress_delta": 0.3 if passed else -0.1,  # tests = real progress
    "error_count_delta": 0 if passed else 1,
})
```

If you don't have a progress metric, **use a binary**:
- `+0.2` for a successful tool call that moves the task forward
- `0.0` for a tool call that returns information but no visible progress
- `-0.05` for a tool call that failed

Constant `0.1` every step means the engine learns nothing.

## The first 3 steps are noisy

The HMM starts with a flat prior: P(H)=0.65, P(D)=0.28, P(B)=0.07.
Engines see elevated gather/gambling actions in the first 2-3 steps as
the belief sharpens. This is normal. Either:

1. Run 2-3 warmup steps before enabling judgment, or
2. Ignore anomaly signals in the first 5 steps, or
3. Set `EngineConfig(theta_healthy=0.55)` to be less conservative initially

## POMCP particle count matters less than simulation count

`pomcp_n_particles=200` is fine for nearly all use cases. What matters is
`pomcp_n_simulations` — more simulations = less variance in Q-values.
Low-simulation POMCP (100-300) can produce inconsistent actions for the
same belief. Stick to 500+ for production, 1000 for safety-critical.

## Content signals are weak by design

The content signal emission tables are deliberately flat — P(anomalous text | Healthy)
is not zero because even a healthy LLM occasionally repeats itself or uses
negation words. This means content signals **alone will not detect failures**.
They sharpen detection precision by reducing false alarms, not by catching
new failures. The structural signals (tool_ok, progress_delta, error_count)
carry the detection burden.

If you need semantic detection, enable `[semantic]` extras and use
`SemanticDriftDetector` — it's heavier but catches derailment that
structural signals miss.

## Don't skip CUSUM

The ablation shows: HMM-only has 0% detection recall. Without CUSUM,
the HMM drifts slowly and never triggers an alarm. CUSUM is the canary.

## What to monitor in production

1. `belief["healthy"]` trend — if it's declining but no alarm fires, your CUSUM `h`
   is too high
2. `drift` (S_t) — if it's constantly near `h`, you're one step from alarm
3. `latency_ms` (from observability) — if FastPOMCP starts taking >50ms, check
   particle count and simulation budget
4. `solver` field — "threshold" means fast path is active (low-entropy belief).
   If you NEVER see "fast_pomcp", your agent is either perfect or your CUSUM is
   too sensitive.

## Engine throws an exception? Fall back to threshold.

The engine has three solver tiers — if FastPOMCP/grid/POMCP all fail,
the threshold gate is the safety net. Don't wrap `engine.step()` in a
try/except that kills the loop. Instead:

```python
try:
    decision = engine.step(obs)
except Exception:
    # Fallback: crude threshold check
    if error_count > 5:
        action = "escalate"
    else:
        action = "continue"
```
