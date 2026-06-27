# Math Assumptions & Limitations

## What each layer assumes (and when it breaks)

### Layer 1: CUSUM + Hawkes

**Assumption**: Observations are conditionally stationary under the healthy model.
CUSUM detects when the observation stream departs from the "in-control" regime.

**Breaks when**: The baseline changes legitimately (e.g. switching from coding to
testing phase). In a real harness, different task phases have different normal
failure rates — CUSUM may alarm during a phase transition even though nothing is
wrong.

**Mitigation**: Increase the threshold `h` for noisy environments. Run `judgment`
with `use_content_signals=True` to get richer, less-trigger-happy observations.

### Layer 2: HMM (Hidden Markov Model)

**Assumption**: Agent health evolves as a first-order Markov process — the next
state depends only on the current state, not the full history.

**Breaks when**: Agent degradation is non-Markovian. For example, an error at
step 5 might cause a failure at step 20 via corrupted context — the Markov
assumption cannot model these long-range dependencies directly.

**Mitigation**: The Hawkes term partially compensates by encoding "how much
excitation is still in the system from past events." But full long-range
memory would require a higher-order HMM or RNN-based state estimator.

**Assumption**: Observation dimensions are conditionally independent given
the latent state (product model: P(o|s) = ∏ P(o_d|s)).

**Breaks when**: `tool_ok` and `progress_delta` are obviously correlated
(a failed tool almost always produces zero or negative progress). The product
model double-counts this evidence, making the HMM overconfident.

**Mitigation**: The continuous emission model (Gaussian/Poisson/Bernoulli)
partially addresses this by using denser likelihoods. A full treatment would
require a joint emission model or copula.

### Layer 3: POMDP / FastPOMCP

**Assumption**: The reward function accurately reflects the costs of
false positives (unnecessary escalation) vs false negatives (missing a real
failure). The default RewardConfig is derived from operational cost reasoning
but has not been calibrated on real user data.

**Breaks when**: The agent operates in a domain with very different costs
(e.g. medical decision support vs casual code generation). A missed failure
in medical is vastly more expensive than an unnecessary escalation.

**Mitigation**: Use `RewardConfig.preset("conservative")` for high-stakes
domains. Ideally, calibrate rewards from annotated trajectories.

**Assumption**: The belief simplex discretisation (grid solver) or particle
approximation (POMCP) is sufficient.

**Grid solver**: 231 points for 3 states. Add one state = exponential growth.
**POMCP**: 200 particles. Fewer particles = higher variance. More particles = more
cost. No convergence guarantees in the finite-sample regime.

### Content signals

**Assumption**: Length anomaly, token repetition, and negation surge are useful
proxies for LLM derailment.

**Breaks when**: The agent is deliberately repetitive (e.g. generating test
cases) or changes output length for legitimate reasons (switching from code
to documentation). Semantic drift — saying subtly wrong things that "look"
normal — is completely invisible to these signals.

**Mitigation**: Use content signals as a soft prior (they are deliberately
weak in the HMM emission tables). For semantic drift detection, an
embedding-based similarity metric is on the roadmap.

## Known false-positive sources

1. **Phase transitions**: Switching from coding → testing naturally increases
   tool failure rates. CUSUM may alarm. **Workaround**: bump CUSUM `h` to 5.0+.
2. **First-step prior**: The HMM starts with P(H)=0.65, biasing initial steps
   toward caution. **Workaround**: 2-3 warm-up steps before enabling judgment.
3. **Rapid action switching**: Fast cycling between continue/gather/correct
   near belief boundaries is prevented by hysteresis (0.08 margin), but the
   margin itself is a magic number.

## What's intentionally left out

- **Semantic factuality checking**: Embedding-based semantic drift is on the
  roadmap. Currently, structural health is the primary signal.
- **Multi-agent monitoring**: Each DecisionEngine instance is single-agent.
  Shared state across engines requires external coordination.
- **Online parameter adaptation**: Baum-Welch is batch. Online EM or particle
  learning is not implemented.
- **Guaranteed worst-case detection delay**: CUSUM has optimality properties
  under i.i.d. observations, but agent observation streams are not i.i.d.
- **Formal verification**: No proofs of correctness, only empirical benchmarks.

## References for the curious

- Tartakovsky, Nikiforov & Basseville (2014). *Sequential Analysis: Hypothesis
  Testing and Changepoint Detection*. CRC Press. — For CUSUM optimality theory.
- Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
  Ch. 17 — For HMM fundamentals and limitations of the Markov assumption.
- Ross, S. (2014). *Introduction to Probability Models*. Academic Press.
  Ch. 4 — For when Markov chains are and aren't appropriate.
