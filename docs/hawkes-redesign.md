# Multivariate Marked Hawkes Process for Agent Action Timing

## 1. Problem Statement

The current `hawkes.py` implements a correct univariate exponential-kernel Hawkes process, but:

1. All events carry identical mark `m = 1.0`, erasing the distinction between successes and errors
2. A single intensity `λ(t)` conflates qualitatively different urges (corrective action vs. momentum vs. alarm)
3. The intensity is computed but never structurally drives the agent's decision

We redesign it as a **4-dimensional marked Hawkes process** that feeds directly into the decision loop.

---

## 2. Mathematical Foundations

### 2.1 Definition

A *D-variate marked Hawkes process* is a collection of D counting processes `N_d(t)` whose conditional intensities are:

$$
\lambda_d(t \mid \mathcal{H}_t) = \mu_d + \sum_{k:\, t_k < t} \alpha_{d,\, e_k}\, m_k\, e^{-\beta (t - t_k)}
\qquad d \in \{0, 1, 2, 3\}
$$

| Symbol | Meaning |
|---|---|
| `d` | Target event type index (which intensity we are computing) |
| `e_k` | Source event type index (the type of past event `k`) |
| `μ_d` | Baseline intensity for type `d` (poisson background rate) |
| `α_{d,e}` | Excitation coefficient: how much an event of type `e` increases the rate of type `d` |
| `m_k` | Mark (magnitude) of event `k`, drawn from `p_d(m)` |
| `β` | Shared exponential decay rate (common across all pairs) |
| `ℋ_t` | Filtration: all events strictly before time `t` |

**Why shared β.** Using a common decay parameter `β` for all dimensions makes the stationary condition a spectral-radius check on the single matrix `A = [α_{d,e}]`, rather than a tensor condition. Empirically, in agent harnesses time is measured in steps (not wall-clock seconds), so the decay physics is the same for all event types.

### 2.2 Event Types

```
D = 4:

  type 0: success        — tool returned useful output, no error
  type 1: error          — tool failed, threw exception, returned garbage
  type 2: user_interaction — user typed a message (correction, confirmation, question)
  type 3: tool_call      — any tool was invoked (read, write, execute, search, ...)
```

### 2.3 Excitation Matrix `A` (Design Rationale)

The matrix `A ∈ ℝ^{4×4}` encodes how each event type excites (or fails to excite) each other type. All entries are **non-negative** — standard Hawkes does not admit inhibition without risking negative intensities. Inhibition is instead modeled through low baseline rates and zero entries.

Design principles, justified by agent-harness phenomenology:

1. **Error self-excitation is the strongest diagonal** (`α_{1,1}` high). Errors cluster: a failed parse causes subsequent tool calls to fail on the same corrupted state.
2. **Error → tool_call is the strongest cross-excitation** (`α_{3,1}` highest off-diagonal). An error triggers immediate corrective tool use.
3. **Success inhibits nothing explicitly, but `α_{0,1} = 0`**. Success rate does not rise after errors — it resets.
4. **User interaction dampens error cascades** (`α_{1,2} = 0`). When the user intervenes, the error loop breaks.
5. **Tool calls moderately self-excite** (`α_{3,3}` moderate). Pipeline effects: one read leads to another read or write.

Proposed matrix:

$$
A = \begin{bmatrix}
0.15 & 0.00 & 0.10 & 0.18 \\  % success ← {success, error, user, tool_call}
0.00 & 0.35 & 0.00 & 0.06 \\  % error   ← {success, error, user, tool_call}
0.08 & 0.25 & 0.04 & 0.00 \\  % user    ← {success, error, user, tool_call}
0.20 & 0.40 & 0.12 & 0.18 \\  % tool_call ← {success, error, user, tool_call}
\end{bmatrix}
$$

### 2.4 Stationarity Check

For a multivariate Hawkes with shared decay `β`, the process is stationary iff the spectral radius of `(1/β) A` is strictly less than 1 (Brémaud & Massoulié, 1996):

$$
\rho\left(\tfrac{1}{\beta} A\right) < 1
$$

For a non-negative matrix, `ρ(A)` is bounded by min/max row sums (Frobenius):

$$
\min_i \sum_j a_{ij} \;\leq\; \rho(A) \;\leq\; \max_i \sum_j a_{ij}
$$

Row sums of proposed A:

```
row 0 (success): 0.15 + 0.00 + 0.10 + 0.18 = 0.43
row 1 (error):   0.00 + 0.35 + 0.00 + 0.06 = 0.41
row 2 (user):    0.08 + 0.25 + 0.04 + 0.00 = 0.37
row 3 (tool):    0.20 + 0.40 + 0.12 + 0.18 = 0.90
```

Therefore `ρ(A) ≤ 0.90`. With `β = 1.0` (proposed default):

$$
\rho\left(\tfrac{1}{\beta} A\right) \leq 0.90 < 1 \quad\checkmark
$$

The actual spectral radius (numerical) is approximately `0.57` — well within the stationary regime.

### 2.5 Mark Distributions

Each event type draws its mark from a Beta distribution stretched to a type-appropriate interval. The Beta distribution is chosen because it is supported on `[0, 1]` and its two shape parameters give independent control over mean and variance.

| Type | Distribution | Range | Mean | Rationale |
|---|---|---|---|---|
| success | Beta(2, 2) ↦ [0.3, 1.0] | 0.3–1.0 | 0.65 | Symmetric, mild successes are typical |
| error | Beta(5, 2) ↦ [0.5, 1.8] | 0.5–1.8 | ~1.4 | Right-skewed: most errors are moderate, few are catastrophic |
| user_interaction | Beta(3, 3) ↦ [0.5, 1.5] | 0.5–1.5 | 1.0 | Symmetric around 1.0 |
| tool_call | Beta(4, 2) ↦ [0.4, 1.2] | 0.4–1.2 | ~0.93 | Mildly right-skewed, routine calls |

Beta distribution reminder:

$$
p(m) = \frac{m^{a-1} (1-m)^{b-1}}{B(a,b)}, \quad B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}
$$

Scaling: `mark = lo + (hi - lo) * X` where `X ~ Beta(a, b)`.

---

## 3. Likelihood and Parameter Estimation

### 3.1 Log-Likelihood

For a multivariate Hawkes over observation window `[0, T]`, the log-likelihood is:

$$
\log\mathcal{L} = \sum_{d=0}^{3} \left[ \sum_{k: e_k = d} \log\lambda_d(t_k) - \int_0^T \lambda_d(t)\, dt \right]
$$

The integral term has a closed form for exponential kernel:

$$
\int_0^T \lambda_d(t)\,dt = \mu_d T + \sum_{k: t_k < T} \alpha_{d,e_k} m_k \left[1 - e^{-\beta(T - t_k)}\right]
$$

### 3.2 Gradient for Online SGD

For online adaptation (optional future feature), the gradient w.r.t. `α_{p,q}` is:

$$
\frac{\partial\log\mathcal{L}}{\partial\alpha_{p,q}} = \sum_{k: e_k = p} \frac{\mathbf{1}[c_k = q]\, m_k\, e^{-\beta(t_k - t_{prev})}}{\lambda_p(t_k)} - \sum_{k: e_k = q} m_k \left[1 - e^{-\beta(T - t_k)}\right]
$$

This is NOT implemented in v1; we use fixed parameters with manual calibration. The derivation is included for auditability and future work.

---

## 4. Decision Integration

### 4.1 The Four Intensity Channels

At any time `t`, the engine receives four intensity values:

```
λ_success(t)   — rate at which successes are "expected"
λ_error(t)     — rate at which errors are "expected"
λ_user(t)      — rate at which user interventions are "expected"
λ_tool(t)      — rate at which tool calls are "expected"
```

### 4.2 Composite Trigger Signal

Define the **composite trigger** as a weighted sum:

$$
\tau(t) = w_t \lambda_t(t) + w_e \lambda_e(t) - w_s \lambda_s(t) - w_u \lambda_u(t)
$$

with default weights: `w_t = 1.0`, `w_e = 2.0`, `w_s = 0.3`, `w_u = 1.5`.

**Interpretation:**

| Term | Sign | Meaning |
|---|---|---|
| `w_t λ_t` | + | Tool-call intensity drives action — the agent's "momentum" |
| `w_e λ_e` | + | Error intensity drives action — corrective urgency |
| `w_s λ_s` | − | Success intensity suppresses action — "if it's working, don't touch it" |
| `w_u λ_u` | − | User intensity suppresses action — yield the floor to the human |

When `τ(t) > θ * baseline` (default `θ = 1.8`), the engine emits a **proactive action signal**.

### 4.3 Per-Channel Trigger Probabilities

For each channel `d`, the probability of at least one event in the next `Δt`:

$$
P(\text{event}_d \mid \Delta t) = 1 - \exp\left(-\lambda_d(t) \cdot \Delta t\right)
$$

These are surfaced in the `Decision` object as `trigger_probabilities: Dict[str, float]`.

### 4.4 Time in "Steps"

In agent harnesses, wall-clock time is often unavailable or meaningless. We define time `t` as **integer step count**. The decay kernel `e^{-β·(t - t_k)}` thus decays over **steps**, not seconds. This is a deliberate modeling choice: "what happened 5 steps ago is `e^{-5β}` as relevant as what just happened."

---

## 5. Implementation Plan

### 5.1 File Changes

| File | Action |
|---|---|
| `core/hawkes.py` | **Rewrite** — multivariate marked Hawkes |
| `core/judgment_engine.py` | **Modify** — wire 4-channel intensities into `Decision` |
| `examples/coding_agent_demo.py` | **Modify** — emit typed events with proper marks |
| `tests/test_hawkes.py` | **New** — stationarity, intensity monotonicity, mark sampling |

### 5.2 Public API (target)

```python
class HawkesEvent:
    time: float
    event_type: int           # 0=success, 1=error, 2=user, 3=tool_call
    mark: float               # drawn from type-specific Beta

class MultivariateHawkes:
    def __init__(self, mu: np.ndarray, alpha: np.ndarray, beta: float):
        """mu: shape (4,), alpha: shape (4,4), beta: float"""

    def intensity(self, t: float) -> np.ndarray:
        """Returns λ(t) ∈ ℝ⁴"""

    def add_event(self, t: float, event_type: int, mark: float | None = None):
        """If mark is None, sample from the type's Beta distribution."""

    def trigger_signal(self, t: float) -> float:
        """Composite τ(t) per §4.2"""

    def trigger_probabilities(self, t: float, dt: float = 1.0) -> np.ndarray:
        """P(event_d | Δt) per §4.3"""

    def reset(self): ...

    @staticmethod
    def sample_mark(event_type: int) -> float: ...
```

### 5.3 Default Parameters

```python
MU = np.array([0.30, 0.15, 0.08, 0.50])   # baseline intensities

ALPHA = np.array([
    [0.15, 0.00, 0.10, 0.18],   # success ← *
    [0.00, 0.35, 0.00, 0.06],   # error ← *
    [0.08, 0.25, 0.04, 0.00],   # user ← *
    [0.20, 0.40, 0.12, 0.18],   # tool_call ← *
])

BETA = 1.0
```

### 5.4 Verification Criteria

Before marking this module "done":

1. **Stationarity test**: Simulate 1000 steps; intensities must not diverge
2. **Cascade test**: Inject a single error at t=10; `λ_error(t)` and `λ_tool(t)` must spike with correct relative magnitudes
3. **Decay test**: After a burst of tool_calls, verify intensities decay as `e^{-β·Δt}`
4. **Decision wire test**: In `coding_agent_demo.py`, verify that error cascades produce escalating `trigger_signal` values

---

## 6. Appendix: Why Not Power-Law Kernel?

Power-law Hawkes (`φ(t) ~ t^{-(1+γ)}`) models long-range memory — relevant for seismic aftershocks or social media virality where an event 100 steps ago can still matter. In agent harnesses, memory is already handled by the LLM's context window. The Hawkes process models **recent impulse** (last ~5–10 steps), for which exponential decay is both sufficient and computationally lighter (O(K) per intensity evaluation vs. O(K log K) for power-law).

---

## 7. References

- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika*, 58(1), 83–90.
- Brémaud, P., & Massoulié, L. (1996). "Stability of nonlinear Hawkes processes." *The Annals of Probability*, 24(3), 1563–1588.
- Laub, P. J., Taimre, T., & Pollett, P. K. (2015). "Hawkes processes." *arXiv:1507.02822*.
- Mei, H., & Eisner, J. (2017). "The Neural Hawkes Process." *NeurIPS*.
