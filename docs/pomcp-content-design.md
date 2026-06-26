# POMCP + Content Signals — Design

## C: POMCP — Online Particle MCTS for POMDP

### Why

The current 231-grid exact value iteration only works for 3 states.
To support more states/observations, we need a method that scales without
grid discretisation.

### What

POMCP (Silver & Veness, 2010) combines:
- **Particle belief** — K unweighted state samples representing belief
- **UCT over histories** — MCTS with UCB1 at each action-observation history node
- **Generative model** — sample s' ~ T(·|s,a), o ~ O(·|s',a), r = R(s,a)

Runtime: N simulations × max_depth steps. Each simulation is O(log N) tree ops.
Default: N=1000, depth=10, K=100 → ~50ms per decision.

### Tree structure

Nodes are at the *history* level (action-observation sequences):

```
root(h=∅)  →  action a  →  observation o  →  child(h·a·o)  →  ...
```

Each node stores:
- `N(h)`: visit count
- `V(h)`: mean return from this history
- `B(h)`: set of particles (states) consistent with this history
- Per action a: `N(h,a)`, `Q(h,a)`, `children[o] = h·a·o`

### Algorithm

```
SEARCH(belief_particles):
    for i in 1..n_simulations:
        s ~ belief_particles      # sample initial state
        SIMULATE(s, root, depth=0)

    return argmax_a Q(root, a)

SIMULATE(s, h, depth):
    if depth >= max_depth or h is terminal:
        return 0

    if N(h) < n_visit_threshold:
        # New node — use rollout
        return ROLLOUT(s, depth)

    # UCB action selection
    a = argmax_a [ Q(h,a) + c * sqrt(log N(h)) / (1 + N(h,a)) ]

    # Step the generative model
    s' ~ T(· | s, a)
    o  ~ O(· | s')
    r  = R(s, a)

    # Descend or create child
    child = h.children.get((a, o))
    if child is None:
        child = new_node()
        h.children[(a, o)] = child

    total_r = r + gamma * SIMULATE(s', child, depth+1)

    # Backpropagate
    N(h) += 1; N(h, a) += 1
    Q(h, a) += (total_r - Q(h, a)) / N(h, a)

    return total_r

ROLLOUT(s, depth):
    # Simple random policy for the remainder
    for d in range(depth, max_depth):
        a ~ random action
        s' ~ T(·|s,a); o ~ O(·|s'); r = R(s,a)
        total += gamma^d * r
        s = s'
    return total
```

### Integration

POMCP runs online at each `engine.step()` call. It replaces the grid lookup
when `use_pomcp=True`. The grid solver remains as fallback.

Belief particles are sampled from the HMM's current filtered belief:
```
particles = [sample from P(S_t = s | o_{1:t}) for _ in range(K)]
```

## D: Content Quality Signals

### What

Three cheap, no-embedding signals extracted from the LLM's text output:

| Signal | Computation | Discretisation |
|---|---|---|
| Length z-score | (len(text) - μ_recent) / σ_recent | 3 bins: low / normal / high |
| Token novelty | |unique tokens| / |total tokens| | 3 bins: repetitive / normal / fresh |
| Negation surge | count(negation words) in current step | 2 bins: normal / elevated |

### How they feed the engine

These become 3 additional HMM observation dimensions (dim 4, 5, 6),
extending the current 4-dim model to 7 dimensions. The HMM emission
tables for content dimensions are:

```
Length anomaly:
             H      D      B
low          0.10   0.20   0.25   ← short responses in degraded states
normal       0.80   0.65   0.50
high         0.10   0.15   0.25

Token novelty:
             H      D      B
repetitive   0.05   0.25   0.40   ← looping = degraded
normal       0.85   0.65   0.45
fresh        0.10   0.10   0.15

Negation surge:
             H      D      B
normal       0.90   0.70   0.50
elevated     0.10   0.30   0.50   ← self-correction = concern
```

These are deliberately mild priors. They will be refined by Baum-Welch
from real logs.

### Observation encoding

The engine's `step()` method now accepts optional `llm_text: str`.
If provided, the ContentSignalExtractor computes the three signals
and they are appended to the HMM observation encoding.

## Implementation Plan

| File | Action |
|---|---|
| `core/pomcp.py` | NEW — POMCPPlanner class |
| `core/content_signals.py` | NEW — ContentSignalExtractor |
| `core/hmm.py` | MODIFY — add 3 content dims, encoding updated |
| `core/engine.py` | MODIFY — dual-mode POMDP, content signal switch |
| `integration/base.py` | MODIFY — extractor passes llm text |
| `tests/test_pomcp.py` | NEW |
| `tests/test_content_signals.py` | NEW |
