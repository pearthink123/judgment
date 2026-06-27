"""Compare POMCP (old) vs FastPOMCP (new) — single-decision latency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import statistics
import numpy as np
from core.pomcp import POMCPPlanner
from core.pomcp_fast import FastPOMCPPlanner

beliefs = [
    np.array([0.80, 0.15, 0.05]),  # mostly healthy
    np.array([0.03, 0.40, 0.57]),  # broken
    np.array([0.10, 0.85, 0.05]),  # degraded
]

configs = [
    (500, "500 sims"),
    (1000, "1000 sims"),
    (2000, "2000 sims"),
]

print("=" * 75)
print(f"{'Config':<20} {'Old POMCP':>14} {'FastPOMCP':>14} {'Speedup':>10}")
print("-" * 75)

for n_sims, label in configs:
    old_times = []
    fast_times = []

    for belief in beliefs:
        # Old POMCP
        old_p = POMCPPlanner(n_simulations=n_sims, n_particles=200, rng=np.random.default_rng(42))
        for _ in range(3):
            old_p.reset()
            t0 = time.perf_counter()
            old_p.search(belief)
            old_times.append(time.perf_counter() - t0)

        # Fast POMCP
        fast_p = FastPOMCPPlanner(
            n_simulations=n_sims, n_particles=200,
            early_stop_margin=0.5,
            early_stop_stability=3,
            rng=np.random.default_rng(42),
        )
        for _ in range(3):
            fast_p.reset()
            t0 = time.perf_counter()
            fast_p.search(belief)
            fast_times.append(time.perf_counter() - t0)

    old_mean = statistics.mean(old_times) * 1000
    fast_mean = statistics.mean(fast_times) * 1000
    speedup = old_mean / fast_mean if fast_mean > 0 else float("inf")

    print(f"{label:<20} {old_mean:>12.1f}ms {fast_mean:>12.1f}ms {speedup:>9.1f}x")

    # Verify actions agree
    old_p2 = POMCPPlanner(n_simulations=n_sims, rng=np.random.default_rng(42))
    fast_p2 = FastPOMCPPlanner(n_simulations=n_sims, rng=np.random.default_rng(42))
    agreements = 0
    for b in beliefs:
        old_a = old_p2.search(b)
        old_p2.reset()
        fast_a = fast_p2.search(b)
        fast_p2.reset()
        if old_a == fast_a:
            agreements += 1
    print(f"{'  action agreement':<20} {agreements:>12}/{len(beliefs)}")
    print()
