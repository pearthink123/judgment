"""Quick profile: where does POMCP spend time?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from core.pomcp import POMCPPlanner

planner = POMCPPlanner(n_simulations=500, n_particles=100, rng=np.random.default_rng(42))
belief = np.array([0.6, 0.3, 0.1])

# Warm up
for _ in range(3):
    planner.search(belief)
    planner.reset()

# Timed runs
times = []
for _ in range(10):
    planner.reset()
    t0 = time.perf_counter()
    action = planner.search(belief)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)

import statistics
print(f"POMCP 500 sims × 10 runs:")
print(f"  mean:   {statistics.mean(times)*1000:.1f} ms")
print(f"  median: {statistics.median(times)*1000:.1f} ms")
print(f"  min:    {min(times)*1000:.1f} ms")
print(f"  max:    {max(times)*1000:.1f} ms")
print(f"  per-sim: {statistics.mean(times)*1000/500:.2f} ms/simulation")
