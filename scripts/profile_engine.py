"""Full engine.step() hot-path profile."""
import sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import DecisionEngine
from core.hawkes import HawkesProcess
from core.cusum import CUSUMDetector
from core.hmm import encode_observation, HiddenMarkovModel
import numpy as np

def fmt_us(x): return f"{x*1e6:.0f}us"
def fmt_ms(x): return f"{x*1e3:.1f}ms"

# 1. Hawkes at current load
engine = DecisionEngine(seed=42)
for _ in range(20):
    engine.step({"tool_ok": True, "progress_delta": 0.12, "has_user_msg": False, "error_count_delta": 0})

hp = engine.hawkes
n = len(hp.events)
times = [time.perf_counter() for _ in range(100)]
for i in range(100):
    hp.intensity(float(n))
times = [times[i+1] - times[i] for i in range(len(times)-1)]
t_hawkes_now = statistics.mean(times)
print(f"Hawkes intensity() at {n} events: {fmt_us(t_hawkes_now)}")

# 2. Hawkes at 400 events
hp2 = HawkesProcess()
for i in range(200):
    hp2.add_event(float(i), 0, mark=0.5)
    hp2.add_event(float(i), 3, mark=0.7)

times2 = []
for _ in range(100):
    t0 = time.perf_counter()
    hp2.intensity(200.0)
    times2.append(time.perf_counter() - t0)
t_hawkes_400 = statistics.mean(times2)
print(f"Hawkes intensity() at 400 events: {fmt_us(t_hawkes_400)}")

# 3. CUSUM
c = CUSUMDetector()
times3 = []
for _ in range(200):
    t0 = time.perf_counter()
    c.update(surprisal_healthy=1.5, hawkes_intensity=1.0, surprisal_degraded=2.0)
    times3.append(time.perf_counter() - t0)
t_cusum = statistics.mean(times3)
print(f"CUSUM update:                    {fmt_us(t_cusum)}")

# 4. HMM forward
hmm = HiddenMarkovModel()
obs_cats = encode_observation(True, 0.15, False, 0)
times4 = []
for _ in range(200):
    hmm.reset()
    t0 = time.perf_counter()
    hmm.forward_step(obs_cats)
    times4.append(time.perf_counter() - t0)
t_hmm = statistics.mean(times4)
print(f"HMM forward_step:                {fmt_us(t_hmm)}")

# 5. Engine threshold mode
engine2 = DecisionEngine(use_fast_pomcp=False, use_pomdp=False, seed=42)
for _ in range(10):
    engine2.step({"tool_ok": True, "progress_delta": 0.12, "has_user_msg": False, "error_count_delta": 0})
times5 = []
for _ in range(50):
    t0 = time.perf_counter()
    engine2.step({"tool_ok": True, "progress_delta": 0.15, "has_user_msg": False, "error_count_delta": 0})
    times5.append(time.perf_counter() - t0)
t_engine_thresh = statistics.mean(times5)
print(f"Engine.step() threshold:          {fmt_ms(t_engine_thresh)}")

# 6. Engine FastPOMCP
engine3 = DecisionEngine(use_fast_pomcp=True, seed=42)
for _ in range(10):
    engine3.step({"tool_ok": True, "progress_delta": 0.12, "has_user_msg": False, "error_count_delta": 0})
times6 = []
for _ in range(30):
    t0 = time.perf_counter()
    engine3.step({"tool_ok": True, "progress_delta": 0.15, "has_user_msg": False, "error_count_delta": 0})
    times6.append(time.perf_counter() - t0)
t_engine_fastpomcp = statistics.mean(times6)
print(f"Engine.step() FastPOMCP:           {fmt_ms(t_engine_fastpomcp)}")

print()
print("=== Full step composition (threshold mode) ===")
overhead = t_engine_thresh - t_hawkes_now - t_cusum - t_hmm
print(f"Hawkes:   {fmt_us(t_hawkes_now)} ({t_hawkes_now/t_engine_thresh*100:.0f}%)")
print(f"CUSUM:    {fmt_us(t_cusum)} ({t_cusum/t_engine_thresh*100:.0f}%)")
print(f"HMM:      {fmt_us(t_hmm)} ({t_hmm/t_engine_thresh*100:.0f}%)")
print(f"Overhead: {fmt_us(overhead)} ({abs(overhead)/t_engine_thresh*100:.0f}%)")
print(f"TOTAL:    {fmt_ms(t_engine_thresh)}")
print()
print(f"FastPOMCP adds: {fmt_ms(t_engine_fastpomcp - t_engine_thresh)}")

