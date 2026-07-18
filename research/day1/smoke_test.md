# Smoke Test & Compute Calibration (Day 1, 2026-07-18)

All numbers below were **measured on this machine** (Ryzen 7 5700G, 8C/16T, 13.9 GB RAM, torch 2.13.0+cpu, Python 3.12.4). Nothing is extrapolated from other hardware.

## 1. Smoke test: training works

Command:
```
.\.venv\Scripts\python.exe -m src.train --pe learned --seed 0 --epochs 500 --threads 4 --run-name smoke_learned_s0
```
Result (CSV: `experiments/smoke_learned_s0/log.csv`, committed):
- Epoch 0: train loss 4.911, val loss 4.811, val acc 0.95%
- Epoch 499: train loss **0.0063**, train acc **99.8%**, val loss 17.18, val acc **28.2%**
- **Training loss decreased monotonically to near-zero — the classic pre-grok memorization phase**, exactly as expected at 500 epochs (grokking literature: val-acc transition at ~10³–10⁴ epochs at frac 0.3, wd 1.0). Val acc already ≥ 28% (vs 0.9% chance) — early generalization signal present.
- Wall time: **128.8 s / 500 epochs = 25.8 s per 100 epochs** (4 threads, per-epoch eval).

## 2. Thread-scaling measurement (single run, 200 epochs, per-epoch eval)

| torch threads | s / 100 epochs | epochs/s |
|---|---|---|
| 1 | 51.4 | 1.95 |
| 2 | 33.7 | 2.97 |
| 4 | 25.8 | 3.88 |
| 8 | 21.3 | 4.71 |
| 16 | 19.4 | 5.15 |

Scaling saturates hard (2.6× speedup for 16×threads) — the model is too small to parallelize within a run. **Conclusion: many single-thread processes beat few multi-thread ones.**

## 3. Parallel-worker measurements (1 thread per worker, 200–400 epochs each)

| Config | Wall time | Aggregate throughput | Effective s/100/run |
|---|---|---|---|
| 8 workers, eval every epoch | 219.7 s / 8×200 ep | 7.28 ep/s | 13.7 |
| 12 workers, eval every epoch | **CRASHED** | — | — |
| **8 workers, eval every 5 epochs** | **267.0 s / 8×400 ep** | **11.99 ep/s** | **8.3** |

- **12 workers is infeasible:** 6/12 processes died with `DefaultCPUAllocator: not enough memory` (13.9 GB RAM ceiling). **Hard cap: 8 workers** (decisions.md D9).
- Val-set forward (8,938 examples) ≈ 40% of per-epoch cost; evaluating every 5 epochs raises throughput 1.65× with negligible loss of onset resolution (±5 epochs vs onsets ~10⁴). Adopted for all Day-2 grid runs (decisions.md D10).

## 4. Extrapolations (at the measured 12.0 ep/s aggregate; planning uses 11 ep/s for safety margin)

- **One full 40,000-epoch run:** solo/16-thread: 2.2 h. Inside the 8-worker pool: 40k ÷ (11/8) ep/s ≈ 8.1 h wall, but 8 run simultaneously (≈ 1.0 h of pool time each).
- **Original 25-run × 40k grid: 1.0M epochs ≈ 25 h — EXCEEDS the 12 h budget → cuts required and adopted** (proposal §4): epoch cap 15k first pass, seeds 4+2 first pass, val eval every 5 epochs.
- **Adopted Day-2 plan (worst case, 11 ep/s):**
  - Pass A core H1+H2: 5 PE × 4 seeds × ≤15k = 300k ep ≈ **7.6 h** (expected ~6.5 h with early stop; learned-APE baseline groks ~10–11k at these settings per Nanda et al.)
  - Pass B OOD H3: 5 PE × 2 seeds × ≤15k = 150k ep ≈ **3.8 h** (expected ~3.2 h)
  - **Worst case A+B = 11.4 h ≤ 12 h budget.** Slack from early stopping feeds Pass C (priority order: core-ext seed 4 → ood-ext → depth2 → p97 → ops), which may also run overnight into Day-3 morning (CPU is free while writing).
- Checkpoint I/O: ~0.9 MB × every 500 epochs × ≤35 runs ≈ ≤1 GB total — fine (gitignored).

## 5. Ready-to-run Day-2 launch commands

```powershell
# sanity first:
.\.venv\Scripts\python.exe -m pytest -q
# Pass A (blocks until done, ~6.5-7.6 h; run in background shell):
.\.venv\Scripts\python.exe -m src.launch --preset core
# Pass B (immediately after A):
.\.venv\Scripts\python.exe -m src.launch --preset ood
# Pass C presets, launch as time allows:
.\.venv\Scripts\python.exe -m src.launch --preset core-ext
.\.venv\Scripts\python.exe -m src.launch --preset ood-ext
.\.venv\Scripts\python.exe -m src.launch --preset depth2
.\.venv\Scripts\python.exe -m src.launch --preset p97
.\.venv\Scripts\python.exe -m src.launch --preset ops
```
`--dry-run` prints the job list for any preset without launching.
