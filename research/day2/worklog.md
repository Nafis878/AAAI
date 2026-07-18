# Day 2 worklog (timestamps local, 2026-07-18/19)

- **22:34** — Session start. Pre-flight: git clean @7984b16, pytest 15/15 green. Disk 7.2 GB free (97% full) → fp16 checkpoints mandatory.
- **22:35–22:45** — Checkpoint upgrade in `src/train.py`: ~35 log-spaced epochs (dense early), fp16 state_dicts under `experiments/<run>/checkpoints/`; `save_ckpt`/`load_ckpt`; early-stop epoch also checkpointed. `src/launch.py`: `--skip-existing` (idempotent relaunch keyed on `checkpoints/ckpt_final.pt`). `src/progress.py`: monitor (per-run status, aggregate ep/s vs 11 basis, disk). Tests: schedule density + fp16 round-trip → **17/17 green**. Dry-runs verified (core = 20 jobs).
- **22:36:41 — PASS A LAUNCHED** (detached Start-Process; `logs/passA.log`). 20 jobs = 5 PE × seeds 0–3, ≤15k epochs, 8 workers, `--skip-existing`. **Gate clocks: hour-2 pace gate at 00:37; hour-6 pivot gate at 04:37.**
- **22:39** — Verified: 8 workers alive, run dirs + log-spaced fp16 checkpoints appearing (nope/learned seeds 0–3 first). Disk 3.1 GB free (dropped 4 GB at launch — pagefile growth suspected; monitor armed at <1.5 GB threshold). Persistent monitor armed on passA.log (failures/completion/disk).
