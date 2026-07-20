# T4 Runbook — GPU Strengthening Grid

For a human running the strengthening grid on an **NVIDIA T4** (Colab / Kaggle / cloud VM). Sessions there can die at any time; every step here is resumable. Follow in order. **Do not skip Phase 2** — it can abort the whole run before you waste GPU hours.

Envelope (NOT a promise; real ETA comes from Phase 2 calibration): 180 runs ≈ 3.2M epochs. At 3× the CPU baseline (11.5 ep/s) ≈ 27 h across several sessions; at 10× ≈ 8 h in one. Priority order means a short session still finishes P0 (the H1-power grid) first.

---

## Phase 1 — Clone & install (~5 min)

```bash
git clone https://github.com/Nafis878/AAAI.git && cd AAAI
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- **Expected:** a CUDA torch build and `True` and `Tesla T4`. On Colab/Kaggle the preinstalled CUDA torch is fine **if** `torch.cuda.is_available()` is True — do NOT `pip install -r requirements.txt` blindly there, because that file pins the CPU wheel version. If you need deps: `pip install numpy pandas matplotlib einops tqdm scipy tabulate` (leave the preinstalled torch alone).
- On a bare VM: install a CUDA torch build for the box's CUDA version from pytorch.org, then the deps above.
- Sanity: `python -m pytest -q` should pass (CPU-side logic; fast).

## Phase 2 — GPU sanity + parity + calibration (MANDATORY, ~15 min)

```bash
python -m src.run_week --sanity --device cuda
```
Runs three checks and writes `logs/parity_report.json` + `logs/calibration.json`:
1. **Smoke** — 300-epoch run on cuda; training loss must fall to <20% of its start.
2. **Parity** — replays the first 2,000 epochs of two committed CPU runs (`nope_s0`, `learned_s3`) on the GPU with identical seeds and compares val-accuracy curves. **Tolerance: mean |Δval_acc| ≤ 0.02 and max ≤ 0.05** over the shared eval grid. CPU↔GPU are NOT bitwise identical (float reduction order differs); curve-level agreement within tolerance is the criterion.
3. **Calibration** — measures aggregate epochs/s at workers ∈ {4,6,8}, A/B-tests `--compile` at 6 workers, prints the best config and the projected P0–P3 ETA.

**ABORT CRITERIA — if any fail, STOP and do not launch the grid:**
- Smoke loss does not fall → environment/torch problem.
- **Parity FAIL** (either run exceeds tolerance) → the GPU numerics diverge from the validated CPU behavior; the grid's statistics would not be comparable. Report `logs/parity_report.json` back and stop.
- Calibration shows aggregate ep/s below the CPU baseline (~11.5) → GPU is misconfigured (likely running on CPU); investigate before spending hours.

Pick `<best_workers>` and whether to add `--compile` from the calibration output. (Tiny models are kernel-launch-bound, so parallel workers usually beat `--compile`; trust the measured numbers.)

## Phase 3 — Launch the grid (resumable)

```bash
export GITHUB_TOKEN=...   # optional; enables 30-min auto-push of results. NEVER commit this.
nohup python -m src.run_week \
    --manifest research/week/grid_manifest.json \
    --device cuda --workers <best_workers> [--compile] \
    > logs/gpu_week.log 2>&1 &
```
- Runs P0→P3 in priority order into `experiments_gpu/`.
- **Resumable:** if the session dies, re-run the *exact same command*. Finished runs (with `checkpoints/ckpt_final.pt`) are skipped; a partially-trained run continues bit-exactly from its last `resume.pt` (saved every 1000 epochs). Nothing is lost beyond ≤1000 epochs of the in-flight runs.
- **Graceful stop:** `Ctrl-C` (or `kill -INT <pid>`) terminates workers cleanly and prints the resume command.

## Phase 4 — Monitor

```bash
python -m src.progress --launch-ts "$(date '+%Y-%m-%d %H:%M:%S')"   # per-run status, aggregate ep/s, ETA
tail -f logs/gpu_week.log                                            # orchestrator events
```
- **With `GITHUB_TOKEN` set:** the orchestrator pushes `experiments_gpu/` + `research/week/` + logs to `origin/main` every 30 min (`scripts/sync_results.sh`). Check the token has `repo`/`contents:write` scope.
- **Without a token:** nothing is pushed. When done (or before closing the session), run `bash scripts/pack_results.sh` and upload the resulting `gpu_week_results_*.zip` for the integration session.

## Phase 5 — On completion (automatic, then verify)

When the grid finishes, the orchestrator automatically runs, over the GPU data:
```
python -m src.analysis --all --exp-root experiments_gpu --out research/week/results
python -m src.figures  --all --exp-root experiments_gpu --out research/week/results
```
writing CSVs/JSON/figures to `research/week/results/`, then a final sync/push.

**"Done" looks like:** `logs/gpu_week.log` ends with `grid finished ... failures: none`; `research/week/results/` contains `h1_stats.json`, `h2_permutation_matched.json`, `h3_ood.json`, `table1_per_pe.*`, and `figures/`. If `failures` lists runs, just re-run the Phase-3 command — it retries only the failed/unfinished runs.

**Send back to the integration session:** the pushed commit (or the `pack_results.sh` zip). That session follows `paper/RESULTS_TODO.md` to fold the numbers into the manuscript.

## Timeline
Grid complete by **~July 24–25** leaves 3 days for the human prose rewrite + author-kit build before the **July 28 AoE** full-paper deadline. The **abstract is due July 21 AoE** and does not depend on these results — register on OpenReview and submit the title+abstract now (see `SUBMISSION_CHECKLIST.md`).

## Security
Never commit `GITHUB_TOKEN` or any credential. `sync_results.sh` reads it from the environment and injects it only into the push URL (never written to disk). If you accidentally expose a token, revoke it immediately on GitHub.
