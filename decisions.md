# decisions.md — append-only log of judgment calls

Format: date · decision · rationale. Newest at the bottom.

## 2026-07-18 (Day 1)

**D1. Python 3.12 venv instead of system Python 3.10 or uv.**
`uv` is not installed; rather than adding new tooling, used stdlib `venv` with the newest available interpreter (`py -3.12` → 3.12.4). 3.12 has better CPython performance than 3.10, which matters on a CPU-only training budget, and current torch CPU wheels support it.

**D2. Torch CPU wheel from the official CPU index.**
No CUDA GPU present (`nvidia-smi` absent; Radeon iGPU has no usable PyTorch backend on Windows worth the setup risk). `--index-url https://download.pytorch.org/whl/cpu` avoids pulling CUDA libraries (~2.5 GB saved) on a machine that cannot use them.

**D3. Git operations via Git Bash only.**
Git 2.49 is present in Git Bash's PATH but not PowerShell's. Standardized on the Bash tool for all git commands instead of editing the system PATH (no system-level changes without need).

**D4. `data/` is gitignored; datasets regenerated deterministically.**
Mod-p datasets are tiny and fully determined by `(p, op, frac, seed)` via `src/data.py`, so committing them adds risk (stale artifacts) without reproducibility benefit. CSV metric logs under `experiments/` ARE committed — they are the experimental record.

**D5. Scoop-check verdict: NOT SCOOPED — default top pick stands.**
Six targeted web searches + three abstract fetches (2026-07-18) found no paper using PE scheme as the controlled variable for grokking dynamics or circuit identity on modular arithmetic. Closest: Zhong et al. 2306.17844 and the NeurIPS 2025 unification paper (both control attention type, not PE; the latter has no dynamics axis). Full table in `research/day1/landscape.md` §2.

**D6. H2 reframed from binary clock/pizza classification to continuous circuit signatures.**
The NeurIPS 2025 unification paper (Moisescu-Pareja et al.) argues Clock and Pizza are parameterizations of one abstract algorithm. Betting the paper on a hard binary would be fragile. H2 therefore measures continuous signatures (Fourier-power concentration, attention-dependence of logits, restricted/excluded loss) and treats clock/pizza labels as descriptive summaries of those signatures. This strengthens, not weakens, novelty: PE → signature shifts is measurable regardless of how the dichotomy debate resolves.

**D7. Git identity set repo-locally** to the user's known email (nafis5341800@gmail.com, name "Nafis") because no global git identity existed; repo-local only, no system-wide change.

**D8. Parameter-count bracket adjusted to fit the mandated architecture.**
The brief fixes the Nanda-style config (1 layer, d_model 128, 4 heads, d_mlp 512, vocab 114, seq len 3) AND demands ~0.4–1.2M params — but that config computes to ~0.23M (embeds 29k + attention 65.5k + MLP 131k). The two constraints are mutually inconsistent. Resolution: the architecture wins (comparability with Nanda et al. 2301.05217 is the scientific point); the model.py assertion bracket is set to 0.2–1.2M and the discrepancy is disclosed here and in the proposal. The 2-layer variant (~0.42M) satisfies the original bracket.

**D9. Parallelism config: 8 workers × 1 torch thread; 12 workers forbidden.**
Measured (smoke_test.md §2–3): intra-run thread scaling saturates (16 threads only 2.6× over 1), so many single-thread processes win — 7.3 ep/s aggregate at 8×1 vs 5.2 ep/s for one 16-thread run. 12 workers crashed 6/12 processes with CPU-allocator OOM on this 13.9 GB machine. Hard ceiling: 8 concurrent runs; launcher (`src/launch.py`) defaults to 8.

**D10. Day-2 compute cuts to fit the 12 h budget (measured necessity).**
Uncut plan (25 × 40k epochs at measured 12 ep/s) ≈ 25 h — infeasible. Cuts: (a) val eval every 5 epochs in grid runs — measured 1.65× throughput, onset resolution loss negligible (±5 epochs vs ~10⁴-epoch onsets); (b) first-pass epoch cap 15k (baseline groks ~10–11k in Nanda et al.; censored runs are data, extension pass reserved); (c) staged seeds (core 4→5, OOD 2→3); (d) wd-sweep and frac-0.5 ablations dropped from Day 2. Full arithmetic in smoke_test.md §4; proposal §4 rewritten accordingly. H1–H3 all remain fully testable within the hard A+B commitment (worst case 11.4 h).

## 2026-07-18 evening → 2026-07-19 (Day 2)

**D11 (Day 2, 22:45). scipy 1.18.0 added to requirements.** Needed for pre-registered Kruskal–Wallis and Mann–Whitney tests (Day-2 brief §Phase 2). Log-rank test, Holm correction, and bootstrap CIs implemented manually in `src/analysis.py` (unit-tested) to avoid heavier deps. Tooling addition only — no experimental-parameter change.

**D12 (Day 2, 22:45). Restricted/excluded loss implemented at the logit level.** Nanda et al.'s original restricted loss projects internal MLP activations onto key-frequency directions; our 1-layer models are architecturally identical but the analysis operates on full-grid logits: group logits by s=(a+b) mod p, Fourier-project the (s, class) matrix onto key frequencies (restricted) or subtract that projection (excluded). Equivalent in spirit, cheaper, and PE-agnostic (works for all 5 conditions incl. RoPE/ALiBi where embedding-space projections differ). Documented in the paper's method section as "logit-level restricted/excluded loss."

**D13 (Day 2, 00:49). HOUR-2 PACE GATE: PASS — no action.** Measured 11.7 ep/s aggregate (conservative, checkpoint-lagged) vs 8.8 trip threshold; Pass A stays at 5 PE × 4 seeds. Evidence snapshot in worklog 00:48. Bonus observations at gate time: 2 early-stops already (nope_s1 grokked, early-stop at ep 7365; learned_s3 at ep 10405), freeing workers ahead of worst-case schedule; preliminary onset ordering nope < learned consistent with H1 (n=1 each, no conclusion drawn).

**D14 (Day 2, 10:53). SESSION INTERRUPTION overnight; gates executed retroactively; Pass B delayed 5.5 h.** The session terminated some time after the 00:49 gate; the 02:30 censoring check, the 04:37 pivot gate, and the automatic Pass-B handoff did not execute on schedule. Pass A itself was unaffected (detached processes): completed 20/20 in 6.74 h, 0 failures, ~05:21. Pass B launched immediately on session resumption (10:54:26). Impact: schedule slack consumed; Pass C shrinks accordingly (see D16). Mitigation for tonight: Pass C launched detached with the same crash-independence.

**D15 (Day 2, 10:55–11:05). CENSORING GATE and PIVOT GATE (retroactive, evidence-based).**
- *Censoring gate:* 15/20 runs censored under the strict sustained-1000 onset rule — but censoring is strongly differential by PE (sinusoidal 4/4 censored with val 0.25–0.93; nope/rope/alibi each have 3/4 runs at val ≥0.99 by cap whose sustained window merely failed to complete). Per the pre-registered rule ("differential censoring BY PE is a positive H1 result; only near-universal censoring triggers the fraction change"), **no fraction change**. The 0.3→0.4 trigger explicitly does NOT fire.
- *Pivot gate (hour-6, deadline 04:37, executed 10:55):* criterion for pivot = no PE effect on onset AND no effect on signatures. Evidence against pivot: (a) differential grok-by-cap (sinusoidal 0/4 vs 11/16 others); (b) all-runs signature permutation test η²=0.434, p=0.013; (c) onset spread 6.4k–13.8k among resolved onsets. **NO PIVOT.** Honest ledger: both strict pre-registered tests are currently underpowered nulls (log-rank p=0.64 with 15 censored; matched-post-grok H2 permutation p=0.52 with n=11) — resolution comes from the pre-authorized 40k extension pass (D16).

**D16 (Day 2, 19:15). PASS C CONTENTS: extend40 + core-ext; ood-ext and depth2/p97/ops dropped.**
Selection principle: the headline needs statistical power, not breadth. (1) `extend40` — 40k-epoch deterministic reruns of the 9 non-grokked censored core runs (all 4 sinusoidal + learned s0/s1, nope s3, rope s3, alibi s3), stored as `_x40`, originals untouched; resolves whether sinusoidal is blocked or delayed (the headline's sharpest claim) and converts up to 9 censored H1 data points into onsets. (2) `core-ext` — 5th core seed, powering the matched-post-grok H2 test (currently p=0.52 at n=11). Arithmetic: 9×40k + 5×15k = 435k epochs ≈ 10.5 h at 11.5 ep/s → done ~05:45, inside the overnight window. Dropped: ood-ext (H3 already strong at R² 0.92 vs −1.59; adding seed 2 is the first Day-3 morning option), depth2/p97/ops (breadth without power; future work). No EXPLORATORY addition — the extension pass IS the de-risking move. Launched detached via scripts/passC.ps1.

**D17 (Day 2, 21:45–22:00). REBOOT-PROOFING after the 20:56 reboot (user directive: "nothing gets lost from now on").**
(1) `train.py --resume-every 1000` (default on): atomic full-precision resume states (model fp32 + optimizer + counters + elapsed); full-batch training uses no RNG after init, so resumption is **bit-exact** — verified by test (`test_resume_is_bit_exact`: interrupted-and-resumed final weights identical to uninterrupted reference; CSV trimmed at resume point to prevent duplicate epochs). Caps any crash/reboot loss at ≤1000 epochs (~15 min) per run. (2) `scripts/passC.ps1` hardened: double-launch guard (exits if grid processes exist). (3) Windows scheduled task `AAAI_PassC_AutoResume` (at-logon, user-level) runs `scripts/passC_auto.ps1` → auto-relaunches Pass C after any reboot; idempotent (guard + --skip-existing + resume). (4) Current workers restarted onto the new code at 21:59:38 (passC4) — cost: the 21:10 rerun's ~3.5k epochs redone once; benefit: bounded loss for the remaining ~9 h. Incident during restart: killing only python processes let the passC2 PowerShell parent advance to core-ext prematurely (~1 min); parent+children then killed cleanly, relaunched, verified 0 stale processes. Metric definitions and training math unchanged — execution-robustness only.

## 2026-07-20 (Day 3)

**D18 (Day 3, 04:35). extend40b: resolve the 6 near-cap censored runs.** Pass-C harvest showed all 9 extended runs grokked (sinusoidal onsets 15.6k–27.5k → "delays", not "blocks") but strict H1 rank tests remain null (log-rank p=0.70) because the 6 runs with val ≥0.99 at 14999-but-incomplete-sustained-window (nope_s2, learned_s2, rope_s0/s2, alibi_s0/s2) are censored at exactly the horizon where sinusoidal's events fall — informative censoring introduced by D16's economy. Extending them (~2 h, 8-worker pool, deterministic reruns to 40k with early stop ~15.2–16k expected) converts them to resolved onsets and gives the log-rank its power. Within the proposal's pre-authorized extension reservation. Also within budget: paper writing proceeds in parallel; stats/figures finalized after harvest (~06:30).

## 2026-07-20 (Session 4 — GPU grid prep)

**D19 (S4, 09:30). W4 reproducibility patch: `analysis --matched` committed.** The paper's headline H2 number (η²=0.368, p=0.001, n=20) previously existed only as an ad-hoc console computation; `h2_permutation.json` holds the all-runs variant (η²≈0.11, p≈0.8 — different conditioning, not a contradiction: it includes censored/non-grokked endpoints). New committed path `matched_postgrok_subset` + `matched_test` writes `h2_permutation_matched.json`, wired into `--all`; verified to reproduce the exact values from the committed `h2_final_signatures.csv`; selection logic unit-tested. Both JSONs are produced so the two conditionings are permanently distinguishable.
