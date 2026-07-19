# DAY 2 REPORT — 2026-07-19

**Status: COMPLETE.** Pass A+B fully executed and analyzed; figures built; headline claim established; Pass C running overnight. Timeline stretched by two external events (overnight session interruption, 2h48m user-requested pause) — both absorbed without data loss.

## 1. Run inventory

| Pass | Contents | Launched | Completed | Grokked | Censored | Failed |
|---|---|---|---|---|---|---|
| A (core, H1+H2) | 5 PE × seeds 0–3, ≤15k ep | 22:36 (Jul 18) | 05:21, 6.74 h | 5 early-stop + 6 at-cap ≥99% | 15 strict / 9 true non-grok | 0 |
| B (OOD, H3) | 5 PE × seeds 0–1, holdout a∈[100,112] | 10:54 | 18:5x, 6.52 h (incl. pause) | 3 | 7 | 0 |
| C (overnight) | extend40 (9 × 40k `_x40`) + core-ext (seed 4 × 5) | 18:59:37 | ETA ~05:30–06:00 | — | — | verified alive |

Every launched run is accounted for; raw CSVs/checkpoints append-only (extensions under new `_x40` names).

## 2. Gate decisions (all logged in decisions.md + worklog)
- **Hour-2 pace gate (00:48): PASS** — 11.7 ep/s ≥ 8.8 (D13). No cut.
- **Censoring gate: NO ACTION** — censoring strongly differential by PE (sinusoidal 4/4 vs others' near-miss sustained windows); pre-registered rule says that's a positive H1 datum (D15).
- **Hour-6 pivot gate: NO PIVOT** — executed retroactively at 10:55 after overnight session interruption (D14); evidence: differential grok-by-cap + η²=0.43 p=0.013 signatures (D15).
- **Incidents:** session interruption overnight (Pass B delayed 5.5 h; D14); user-requested pause 13:18–16:07 (suspended in place, zero loss); false-alarm "stall" 22:50–22:58 Day 1 (diagnosed, tooling hardened, logged).

## 3. Hypothesis verdicts (full detail in `results_analysis.md`)
- **H1 SUPPORTED** (via differential grokking-by-cap): sinusoidal 0/6 grokked across both grids vs 14/24 for other PEs (exploratory Fisher p=0.014); resolved onsets ALiBi 7130 < RoPE 8965 < Learned 9410 < NoPE 10072 (median). Strict log-rank p=0.64 — underpowered by 23/30 censoring; Pass C resolves.
- **H2 PARTIAL**: all-runs signature permutation η²=0.434, p=0.013; pre-registered matched-post-grok version p=0.52 (n=11) pending Pass-C power. Mechanism confirmed: Fourier concentration rises at grok onset (5/5); grokked runs = full Fourier circuits (restricted loss ≈0), sinusoidal partial (conc 0.33). Candidate contrast: Learned's uniform-ablation Δ 5.85 vs others 1.3–2.3.
- **H3 SUPPORTED** (modest n=10, per proposal framing): signatures LOO R²=0.92 vs PE label −1.59; ρ=0.94. Grokked OOD runs: 99.6–99.9% on never-seen operand rows.

## 4. Headline claim
**PE scheme causally modulates whether/when grokking happens (sinusoidal APE blocks it at the 15k horizon: 0/6 vs 14/24), and the formed circuit's signature — not the PE label — predicts OOD behavior (R² 0.92 vs −1.59).** Final wording depends on Pass C: "blocks" vs "delays ≥2.7×" for sinusoidal.

## 5. Figure/table inventory (all regenerable: `python -m src.figures --all`)
`paper/figures/`: fig1_dynamics, fig2_onset_survival, fig3_signature_trajectories, fig4_final_signatures, fig5_ood_vs_signature (.pdf+.png each, CVD-validated palette).
`research/day2/results/`: h1_runs.csv, h1_stats.json, h2_final_signatures.csv, h2_trajectories.csv, h2_permutation.json, h3_ood.json, table1_per_pe.{csv,md}, table2_ablations.{csv,md}.
Paper placement decided in results_analysis.md §Figure placement.

## 6. Pass C (running)
extend40: 40k reruns of the 9 non-grokked core runs (deterministic seeds → identical trajectory to 15k, then continue; `_x40` dirs; finished extensions auto-supersede originals in analysis). Then core-ext seed 4. 435k epochs ≈ 10.5 h → **ETA ~05:30–06:00**. Monitor armed (completion/errors/disk). Rationale and drops: D16.

## 7. Top 3 risks for Day 3
1. **Sinusoidal groks somewhere in 15k–40k** → headline softens from "blocks" to "delays ≥Nx". Both are H1-positive; results_analysis.md §Headline pre-words both variants. Check `_x40` sinusoidal runs FIRST on Day 3.
2. **Matched-post-grok H2 stays null even at n≈16–20** → paper reframes H2 honestly: "PE shapes end-state signatures (p=0.013) chiefly through grok/no-grok fate; among grokked runs, signatures converge" — itself a publishable finding (algorithmic universality across PEs, cf. NeurIPS-2025 unification). Do NOT oversell the learned-vs-others ablation contrast unless it clears the permutation test.
3. **Citation re-verification debt**: every `[UNVERIFIED — from seed]` and `[V-listed]` item in DAY1_REPORT §3 must be re-opened before entering the paper (AAAI policy: no unverified references). Budget ~1 h of Day 3.

## 8. Day-2 acceptance criteria self-check
- [x] All Pass A+B runs completed/accounted (30/30, 0 failures; censored explicitly tabulated)
- [x] Checkpoints: ~35 log-spaced fp16/run + final, all 30 runs + smoke (fig3 built from them)
- [x] pytest green: 26 tests incl. metric + naming tests
- [x] `python -m src.figures --all` regenerates everything from raw data (verified twice)
- [x] All pre-registered analyses reported incl. nulls (log-rank p=0.64; matched H2 p=0.52); exploratory items labeled
- [x] Gate decisions each logged incl. "no action" (D13–D15)
- [x] results_analysis.md headline backed by numbers
- [x] Pass C launched detached + verified alive (8 workers + monitor)
- [x] DAY2_REPORT.md written; CLAUDE.md updated for Day 3
- [x] One commit per phase; raw data never deleted/modified (pause scripts and `_x40` scheme preserved append-only)
