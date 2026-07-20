# RESULTS_TODO — mechanical map for the integration session

After the GPU grid completes, results land in `research/week/results/`. This maps each manuscript number/figure/table to the result file that updates it. GPU numbers become the **primary** statistics; the current CPU numbers move to a "cross-hardware replication" paragraph (do not delete them).

## Numbers in `paper/main.tex`

| Manuscript location | Current (CPU) value | Replace from | Notes |
|---|---|---|---|
| Abstract + H2 §: matched permutation | η²=0.368, p=0.001, n=20 | `research/week/results/h2_permutation_matched.json` | Now n≈50 (10 seeds); expect tighter |
| Abstract + H1 §: sinusoidal median onset | 20,980 vs 14,350–15,215 | `h1_stats.json → median_onset_by_pe` | 10-seed medians |
| H1 §: omnibus tests | Kruskal p=0.19, log-rank p=0.32 (null) | `h1_stats.json → kruskal, logrank` | **Primary test now**; CPU null → replication paragraph |
| H1 §: Fisher 0/6 vs 14/24 | p=0.014 (exploratory) | recompute on GPU grokked-by-15k counts | Keep labeled exploratory |
| H2 §: uniform-ablation split | RoPE 1.19/ALiBi 1.96 vs NoPE 4.00/sinus 4.19/learned 5.59 | `table1_per_pe.csv → uniform_abl_delta` | Direction is the claim |
| H3 §: LOO R² | 0.92 (sig) vs −1.59 (PE) | `h3_ood.json → loo_r2_signatures, loo_r2_pe_label` | **ADD** `loo_r2_grok_flag` + `signatures_beat_grok_flag` (S-H3) |
| §Reproducibility: runtime | ~22 h, 8-core CPU | `logs/calibration.json` + grid wall-time | Report T4 hours; CPU as replication |
| §Experimental Setup: run count | 50 runs | manifest total that completed | GPU primary + 50 CPU replication |

## New content to ADD (from the pre-registered plan)

| New paragraph/table | Source | Plan ref |
|---|---|---|
| S-R1 replication table (settings × {delay-rank, matched-η², attention-split}) | `h1_stats.json` + per-setting `matched_test` + `attention_split_delta` per setting | strengthening_plan §S-R1 |
| S-R1 stratified permutation | `stratified_permutation` output | §S-R1 combined |
| S-H3 grok-flag baseline sentence | `h3_ood.json → loo_r2_grok_flag` | §S-H3 |
| Cross-hardware replication paragraph | CPU `research/day2/results/` vs GPU `research/week/results/` + `logs/parity_report.json` | hardware doctrine |

## Figures (regenerate; do not hand-edit)

| Figure | File | Change |
|---|---|---|
| Fig 1–5 | `research/week/results/figures/*.pdf` | Regenerated from GPU data by the orchestrator; copy into `paper/figures/` (or point `\includegraphics` at the new dir) |
| Table 1 | `research/week/results/table1_per_pe.csv` | 10-seed medians; update the LaTeX `tabular` in `main.tex` §H2 |
| Table 2 (ablations) | `research/week/results/table2_ablations.csv` | Now populated by S-R1 settings |

## Decision points for the integration session (apply strengthening_plan criteria)
1. Read `strengthening_plan.md` success/partial/failure criteria BEFORE looking at numbers.
2. **S-H1:** if log-rank now p<0.05 → promote to primary and soften the "seed-limited" caveat. If still null → keep the honest "direction replicates, power-limited" framing.
3. **S-R1:** fill the sign-consistency table; if the attention-split holds ≤2/5 settings, narrow the mechanistic claim to p113-add (do NOT overclaim generality).
4. **S-H3:** if `signatures_beat_grok_flag` is false, rewrite H3 to the pre-committed honest-null wording (signature ⊇ grok status; PE label ⊉).
5. Update the title if the attention-dependence finding generalizes (it currently names it); keep if setting-specific with a scoped claim.
