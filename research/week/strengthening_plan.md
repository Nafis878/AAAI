# Strengthening Addendum — PRE-REGISTRATION

**Committed 2026-07-20 (Session 4), BEFORE any GPU run exists.** This document fixes the analyses, predictions, and success/failure criteria for the GPU strengthening grid. Its git commit timestamp is the pre-registration record. Deviations require a dated `decisions.md` entry.

## Why (weaknesses being fixed)
- **W1 scope:** all current claims rest on one setting (mod-113 addition, 1 layer). → S-R1 replicates across four new settings.
- **W2 H1 power:** pre-registered omnibus onset tests are null at 4–5 seeds/PE (Kruskal p=0.19, log-rank p=0.32); only the exploratory Fisher test is significant. → S-H1 reruns at 10 seeds/PE.
- **W3 H3 circularity:** "signature predicts OOD better than PE label" could be dismissed as "signature merely encodes whether the run grokked." → S-H3 adds a binary grok-flag baseline and pre-commits to reporting the honest outcome either way.

## Hardware doctrine (binding)
GPU (T4) runs are the paper's **primary** statistics; they live in `experiments_gpu/`. The 50 existing CPU runs are reported as an **independent cross-hardware replication** and are NEVER pooled into GPU tests. The CPU↔GPU parity check (runbook phase 2) validates that the two hardware classes produce the same curves within tolerance; that validation is a robustness result, not a license to merge datasets.

---

## S-H1 — H1 statistical power (fixes W2)

**Grid:** p=113, add, L1, **10 seeds (0–9) × 5 PE**, cap 40k, early stop on sustained ≥99% (manifest P0).

**Pre-registered tests** (primary H1 tests in the revised paper; the CPU 4–5-seed versions become the replication):
1. k-group **log-rank** across the 5 PEs on onset (censored included). `logrank_k` in `src/analysis.py`.
2. **Kruskal–Wallis** on resolved onsets.
3. **Holm-corrected pairwise Mann–Whitney**, sinusoidal vs each other PE.

**Directional prediction:** sinusoidal onset **stochastically dominates** (is delayed) relative to every other PE; its median onset rank is 5/5.

**Success:** log-rank p<0.05 AND sinusoidal significantly slower than ≥3 of the other 4 PEs after Holm. **Partial:** sinusoidal-slowest direction holds (rank 5) but omnibus p≥0.05 — reported as "directionally replicated, still seed-limited." **Failure:** sinusoidal not the slowest / direction inconsistent — reported plainly; the CPU horizon effect would then stand alone and the H1 claim weakens to "sinusoidal fails to grok within standard horizons" without the stronger delay claim. Expected censoring at 40k ≤10%; if >25% censor, note power still limited and rely on log-rank.

## S-R1 — cross-setting replication (fixes W1)

**Settings** (each 5 seeds × 5 PE, cap 40k): p=97 add (P1); p=113 sub (P1); p=113 mul (P2); p=113 add depth-2 (P2). Multiplication and subtraction may not grok for all PEs at 30% data — non-grokking is itself a reported datum, handled by censoring.

**Per-setting readouts** (`src/analysis.py`: `sinusoidal_delay_rank`, `matched_test` per setting, `attention_split_delta`):
- (a) **Dynamics direction:** median-onset rank of sinusoidal (prediction: slowest, or at least slower-than-median).
- (b) **Circuit separation:** matched-post-grok permutation η²/p among grokked runs (prediction: p<0.05 where ≥3 PEs grok).
- (c) **Attention-dependence split:** attention-level {RoPE, ALiBi} uniform-ablation Δ **<** embedding-level/absent {NoPE, sinusoidal, learned} (prediction: direction holds).

**Combined evidence:**
- **Sign-consistency count:** number of settings (out of 4 new + the anchor p113-add-L1) in which each of (a),(b),(c) holds, reported as a table.
- **Stratified permutation** (`stratified_permutation`): pool all grokked runs across settings, permute PE labels *within* each setting, statistic = mean within-setting η². Tests whether PE→signature structure is consistent across settings beyond chance.

**Success:** each of (a),(b),(c) holds in ≥3 of 5 settings AND the stratified permutation p<0.05. **Partial:** the attention-split (c) — the paper's mechanistic core — holds in ≥3/5 but dynamics or per-setting circuit tests are mixed. **Failure:** (c) holds in ≤2/5 settings — then the attention-dependence finding is reported as p=113-add-specific, not general, and the paper's scope claim is narrowed accordingly. All outcomes reported.

## S-H3 — OOD without circularity (fixes W3)

**Grid:** p=113 add, **4 seeds × 5 PE** rerun on GPU with operand-row holdout a∈[100,112] (P3); PLUS an **alternative OOD construction** b∈[100,112] at 2 seeds × 5 PE (P3, `--ood-b-range`).

**Pre-registered comparison** (`h3_analysis`, now three predictors): leave-one-out prediction of OOD accuracy from
(i) full 6-metric **signature vector** (ridge),
(ii) **PE label** (group mean),
(iii) **binary grokked-flag** alone (group mean).

**Prediction:** R²(signature) > R²(grok-flag) > R²(PE-label). The signature carries graded circuit information beyond the binary grok/no-grok split.

**Success:** R²(signature) meaningfully exceeds R²(grok-flag) (Δ>0.05) on BOTH OOD constructions → signatures carry more than grok status. **Honest-null (pre-committed):** if R²(signature) ≈ R²(grok-flag), we report exactly this — *"the circuit signature predicts OOD, but not measurably beyond whether the run grokked; the contribution is that the signature ⊇ grok status while the PE label ⊉ it"* — which still defeats the "PE label is enough" position and defuses the circularity critique. **Failure:** PE label matches signature → H3 contribution retracted. `signatures_beat_grok_flag` in `h3_ood.json` records the outcome mechanically.

---

## What counts, in one line each
- **S-H1 replication success** = log-rank significant + sinusoidal delayed. Failure = sinusoidal not slowest.
- **S-R1 replication success** = attention-split holds ≥3/5 settings + stratified p<0.05. Failure = split ≤2/5.
- **S-H3 success** = signatures beat grok-flag on both OOD types. Honest-null = they tie (still publishable). Failure = PE label suffices.

## Exploratory (labeled, not confirmatory)
Onset ordering among non-sinusoidal PEs; depth-2 circuit differences beyond the ablation split; per-frequency embedding structure; `torch.compile` speedups. Any analysis not listed above is exploratory.
