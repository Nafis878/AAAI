# CPU Findings Summary — Positional Encoding as a Hidden Variable in Grokking

**Prepared 2026-07-20.** This is the complete, self-contained summary of what the project established on the CPU-only machine (50 deterministic runs, 0 failures). It is the final basis for the paper. Every number regenerates from committed data via `python -m src.analysis --all --matched && python -m src.figures --all`.

## The question

Grokking studies on modular arithmetic conventionally fix the positional-encoding (PE) scheme; mechanistic "clock/pizza" circuit studies vary attention type. PE was an uncontrolled variable in both. We made **PE the controlled independent variable** — {NoPE, learned, sinusoidal, RoPE, ALiBi} — with everything else matched (task $(a+b)\bmod 113$, 1-layer ~0.23M-param transformer, 30% train fraction, AdamW lr 1e-3 wd 1.0, matched seeds and data splits), and asked what PE changes.

## Run inventory (50 runs, 0 failures, all reported)

| Set | Runs | Purpose |
|---|---|---|
| Pass A core (5 PE × seeds 0–3) | 20 | H1 dynamics, H2 signatures |
| Pass B OOD (5 PE × seeds 0–1, operand rows a∈[100,112] held out) | 10 | H3 |
| Pass C extend40 + core-ext (seed 4) | 14 | resolve censored onsets to 40k |
| extend40b (6 near-cap runs to 40k) | 6 | resolve remaining onsets |

24 of 25 core onsets resolved (only the seed-4 batch remained censored at cap). Raw CSVs + checkpoints in `experiments/`; stats/figures in `research/day2/results/` and `paper/figures/`.

## Findings

### H1 (dynamics) — PARTIALLY SUPPORTED: sinusoidal delays grokking
- **Sinusoidal APE grokked 0/6 runs at the standard 15k-epoch horizon vs 14/24 for all other PEs combined** (Fisher exact p=0.014, *exploratory*).
- 40k extensions prove it is a **delay, not a block**: every sinusoidal onset lands at 15,650–27,455 (median **20,980**) vs medians of 14,350–15,215 for the other PEs; every sinusoidal onset exceeds 15 of the 20 other-PE onsets.
- **Honest limit:** the pre-registered omnibus rank tests are **null** at 4–5 seeds/PE (Kruskal–Wallis H=6.09 p=0.19; k-group log-rank χ²=4.67 p=0.32; all Holm-adjusted pairwise p≥0.57). Only the sinusoidal horizon effect exceeds the measured seed-noise floor; the median ordering among the other four PEs is descriptive only. Seed-level onset variance is comparable to PE-level shifts — itself a finding worth stating for small-n grokking work.

### H2 (circuit signatures) — SUPPORTED
- **Algorithm is PE-invariant.** Every grokked run, sinusoidal included, converges to the same Fourier-multiplication circuit family: restricted loss ≈ 0 (top-5 frequencies alone solve the task), embedding Fourier concentration 0.83–0.88 (sinusoidal highest at 0.88). This extends circuit-universality across the PE axis.
- **But PE selects the circuit's attention-dependence.** The pre-registered matched-post-grok permutation test (balanced, n=20, 4/PE, endpoints ≥1000 epochs past onset) is strongly positive: **η²=0.368, p=0.001** (reproduced exactly by `python -m src.analysis --matched`). The separation is carried by the uniform-attention ablation Δloss, which groups PEs by *where they inject position*:
  - **attention-level PEs** (RoPE 1.19, ALiBi 1.96) → circuits nearly indifferent to attention ablation;
  - **embedding-level / absent PEs** (NoPE 4.00, sinusoidal 4.19, learned 5.59) → strongly attention-dependent circuits.
  - Interpretation (labeled post hoc): when position enters inside attention, the model implements Fourier mixing position-agnostically; otherwise the learned attention pattern itself becomes load-bearing.
- Mechanism check: embedding Fourier concentration rises at each run's grok onset (within one checkpoint for all resolved runs) and never rises in non-grokked runs — circuit formation and generalization are one event whose timing PE controls.

### H3 (OOD) — SUPPORTED (modest n, as pre-registered)
- On held-out operand rows, grokked runs transfer near-perfectly (99.6–99.9%); non-grokked runs sit at 16–27%.
- **Circuit signature predicts OOD accuracy; the PE label does not:** leave-one-out R² = **0.92** from signatures vs **−1.59** from PE label; Spearman ρ=0.94 (n=10). PE label fails because most PEs contain both outcomes across seeds — the signature tracks the outcome.

## One-sentence headline

**Positional-encoding choice leaves the grokked algorithm invariant but controls when it forms (sinusoidal delays it, 0/6 at standard horizons) and how much it relies on attention (attention-level PEs yield attention-robust circuits, embedding-level/absent PEs yield attention-dependent ones; η²=0.37, p=0.001) — and a run's circuit signature, not its PE label, predicts out-of-distribution accuracy (R² 0.92 vs −1.59).**

## Honest limitations (stated in the paper)
- Single task family (p=113 addition), single architecture scale (0.23M params, 1 layer), 4–5 seeds/PE, CPU horizon of 15k/40k epochs, OOD limited to operand-row holdout (n=10).
- H1 omnibus rank tests are underpowered at this seed count; the sinusoidal horizon effect and the H2 signature separation are the claims that clear the noise floor. The attention-dependence grouping, though significant under the pre-registered test, has a post-hoc interpretation.
- Known review-facing weaknesses that a larger same-hardware grid would address (more seeds for H1 power; other moduli/operations/depths for scope; a grokked-flag baseline for H3) are documented as future work — they were scoped out of this CPU project.

## Where things live
- Paper: `paper/main.tex` (+ `references.bib`, 12 arXiv-verified citations; `README_BUILD.md`).
- Figures: `paper/figures/fig1..5` (PDF+PNG). Stats: `research/day2/results/*.json|csv` incl. `h2_permutation_matched.json` (the headline number).
- Full narrative: `research/day3/DAY3_REPORT.md`; decision log: `decisions.md` (D1–D19); reproducibility: `research/day1/aaai_requirements.md` §8 checklist.
- Human submission items: `SUBMISSION_CHECKLIST.md`.
