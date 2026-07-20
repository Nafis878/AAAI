# Results Analysis — FINAL (updated 2026-07-20 ~07:15 after Pass C + extend40b; original Day-2 version below)

## FINAL verdicts (50 runs, 0 failures; all numbers from `analysis --all` @ final data)

- **H1 — PARTIALLY SUPPORTED.** Sinusoidal delays grokking: 0/6 grokked at the 15k horizon vs 14/24 others (Fisher p=0.014, EXPLORATORY); with all extensions resolved, median onset 20,980 vs 14,350–15,215; every sinusoidal onset ≥15,650 (above 15/20 other-PE onsets). Pre-registered omnibus tests NULL at n=4–5/PE (Kruskal H=6.09 p=0.19; log-rank χ²=4.67 p=0.32; pairwise Holm ≥0.57) — reported in full. Seed variance ≈ PE variance except sinusoidal.
- **H2 — SUPPORTED (revised direction).** Algorithm-level convergence: ALL grokked runs (incl. delayed sinusoidal) form the same Fourier circuit family (restricted loss ≈0, conc 0.83–0.88 — sinusoidal highest at 0.88). Pre-registered matched-post-grok permutation test POSITIVE with clean ≥1000-epoch-post-onset endpoints: **η²=0.368, p=0.001, n=20 balanced (4/PE)**. Driver: attention-facing metrics — uniform-ablation Δ groups by *where PE injects position*: RoPE 1.19 / ALiBi 1.96 (attention-level → attention-independent circuits) vs NoPE 4.00 / sinusoidal 4.19 / learned 5.59 (embedding-level or absent → attention-dependent). Interpretation post hoc. NOTE: earlier all-25-runs test read η²=0.11 p=0.84 because 6 endpoints were mid-transition cap checkpoints; extend40b replaced them with stabilized post-grok endpoints (D18) — the pre-registered "matched post-grok" conditioning is the valid one.
- **H3 — SUPPORTED** (unchanged): signatures LOO R²=0.92 vs PE-label −1.59; ρ=0.94; grokked OOD runs 99.6–99.9% on held-out rows.

**FINAL HEADLINE:** *PE choice leaves the grokked algorithm invariant but controls when it forms (sinusoidal delays ~1.4× median, 0/6 at standard horizons) and where it lives — attention-level PEs yield attention-ablation-robust circuits, embedding-level/absent PEs yield attention-dependent ones (η²=0.37, p=0.001) — and the circuit signature, not the PE label, predicts OOD (R² 0.92 vs −1.59).*

Paper figure placement: Fig 2 (survival) + Fig 4 (signatures) + Fig 5 (OOD) main; Fig 1, Fig 3 main-if-space; Tables 1 in main.

---

# Day-2 Results Analysis (written 2026-07-19 ~19:10, after Pass A+B complete)

All numbers regenerable via `python -m src.analysis --all && python -m src.figures --all`.
Run inventory: Pass A 20/20 complete (6.74 h, 0 failures), Pass B 10/10 complete (6.52 h incl. 2h48m user-requested pause, 0 failures). Every launched run is reported below; none excluded.

## Headline claim (single sentence)

**Positional-encoding scheme causally modulates whether and when a transformer groks modular addition — sinusoidal APE blocks grokking entirely at the 15k-epoch horizon (0/6 runs vs 14/24 for all other PEs) — and the circuit signature a run develops, not its PE label, predicts out-of-distribution accuracy (LOO R² 0.92 vs −1.59).**

(Pending Pass-C extension: whether sinusoidal is *blocked* or merely *delayed* beyond 15k — either resolution supports H1; the claim wording will be finalized on Day 3 from the 40k data.)

## Verdict per hypothesis

### H1 (dynamics): SUPPORTED — via differential grokking-by-cap; strict onset stats underpowered
- Resolved onsets (early-stopped, sustained ≥99% ≥1000 ep): ALiBi 7130; RoPE 8965; Learned 9410; NoPE 6370 & 13775 (medians in Table 1).
- **Grokked-by-cap (val ≥0.99 at 15k), core grid: sinusoidal 0/4; nope 3/4; rope 3/4; alibi 3/4; learned 2/4.** OOD grid repeats the pattern: sinusoidal 0/2, others 3/8 (slower overall with the reduced pool).
- EXPLORATORY (not pre-registered): Fisher exact, sinusoidal-vs-rest grokked-by-cap, core+OOD combined: 0/6 vs 14/24 → p = 0.0136 (two-sided).
- Pre-registered log-rank on onsets: χ²(4)=2.51, **p=0.64 — null**; 23/30 runs censored at a common cap destroys power. Kruskal–Wallis/pairwise MWU not computable (≤2 uncensored per PE). This is the honest strict readout; the extension pass exists precisely to resolve it.

### H2 (circuit signatures): PARTIALLY SUPPORTED
- All-runs permutation test (final signatures, core grid): **η²=0.434, p=0.013** — PE shapes the end-state signature distribution. But this conditioning mixes "which circuit" with "whether grokked."
- Pre-registered matched-post-grok test: η²=0.29, **p=0.52 (n=11) — null at current power.** Suggestive structure inside it: uniform-attention ablation Δloss separates Learned (median 5.85) from NoPE/RoPE/ALiBi (1.29–2.27) — a candidate "learned APE builds attention-dependent circuits" story needing Pass-C seeds.
- Mechanism validation (Fig 3): embedding Fourier concentration rises at each run's grok onset (5/5 resolved-onset runs within ±1 checkpoint); grokked runs end at conc 0.57–0.81 with restricted loss ≈0 and excluded loss 12–18 (full Fourier circuits); sinusoidal plateaus at conc 0.33, restricted loss 1.5 (partial circuit).

### H3 (OOD): SUPPORTED (modest n, as pre-registered "supporting evidence")
- **LOO prediction of OOD accuracy: signatures R²=0.92 vs PE-label R²=−1.59; Spearman ρ=0.94 (n=10).**
- Content: grokked OOD runs achieve 0.996–0.999 accuracy on operand rows a∈[100,112] never seen in training; censored runs 0.16–0.27. PE label fails because most PEs contain both outcomes across seeds; the signature tracks the outcome.

## Surprises / anomalies (candidate paper paragraph material)
1. **Sinusoidal APE is the outlier, not NoPE.** A priori the "no positional information" condition seemed most at-risk; instead the fixed sinusoidal embedding is what consistently prevents grokking (0/6 across both grids), while NoPE groks fastest in one seed (6370) and most often (3/4). Hypothesis for Day 3 discussion: the fixed sinusoidal position vectors occupy the same Fourier subspace the task circuit needs in the embedding, interfering with key-frequency formation (its embed conc stalls at 0.33).
2. **Grokking-by-cap is strongly seed-dependent within PE** (e.g., learned: seeds {2,3} grok, {0,1} don't by 15k) — onset variance across seeds is comparable to variance across PEs; only sinusoidal escapes this noise floor. Any H1 claim must be about distributions, not point estimates.
3. **OOD grokking is uniformly later** (12.5k–14.7k vs 6.4k–13.8k core) with 11.5% less training data — consistent with known data-fraction scaling of grokking; incidentally strengthens H3's variance.
4. Val-acc plateau sits at ~0.25–0.30 (well above 1/113 chance) long before grokking — partial structure generalizes early; sinusoidal's plateau is distinctly lower (0.17–0.25).

## Figure/table placement for the paper
- **Main:** Fig 2 (survival — the H1 picture), Fig 3 (concentration trajectories — mechanism), Fig 5 (OOD vs signature — H3), Table 1.
- **Main if space, else appendix:** Fig 4 (signature scatter). Fig 1 (dynamics medians) → appendix unless Pass C separates the medians visibly.
- **Appendix:** Table 2 (ablations, populated after Pass C), per-run onset table, exploratory Fisher test.

## Limitations (state plainly in the paper)
- 4 seeds/PE core + 2/PE OOD at Day-2 close (Pass C adds a 5th and extends censored runs to 40k).
- 15k-epoch cap censors 23/30 runs; strict pre-registered survival stats are underpowered pending extension.
- Single task family (mod-113 addition; p=97/sub/mul dropped for budget), single architecture scale (~0.23M params), CPU-only.
- Matched-post-grok H2 comparison currently n=11 → the attention-dependence contrast is suggestive, not confirmed.
- OOD design holds out operand rows (a∈[100,112]); other OOD constructions untested.
