# Transformer Research Landscape 2024–2026 & Scoop Check

**Prepared:** 2026-07-18. **Citation tags:** `[V-fetched]` = abstract/page opened this session; `[V-listed]` = paper title+ID confirmed in web-search results this session; `[UNVERIFIED — from seed]` = seeded appendix only. Access date for all verified items: 2026-07-18.

## 1. Subarea survey (with CROWDED/OPEN call and CPU feasibility)

### 1.1 Efficiency: sparse/linear attention, KV-cache compression, quantization — **CROWDED**, mostly CPU-infeasible
Massive 2024–2026 activity: KV-cache management surveys (2412.19442 `[V-listed]`), eviction/compression methods (CAKE 2503.12491, Expected Attention 2510.00636, 2606.26472, MiniKV — all `[V-listed]`), 2-bit caches, latent attention. Requires LLM-scale serving experiments to be competitive — infeasible here and saturated. ASEntmax (2506.16640, sparse attention for long-context generalization, `[V-listed]`; seed says ICLR 2026) shows the algorithmic end is active too. **Avoid.**

### 1.2 Positional encodings & length generalization — **ACTIVE but with OPEN niches**, CPU-feasible at small scale
Kazemnejad et al. 2305.19466 (`[V-listed]`; NoPE recovers position from the causal mask, Theorem 1) is the anchor. 2024–2026 additions: RoPE-to-NoPE hybrid attention (2501.18795 `[V-listed]`), dropping PEs to extend context (2512.12167 `[V-listed]`), spectral analysis of content-position coupling (2505.13027 `[V-listed]`), randomized-float PEs (2602.14050 `[UNVERIFIED — from seed]`), C-RASP formal framework (2410.02140 `[UNVERIFIED — from seed]`). The *accuracy/extrapolation* side is crowded; the *training-dynamics and circuit* side of PE choice is open (see scoop check). **OPEN niche, feasible.**

### 1.3 Grokking & training dynamics — **ACTIVE, still OPEN**, ideal CPU scale
Foundations: Power et al. 2201.02177, Nanda et al. 2301.05217 (`[V-listed]`, both). 2024–2026 wave: weight-decay as a scalar control parameter for memorization/grokking/collapse regimes with cheap attention diagnostics (2605.20441 `[V-fetched via search summary]`, submitted 2026-05-19, 0.82M–85M params); grokking transferability (2601.09049 `[V-listed]`); circuit synchronization preceding generalization (2606.12966 `[V-listed]`); low-dimensional optimization dynamics (2602.16746 `[V-listed]`); latent structure preceding grokking in MLPs (2603.23784 `[V-listed]`); measurement-validity audit of grokking representation metrics (2607.06639 `[V-listed]` — a caution: at-grok checkpoints are not converged, metrics must be handled carefully). Small models, algorithmic tasks — **exactly CPU scale. OPEN along the PE axis.**

### 1.4 Mechanistic interpretability of algorithmic tasks — **ACTIVE, partially open**, CPU-feasible
Clock circuit (Nanda 2301.05217), clock-vs-pizza (Zhong et al. 2306.17844 `[V-fetched]` — controlled variable is **attention type/hyperparameters, not PE**). Major 2025 development: "Unifying Mechanistic Interpretations of Neural Networks Trained on Modular Addition" (Moisescu-Pareja, McCracken, Létourneau, Precup, Love; NeurIPS 2025, `[V-fetched]` via neurips.cc/virtual/2025/133808): Clock and Pizza are *different parameterizations of the same abstract algorithm*; they vary attention (sigmoidal vs constant), layers, seeds — **not positional encoding, and no grokking dynamics**. Related: universal abstract algorithm for modular addition (2505.18266 `[V-listed]`), modular polynomials circuits (Furuta et al. 2402.16726 `[V-listed]`), modular exponentiation circuits with grokking-like dynamics (2506.23679 `[V-fetched]` — does not vary PE), Discrete-Log Clock for modular multiplication (2606.17399 `[V-listed]`), representation manifolds of modular addition (2512.25060 `[V-listed]`). **Consequence for our H2:** binary clock/pizza classification is contested; circuit-identity metrics must be continuous (Fourier concentration, attention-dependence, gradient-symmetry) rather than a hard two-way label. **OPEN along the PE axis; feasible.**

### 1.5 Attention temperature / entropy-collapse scaling — **OPEN at small scale**, CPU-feasible
Unified inverse-temperature scaling (2605.12697 `[UNVERIFIED — from seed]`); attention-entropy diagnostics appear in 2605.20441 `[V-fetched via search summary]`. Small-scale adjudication of competing scaling claims is viable on CPU. **OPEN, feasible — candidate material.**

### 1.6 SSM vs transformer — **CROWDED at benchmark scale**
Jelassi et al. 2402.01032 (copying, ICML 2024) `[UNVERIFIED — from seed]`; generic Mamba-vs-transformer benchmarking is saturated and needs scale. **Avoid.**

### 1.7 Small-model research — **OPEN-ish**, feasible
TinyStories (2305.07759) `[UNVERIFIED — from seed]` legitimized sub-10M-param research; our whole project rides this wave. Feasible by construction.

### 1.8 Applications beyond NLP/vision (incl. time series) — **CROWDED/contested**
Large time-series transformers remain contested (linear models competitive — Zeng et al. 2205.13504, AAAI 2023 `[UNVERIFIED — from seed]`). Not our comparative advantage. **Avoid.**

## 2. SCOOP CHECK (mandatory)

**Queries run this session (2026-07-18), via WebSearch + arXiv abstract fetches:**
1. `grokking positional encoding transformer modular arithmetic`
2. `positional embedding modular arithmetic circuits mechanistic interpretability clock pizza`
3. `grokking RoPE NoPE ALiBi positional encoding comparison learning dynamics`
4. `arxiv 2026 grokking positional encoding circuit formation transformer "modular addition"`
5. `"positional encoding" OR "positional embedding" grokking onset delay ablation NoPE transformer algorithmic`
6. `arxiv 2605.20441 weight decay grokking attention diagnostics` (seed-ID verification)
Plus targeted abstract fetches: arXiv 2407.17963, 2506.23679, NeurIPS 2025 #133808 (unification).

**Decision rule:** scooped ⟺ an existing paper uses positional-encoding scheme as the controlled variable for (a) grokking dynamics OR (b) circuit identity on modular arithmetic.

### Nearest-neighbor papers and differentiation

| # | Paper | Why it is close | Why it does NOT scoop the top pick |
|---|---|---|---|
| 1 | Zhong et al. 2023, "The Clock and the Pizza" (2306.17844) `[V-fetched]` | Circuit multiplicity on modular addition — the hardest differentiation target | Controlled variable is **attention type (constant vs standard) and width/depth**, not PE; no grokking-dynamics axis |
| 2 | Nanda et al. 2023, progress measures (2301.05217) `[V-listed]` | Same task, same 1-layer architecture, source of our metrics | PE is **fixed** (learned APE); PE never varied |
| 3 | Moisescu-Pareja et al., NeurIPS 2025 unification `[V-fetched]` | Most recent authority on modular-addition circuits | Varies attention parameterization/layers/seeds; **abstract does not address PE; no grokking dynamics** |
| 4 | Kazemnejad et al. 2023 (2305.19466) `[V-listed]` | Compares the same PE menu incl. NoPE | Outcome is **length-generalization accuracy** on downstream tasks — no grokking, no circuits |
| 5 | Xu et al. 2024 (2407.17963) `[V-fetched]` | Relative PE ↔ generalization on arithmetic | Theory of **OOD/length generalization**; no grokking dynamics, no circuit identity, no PE-scheme grid |
| 6 | Furuta et al. 2024 (2402.16726) `[V-listed]` | Circuits in grokked transformers on modular polynomials | Varies the **operation**, not PE |
| 7 | 2605.20441 (2026) weight-decay regimes `[V-fetched via search summary]` | Grokking dynamics control-parameter study; we reuse its attention-diagnostic idea | Control parameter is **weight decay**, not PE |
| 8 | 2506.23679 (2025) modular exponentiation `[V-fetched]` | Grokking-like dynamics + circuit analysis on modular task | Varies **training strategy/task**; explicitly no PE comparison |
| 9 | 2511.23443 (2025) sinusoidal activation for modular addition `[V-listed]` | "Fourier-style" inductive bias accelerating modular addition | Varies **activation/embedding**, not positional encoding scheme |
| 10 | 2505.13027 (2025) spectral analysis of content-position coupling `[V-listed]` | PE-mechanism analysis | General spectral theory of PEs; **not grokking, not modular-arithmetic circuits** |
| 11 | 2607.06639 (2026) at-grok measurement-validity audit `[V-listed]` | Methodology for grokking metrics | Audits metrics; doesn't vary PE. We adopt its caution (evaluate circuits post-convergence, not only at grok onset) |

### Verdict: **NOT SCOOPED**

No located paper uses PE scheme as the controlled independent variable for grokking onset/sharpness or for circuit identity on modular arithmetic. Each sub-claim has individual precedent (grokking metrics: #2, #7; circuit multiplicity: #1, #3; PE menus: #4); their causal unification via PE remains unclaimed — consistent with the seeded gap analysis. Two obligations for the paper:
1. Frame **PE (not attention type)** as the controlled variable and cite #1/#3 explicitly as the adjacent-but-different axis.
2. Treat circuit identity as a **continuous signature** (Fourier concentration, attention-dependence, logit attribution), not a hard clock/pizza binary, per #3.

Evidence and verdict also logged in `decisions.md` (D5).

## 3. Implications for candidate selection

- The default top pick survives the scoop check with a required framing adjustment (continuous circuit signatures).
- Fallback (NoPE position-circuit formation during training) also unscooped as far as searched: the NoPE literature (#4, 2501.18795, 2512.12167) is about capability/extrapolation, not training-time circuit formation; one "optimization perspective on NoPE" line of work exists (seen in search results, rate-of-development theory) — Day-2 fallback would need its own scoop check before activation.
- Attention-temperature adjudication (1.5) is a viable third candidate at CPU scale.
