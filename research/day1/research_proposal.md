# RESEARCH PROPOSAL (BINDING for Days 2–3)

**Title:** Positional Encoding as the Hidden Variable in Grokking: PE Scheme Jointly Determines Learning Dynamics and Circuit Identity on Modular Arithmetic

**Status:** LOCKED 2026-07-18 after scoop check (`landscape.md` §2, decisions D5–D6). Any deviation requires a written rationale entry in `decisions.md`.

## 1. Hypotheses

- **H1 (dynamics).** At fixed data fraction, architecture, and optimization settings, the PE scheme causally shifts grokking onset and sharpness. Falsified if all PE conditions' onset distributions overlap within seed noise.
- **H2 (circuit signature).** PE scheme biases which internal algorithm forms, measured as *continuous signatures*: Fourier-power concentration, restricted/excluded loss, logit attribution, and attention-uniformity discriminators (clock/pizza labels used only as descriptive summaries — see D6). Falsified if signatures are statistically indistinguishable across PEs at matched post-grok checkpoints.
- **H3 (OOD).** Circuit signature predicts held-out operand-range behavior better than the PE label alone (signature-based regressor beats PE-label-based predictor of OOD accuracy under leave-one-run-out CV).

## 2. Fixed experimental parameters (no TBDs)

### Task & data
- Primary task: **mod-p addition**, p = 113; secondary p = 97. Secondary ops: subtraction, multiplication (ablations only).
- Tokens: sequence `[a, b, =]`, vocab size p+1 (digits 0..p−1 plus `=`), loss/accuracy read at the `=` position predicting c = (a∘b) mod p.
- **Train fraction 0.3** of all p² pairs; deterministic split: pairs shuffled by `numpy.random.default_rng(seed)` permutation, first 30% train, rest validation. **Split seed = model seed** (each seed s ∈ {0,1,2,3,4} defines one split shared by all 5 PE conditions → paired comparisons within seed).
- Full-batch training (whole train set each step; 1 epoch = 1 step).
- **OOD design (H3):** separate run set trained with row-band holdout — all pairs with a ∈ [100, 112] excluded from train+val; OOD accuracy = accuracy on those 13×113 held-out pairs. 30% fraction drawn from remaining pairs, same seed policy.

### Optimization
- AdamW, lr 1e-3, weight decay 1.0, betas (0.9, 0.98), eps 1e-8, no lr schedule, no gradient clipping.
- Max 40,000 epochs. **Early stop:** stop at the first epoch after val acc ≥ 0.99 has held for 1,000 consecutive epochs (record onset = first epoch of the sustained window). Runs that never grok terminate at 40k.
- Loss: cross-entropy at the `=` position, float32, CPU.

### Models
- **Primary (circuit clarity):** 1-layer decoder-only, d_model 128, 4 heads (d_head 32), d_mlp 512, ReLU, **no LayerNorm**, untied embed/unembed, no biases in attention projections — Nanda-style. Measured parameter count: **~0.23M** (see decisions.md D8: the brief's 0.4–1.2M bracket is inconsistent with the mandated Nanda config; config wins, assertion bracket set to 0.2–1.2M).
- **Secondary (robustness):** 2-layer pre-LN variant, same dims (~0.42M params).
- **PE conditions (the independent variable):** `{nope, learned, sinusoidal, rope, alibi}` — learned/sinusoidal added to token embeddings; RoPE applied to Q,K; ALiBi as additive head-specific linear attention bias; NoPE = nothing (causal mask only).
- Init: PyTorch defaults under `torch.manual_seed(seed)`.

### Core grid (Day 2)
**5 PE × 5 seeds (0–4) × {1-layer, p=113, add, frac 0.3} = 25 runs**, plus ablations (§4).

### Metrics (logged to CSV; checkpoints every 500 epochs + final; grid runs log every 5 epochs — measured compute cut, see §4 and decisions.md D10)
1. Train/val loss and accuracy (every logged row).
2. **Epochs-to-grok**: first epoch of the sustained (≥1,000-epoch) ≥99% val-acc window; ∞ (censored at 40k) if never.
3. **Sharpness**: epochs from val acc 10% → 99%; and max d(val acc)/d(log₁₀ epoch).
4. **Fourier concentration**: fraction of embedding-matrix and logit-spectrum power in the top-5 frequency pairs (Nanda's key-frequency analysis).
5. **Restricted loss** (logits reconstructed from key frequencies only) and **excluded loss** (key frequencies removed) — Nanda 2301.05217 definitions.
6. **Attention diagnostics**: per-head attention entropy at `=`, and **uniform-attention ablation delta** (loss change when attention is replaced by uniform weights — the clock-vs-pizza discriminator, Zhong 2306.17844).
7. **Circuit-signature summary vector** = (4,5,6) at final checkpoint; descriptive clock/pizza label via uniform-ablation threshold.
8. **OOD accuracy** on the held-out operand rows (OOD run set).
- Analysis cautions: compute circuit metrics at final (converged) checkpoints, not only at grok onset (per 2607.06639).

### Statistics
Paired within-seed comparisons across PE conditions; report per-condition mean ± std over 5 seeds, Wilcoxon signed-rank tests (n=5 pairs, exact), and rank-consistency across seeds; effect sizes as median paired differences. Censored runs handled by rank statistics.

## 3. Ablations (Day 2, after core grid — executable ordering and sizes finalized in §4 Pass C after compute measurement)
1. **Depth 2** (pre-LN): 5 PE × 2 seeds, p=113 add.
2. **p = 97**: 5 PE × 2 seeds, add.
3. **Operation**: subtraction and multiplication, {nope, learned, rope} × 1 seed, p=113.
4. **OOD run set (H3)**: 5 PE × 2–3 seeds with row-band holdout — promoted into the core commitment (Pass B, §4).
5. Weight-decay sweep and data-fraction 0.5 — dropped from Day 2 per measured budget (§4); revived only if Passes A–B finish early.

## 4. Day-2 compute plan (calibrated from measured Phase-4 numbers; raw data in `smoke_test.md`)

**Measured on this machine (2026-07-18):** best throughput = **8 parallel single-thread workers, val eval every 5 epochs → 12.0 epochs/s aggregate** (8.3 s per 100 epochs effective per run). 12 workers OOM-crash (13.9 GB RAM ceiling); thread-scaling within a run saturates at 2.6×. Planning uses a conservative 11 ep/s.

**The uncut 25-run × 40k-epoch grid would be ~25 h — over budget. Adopted cuts (decisions.md D10):**
1. **Val eval every 5 epochs** for grid runs (measured 1.65× throughput; onset resolution ±5 epochs vs onsets ~10⁴). Train metrics still every logged row; smoke test verified identical training trajectory (eval is out-of-graph).
2. **Epoch cap 15,000 first pass** (learned-APE baseline groks ~10–11k at these exact settings in Nanda et al.); runs censored at 15k are recorded as "onset > 15k" — itself an H1 datum. An overnight 40k extension pass is reserved for scientifically critical censored conditions (e.g., a PE that never groks in any seed).
3. **Seeds staged:** core 4 seeds (0–3) then extend to 5; OOD 2 seeds (0–1) then extend to 3.

**Run passes** (launcher presets in `src/launch.py`; commands in `smoke_test.md` §5):
| Pass | Contents | Epochs worst | Hours worst (11 ep/s) | Expected |
|---|---|---|---|---|
| A (H1+H2 core) | 5 PE × seeds 0–3, ≤15k | 300k | 7.6 | ~6.5 |
| B (H3 OOD) | 5 PE × seeds 0–1, holdout a∈[100,112], ≤15k | 150k | 3.8 | ~3.2 |
| **A+B** | **hard Day-2 commitment** | **450k** | **11.4 ≤ 12 h budget** | **~9.7** |
| C (slack/overnight, priority order) | core-ext (seed 4) → ood-ext (seed 2) → depth2 (5 PE × 2s) → p97 (5 PE × 2s) → ops (sub/mul × 3 PE × 1s) | ≤450k | overnight OK | — |

Pass C replaces §3's original ablation list as the executable ordering; wd-sweep and frac-0.5 ablations are **dropped from Day 2** (first candidates to revive if Passes A–B finish early; otherwise noted as future work in the paper).

### Day-2 hour-by-hour schedule (measured basis)
| Hour | Work |
|---|---|
| 0:00–0:10 | Resume per CLAUDE.md; `pytest -q`; launch **Pass A** in background |
| 0:10–2:30 | Implement `src/analysis.py`: Fourier concentration, restricted/excluded loss, uniform-ablation delta, onset/sharpness extraction; unit-test against `experiments/smoke_learned_s0` checkpoints |
| 2:30–4:00 | Implement figure pipeline (matplotlib): dynamics curves per PE, onset box plots, signature scatter; test on partial Pass-A CSVs |
| 4:00–6:45 | Monitor Pass A; run H1 analysis on completed runs incrementally |
| ~6:45 | Pass A done (worst 7.6 h) → launch **Pass B**; full H1 stats (paired Wilcoxon, rank consistency) + H2 signatures on all Pass-A checkpoints |
| 6:45–10:30 | H2 analysis + figures; Pass B completes (~10:30 worst) |
| 10:30–11:30 | H3 analysis (signature vs PE-label OOD prediction, leave-one-run-out) |
| 11:30–12:00 | `research/day2/RESULTS.md`; launch Pass C overnight; commit; update CLAUDE.md for Day 3 |

### Day-2 hour-by-hour schedule
| Hour | Work |
|---|---|
| 0:00–0:15 | Resume per CLAUDE.md; `pytest`; launch core grid (25 runs, 4 workers) |
| 0:15–2:30 | While grid runs: implement `src/analysis.py` (Fourier metrics, restricted/excluded loss, uniform-ablation delta) + tests against the smoke-test checkpoint |
| 2:30–3:00 | Core grid done: sanity-scan CSVs; launch OOD set (15 runs) |
| 3:00–4:30 | H1 analysis: onset/sharpness stats + figures; launch ablations 1–2 |
| 4:30–6:30 | H2 analysis on core checkpoints: signatures, per-PE contrasts; launch ablations 3–5 |
| 6:30–8:00 | H3 analysis (OOD runs done): signature-vs-label prediction; assemble figure set |
| 8:00–10:30 | Ablation analyses as runs complete; robustness table; re-run any failed/censored runs |
| 10:30–12:00 | Results write-up in `research/day2/RESULTS.md`; commit everything; update CLAUDE.md for Day 3 |

## 5. Day-3 paper plan
1. Morning: draft structure — Intro (PE as uncontrolled variable in grokking literature), Related Work (table from landscape.md §2), Method, Results (H1→H2→H3), Limitations (CPU scale, 5 seeds, p ∈ {97,113} only), AI-use documentation statement (required by AAAI policy).
2. Afternoon: figures finalized from Day-2 CSVs; reproducibility checklist; every citation re-verified against arXiv (especially 26xx preprint IDs).
3. Evening: validate against the COMPLIANCE CHECKLIST in `aaai_requirements.md`; final repo cleanup; camera-ready-style PDF build if a LaTeX toolchain is available, else a complete .tex source + build instructions.
- Authorship note (AAAI policy): the human researcher is the author; this AI system's role (research assistance, code, analysis) is documented in the manuscript's AI-use statement; no AI-generated body text except as experimental artifact.

## 6. Fallback (pre-authorized trigger conditions)
Switch to **C2: NoPE position-circuit formation** (candidates.md) only if: (a) a scooping paper surfaces (D5 criteria), or (b) Day-2 core grid shows *no* PE effect whatsoever (null H1 AND null H2) by hour 6 — in which case the null result is folded into C2's framing (how NoPE self-organizes position, with the PE grid as controls). Either trigger requires a decisions.md entry. All infrastructure is shared; no code is discarded.
