# Candidate Research Problems (Day 1, 2026-07-18)

Four candidates evaluated. Selection criteria: novelty (per `landscape.md` scoop check), CPU-8-core feasibility within a ~12 h Day-2 budget, and paper-shaped falsifiable hypotheses. **C1 selected** (see `research_proposal.md`); C2 is the designated fallback.

---

## C1 — Positional Encoding as the Hidden Variable in Grokking ★ SELECTED

- **Problem statement.** Grokking studies on modular arithmetic fix the positional encoding (usually learned APE) and vary optimization or architecture; circuit-multiplicity studies vary attention type. Nobody has asked whether the PE scheme itself causally shifts (a) when/how grokking happens and (b) which internal algorithm forms.
- **Testable hypotheses.**
  - H1 (dynamics): at fixed data fraction, architecture, and optimizer, PE scheme shifts grokking onset (epochs to sustained ≥99% val acc) and sharpness by measurable, seed-robust margins.
  - H2 (circuit signature): PE scheme biases the formed circuit's continuous signatures — Fourier-power concentration, restricted/excluded loss, attention-dependence of logits.
  - H3 (OOD): circuit signature predicts held-out operand-range behavior better than the PE label alone.
- **Method sketch.** 5 PE conditions {NoPE, learned APE, sinusoidal APE, RoPE, ALiBi} × ≥5 seeds on mod-113 addition, 1-layer Nanda-style transformer; full metric battery per epoch-checkpoint; circuit analysis post-convergence (heeding 2607.06639's at-grok caution).
- **Experimental design.** Baselines: learned-APE (replicates Nanda et al.). Metrics: epochs-to-grok, sharpness, Fourier concentration, restricted/excluded loss, attention entropy, OOD accuracy. Ablations: p ∈ {113, 97}, op ∈ {add, sub, mul}, frac ∈ {0.3, 0.5}, wd sweep, depth {1, 2}.
- **Nearest prior work.** Zhong et al. 2306.17844 (attention type, not PE); NeurIPS 2025 unification (no PE, no dynamics); Kazemnejad 2305.19466 (PE menu but accuracy-only); 2605.20441 (weight decay as the control knob). Full table in `landscape.md` §2.
- **Novelty call: OPEN** — scoop check negative (decisions.md D5).
- **Feasibility: HIGH.** ~0.5M params, 12.8k examples full-batch; entire core grid is 25 small runs — precisely CPU-scale. Confirmed by Phase-4 smoke test.

## C2 — How NoPE Transformers Build Position Circuits During Training (FALLBACK)

- **Problem statement.** Kazemnejad et al. proved NoPE *can* recover position from the causal mask; nobody has shown *how and when* that circuit actually forms during training, or what breaks it.
- **Testable hypothesis.** NoPE models develop an identifiable position-reconstruction circuit (first-layer uniform-attention head reading the causal-mask prefix statistics) whose formation time predicts downstream length generalization; ablating BOS/uniform-attention heads destroys it.
- **Method sketch.** Train NoPE models on position-sensitive algorithmic tasks (e.g., subtraction, copy/reverse at short lengths); probe position decodability per layer per checkpoint; causal ablations (BOS removal, head knockout); compare against learned-APE controls.
- **Experimental design.** Metrics: position-probe accuracy over training, head-ablation deltas, length-generalization curves. Baselines: APE/RoPE models. Ablations: task, depth, seed.
- **Nearest prior work.** Kazemnejad 2305.19466 (theory + final accuracy); NoPE optimization-rate analyses seen in search results; 2512.12167 (dropping PEs, capability-focused).
- **Novelty call: LIKELY OPEN** (needs its own scoop check before activation — noted in landscape.md §3).
- **Feasibility: HIGH** — reuses this repo's entire data/model/train stack (NoPE is one of our PE configs).

## C3 — Small-Scale Adjudication of Attention-Temperature Scaling Claims

- **Problem statement.** Competing prescriptions exist for attention inverse-temperature scaling vs entropy collapse (e.g., unified inverse-temperature scaling, 2605.12697 `[UNVERIFIED — from seed]`). They make divergent predictions at small scale where exhaustive sweeps are affordable.
- **Testable hypothesis.** A dense (temperature × width × depth) sweep on algorithmic + tiny-LM tasks can falsify at least one published scaling prescription's small-scale predictions.
- **Method sketch.** Grid over softmax temperature multipliers and d_head; measure attention entropy trajectories, stability, final loss; fit each published scaling law and compare predictive error.
- **Experimental design.** Metrics: entropy collapse incidence, loss vs prediction residuals. Baselines: default 1/√d. Ablations: task, depth.
- **Nearest prior work.** 2605.12697 (seed); entropy diagnostics of 2605.20441.
- **Novelty call: MODERATE** — adjudication papers are useful but rated lower-novelty by AAAI reviewers unless a clear winner emerges.
- **Feasibility: MEDIUM-HIGH** — grid is larger than C1's (3D sweep), riskier in a 12 h budget.

## C4 — Data-Fraction × Weight-Decay Phase Map of Grokking Under PE Choice

- **Problem statement.** 2605.20441 maps grokking regimes along weight decay alone. A 2D phase map (data fraction × weight decay) conditioned on PE scheme would test whether PE shifts phase *boundaries*, not just onset times.
- **Testable hypothesis.** PE scheme translates/deforms the memorization↔grokking boundary in the (fraction, wd) plane.
- **Method sketch.** Coarse 2D grid (4 fractions × 4 wds) × 2–3 PEs × 3 seeds on mod-113 addition; classify each run's terminal regime; fit boundary curves.
- **Experimental design.** Metrics: regime classification, boundary location ± CI. Baselines: learned APE map (replication of 2605.20441's axis).
- **Nearest prior work.** 2605.20441 (1D wd axis, fixed PE); Power et al. 2201.02177 (fraction axis, fixed PE).
- **Novelty call: OPEN but derivative** — reads as an extension of 2605.20441.
- **Feasibility: LOW-MEDIUM for Day 2** — 96–144 runs; exceeds the 12 h CPU budget unless heavily cut. Better as a *subset* inside C1's ablations (which is exactly what C1's frac × wd ablations do).

---

## Selection decision

**C1 selected.** Highest novelty (confirmed open gap), tightest fit to hardware, richest per-run information (dynamics + circuits + OOD from the same 25 runs), and C2/C4 survive as its fallback and ablation subset respectively — no work is wasted under any pivot. Rationale recorded here and in `research_proposal.md`; default top pick retained, so no `decisions.md` override entry is required.
