# DAY 3 REPORT — 2026-07-20

**Status: COMPLETE.** All 50 runs finished and analyzed (0 failures across the entire project); final statistics locked; paper written with verified citations; compliance walked. Repository pushed to origin.

## 1. Final run inventory (nothing excluded)

| Set | Runs | Outcome |
|---|---|---|
| Pass A core (5 PE × seeds 0–3) | 20 | 5 early-stop groks; 15 censored at 15k |
| Pass B OOD (5 PE × seeds 0–1) | 10 | 3 groks; 7 censored |
| Pass C extend40 (`_x40`) | 9 | ALL grokked (15.6k–27.5k) |
| Pass C core-ext (seed 4) | 5 | all censored at 15k |
| extend40b (`_x40`, D18) | 6 | ALL grokked (~15.2k–16.5k) |
| **Total** | **50** | **0 failures; 24/25 core onsets resolved** |

## 2. Final hypothesis verdicts (full detail: `research/day2/results_analysis.md` §FINAL)

- **H1 partially supported.** Sinusoidal delays grokking: 0/6 at the 15k horizon vs 14/24 (Fisher p=0.014, exploratory); median onset 20,980 vs 14,350–15,215. Pre-registered omnibus rank tests null at n=4–5/PE (Kruskal p=0.19; log-rank p=0.32; pairwise Holm ≥0.57) — reported in full.
- **H2 supported.** Algorithm invariant across PEs (restricted loss ≈0, Fourier conc 0.83–0.88 everywhere), but the pre-registered matched-post-grok permutation test is positive: **η²=0.368, p=0.001 (n=20, 4/PE)**, driven by attention-dependence — attention-level PEs (RoPE 1.19, ALiBi 1.96 uniform-ablation Δ) vs embedding-level/absent (NoPE 4.00, sinusoidal 4.19, learned 5.59). Interpretation labeled post hoc.
- **H3 supported.** Signature→OOD LOO R²=0.92 vs PE-label −1.59; ρ=0.94 (n=10, modest-n as pre-registered).

**Headline:** PE leaves the grokked algorithm invariant but controls when it forms and how much it relies on attention; the circuit signature, not the PE label, predicts OOD.

## 3. Deliverables

- `paper/main.tex` — full AAAI-27 manuscript (anonymous submission variant, AI-use statement, 5 figures, summary table)
- `paper/references.bib` — 12 citations, every one verified this session (11 via arXiv API batch query 2026-07-20; NeurIPS-2025 unification via neurips.cc 2026-07-18)
- `paper/README_BUILD.md` — build instructions (author kit required for submission build; no LaTeX toolchain on this machine)
- `paper/figures/` — 5 figures (PDF+PNG) regenerated from final data
- `research/day2/results/` — final CSVs/JSON stats + Tables 1–2
- 28 tests green; every number regenerable via `python -m src.analysis --all && python -m src.figures --all`

## 4. Compliance checklist walk (`research/day1/aaai_requirements.md` §8)

**Format**
- [x] AAAI-27 two-column source (`aaai2027.sty` submission mode, Times/Helvetica/Courier Type-1 fonts)
- [ ] **BUILD-TIME ITEMS — for the human submitter:** page count (est. ~7 pages content — verify against the 7+2 limit after building with the kit), font embedding, and appending the kit's `ReproducibilityChecklist.tex` after references. The author kit must be downloaded from aaai.org/authorkit27/ (this machine has no LaTeX toolchain; see README_BUILD.md).
- [x] Anonymous: no names, affiliations, acknowledgements, or identifying URLs in the manuscript
- [x] Reproducibility content ready (Reproducibility section + per-run configs/seeds/logs in repo)

**Content & integrity**
- [x] Every citation opened/verified this session; zero from-memory references
- [x] AI-use documentation statement included in the manuscript
- [x] No AI system authored or cited as a source
- [x] All numbers traceable to committed CSV/JSON under `research/day2/results/`
- [x] Pre-registered nulls reported in full; exploratory/post-hoc items labeled
- [x] Limitations section present and specific
- [ ] **POLICY ITEM — for the human submitter:** AAAI prohibits LLM-generated body text except as experimental artifact (editing/polishing of author-written text is allowed). This manuscript was AI-drafted under human direction, which the AI-use statement discloses — but before submission the human author must substantively revise and take ownership of the prose to satisfy the policy as written. This cannot be discharged by the AI side.

**Reproducibility package**
- [x] Code + deterministic data generation + two-command regeneration of all figures/stats
- [x] Hyperparameters and seed policy documented (proposal + per-run config.json)
- [x] Hardware/runtime disclosure (8-core CPU, ~22 h total, no GPU)

## 5. Project acceptance criteria (3-day brief) — final self-check
- [x] Novel, scoop-checked problem; binding pre-registered proposal executed without unauthorized parameter changes (all deviations D1–D18 logged with rationale)
- [x] Every launched run reported; censoring handled by survival/rank statistics; raw data append-only throughout
- [x] Publication-quality figures (CVD-validated palette) + full statistical reporting including nulls
- [x] Git history: one commit per phase, pushed to https://github.com/Nafis878/AAAI
- [x] Two explicit human-action items remain (kit build + prose ownership) — see §4
