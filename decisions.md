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
