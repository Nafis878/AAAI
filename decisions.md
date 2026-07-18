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
