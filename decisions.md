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
