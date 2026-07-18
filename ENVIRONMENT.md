# ENVIRONMENT.md — measured on Day 1 (2026-07-18)

All values below were measured on this machine on 2026-07-18, not assumed.

## Hardware

| Component | Value | How measured |
|---|---|---|
| CPU | AMD Ryzen 7 5700G with Radeon Graphics | `Get-CimInstance Win32_Processor` |
| Cores / threads | 8 physical cores / 16 logical processors | same |
| RAM | 13.9 GB total | `Get-CimInstance Win32_ComputerSystem` |
| GPU | AMD Radeon(TM) Graphics (integrated) — **no CUDA, no nvidia-smi** | `Get-CimInstance Win32_VideoController`; `nvidia-smi` not on PATH |
| Verdict | **CPU-only training.** All experiment planning is sized to 8C/16T CPU. | — |

## OS & tooling

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Shells | PowerShell 5.1 (primary), Git Bash (POSIX) |
| Git | 2.49.0.windows.1 (available in Git Bash; **not** on PowerShell PATH) |
| Python (system) | 3.10.4 on PATH; 3.12.4 via `py -3.12` |
| Project venv | `.venv/` created with `py -3.12 -m venv .venv` → Python 3.12.4 |
| uv | not installed (used stdlib venv instead) |

## Python environment

- Activate (PowerShell): `.\.venv\Scripts\Activate.ps1` — or call `.\.venv\Scripts\python.exe` directly (preferred in scripts).
- Activate (Git Bash): `source .venv/Scripts/activate`
- Packages pinned in `requirements.txt` (torch is the CPU wheel from `https://download.pytorch.org/whl/cpu`).

## Compute budget implications

- Day-2 experimental grid must fit in ~12 wall-clock hours on this CPU.
- Measured timing from the Phase-4 smoke test is recorded in `research/day1/smoke_test.md`; the Day-2 plan in `research/day1/research_proposal.md` is sized from those measurements.
- Parallelism strategy: multiple independent training processes (one per run) with `torch.set_num_threads()` capped so total threads ≤ 16; exact worker count chosen from measured contention in the smoke test.
- RAM is not a constraint: each mod-113 run holds ~12.8k examples of length 3 and a ~0.5M-param model (< 100 MB per process).
