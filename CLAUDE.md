# CLAUDE.md — AAAI-27 Transformer Research Project

## Mission

3-day autonomous research project producing a novel, reproducible transformer paper meeting AAAI-27 quality standards.

- **Day 1 (2026-07-18):** requirements, literature scoop check, problem commitment, data + training scaffolding, compute calibration. **Status: COMPLETE — see `research/day1/DAY1_REPORT.md`.**
- **Day 2:** run the full experimental grid per `research/day1/research_proposal.md`; analysis + figures.
- **Day 3:** write the paper in `paper/`, validate against the compliance checklist in `research/day1/aaai_requirements.md`.

## Research topic (BINDING — locked on Day 1)

**"Positional Encoding as the Hidden Variable in Grokking: PE Scheme Jointly Determines Learning Dynamics and Circuit Identity on Modular Arithmetic."**
Full hypotheses (H1 dynamics / H2 circuit identity / H3 OOD), fixed parameters, and the Day-2 hour-by-hour schedule live in `research/day1/research_proposal.md`. Overturning any of it requires written rationale in `decisions.md`.

## Directory map

```
CLAUDE.md            — this file (update at end of every phase/session)
ENVIRONMENT.md       — measured hardware/OS/tooling facts
decisions.md         — every judgment call, with rationale (append-only log)
requirements.txt     — pinned Python deps (torch is CPU wheel)
research/day1/       — Day-1 deliverables (requirements, landscape, proposal, smoke test, report)
src/                 — data.py (mod-p datasets), model.py (transformer + pluggable PE), train.py (training loop)
tests/               — pytest suite; must stay green
data/                — generated datasets (gitignored; regenerate via src/data.py — fully deterministic)
experiments/         — per-run CSV metric logs (committed; checkpoints gitignored)
paper/               — Day-3 manuscript
logs/                — scratch logs (gitignored)
```

## Conventions

- Python 3.12 venv at `.venv/`; invoke as `.\.venv\Scripts\python.exe` (PowerShell) or `.venv/Scripts/python` (Git Bash). Do NOT use the system Python 3.10.
- Git works only in Git Bash on this machine (not PowerShell PATH).
- Everything deterministic: all randomness goes through explicit seeds passed as arguments. No network downloads for data — datasets are synthesized by `src/data.py`.
- One git commit per phase/work-unit with a descriptive message.
- Scientific integrity: never fabricate results or citations. Citation status tags: verified-with-URL+access-date, or `[UNVERIFIED — from seed]`.
- AAAI LLM policy (constrains Day 3): LLMs cannot be authors or cited sources; LLM-generated body text prohibited except as experimental artifact; AI use must be documented in the manuscript. See `research/day1/aaai_requirements.md`.
- CPU-only machine (8C/16T Ryzen 5700G, 13.9 GB RAM). Cap total torch threads across parallel runs at 16. Day-2 grid budget: ~12 h; sized from measured timing in `research/day1/smoke_test.md`.

## How Day 2 should start (target: training within 5 minutes)

1. Read this file, then `research/day1/research_proposal.md` (the binding plan) and `research/day1/smoke_test.md` (measured timing + launch commands).
2. Sanity check: `.\.venv\Scripts\python.exe -m pytest -q` (must pass).
3. Launch the core grid exactly as specified in the proposal's Day-2 schedule (the ready-to-run launcher command is in `smoke_test.md` / proposal §Day-2 schedule).
4. While the grid runs, implement analysis code (Fourier metrics, restricted/excluded loss) — specs in the proposal.
5. Log any deviations in `decisions.md`; update this file at session end.
