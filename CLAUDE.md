# CLAUDE.md — AAAI-27 Transformer Research Project

## Mission

3-day autonomous research project producing a novel, reproducible transformer paper meeting AAAI-27 quality standards.

- **Day 1 (2026-07-18):** requirements, literature scoop check, problem commitment, data + training scaffolding, compute calibration. **Status: COMPLETE — see `research/day1/DAY1_REPORT.md`.**
- **Day 2 (2026-07-19):** Pass A (20 core) + Pass B (10 OOD) executed and analyzed, 0 failures; figures built; headline claim established. **Status: COMPLETE — see `research/day2/DAY2_REPORT.md` + `results_analysis.md`. Pass C (9 × 40k extensions + seed 4) running overnight, ETA ~05:30–06:00.**
- **Day 3 (2026-07-20):** Pass C + extend40b harvested (50/50 runs, 0 failures); final stats locked (H2 flipped positive: η²=0.368, p=0.001); paper written with 12 verified citations. **Status: COMPLETE — see `research/day3/DAY3_REPORT.md`.** Two human-action items remain: build with the official author kit, and human revision/ownership of the AI-drafted prose (AAAI LLM policy) — detailed in DAY3_REPORT §4.

**PROJECT IS FINAL ON THE CPU RESULTS.** One-page synthesis of all findings: `research/CPU_FINDINGS_SUMMARY.md`. A later GPU-strengthening effort was explored and then removed at the user's request (decisions.md D20); it is recoverable from git history at commit e82559f if ever wanted. Headline H2 number regenerates via `python -m src.analysis --matched` (W4 fix, kept).

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

## How Day 3 should start

1. Read this file, then `research/day2/DAY2_REPORT.md` (run inventory, verdicts, risks) and `research/day2/results_analysis.md` (headline claim + figure placement). `decisions.md` D13–D16 cover Day-2 judgment calls.
2. **Harvest Pass C first**: check `logs/passC4.log` for "PASS C COMPLETE" (relaunch history: reboot 20:56 killed passC; passC2 relaunched 21:10; restarted as passC4 at 21:59 on resume-capable code — D17. Training is now reboot-proof: bit-exact resume every 1000 epochs + at-logon auto-relaunch task `AAAI_PassC_AutoResume`); look at the sinusoidal `_x40` runs before anything else — they decide the headline wording ("blocks" vs "delays ≥Nx" grokking; DAY2_REPORT risk 1). If Pass C is still running, harvest what's done (`--skip-existing` relaunch if it crashed: `scripts/passC.ps1`).
3. Regenerate everything: `python -m src.analysis --all && python -m src.figures --all` (finished `_x40` runs auto-supersede their originals), plus `--h2 --trajectories` for updated Fig 3. Re-run the matched-post-grok H2 permutation (now ~n=16–20) — it decides how H2 is framed (DAY2_REPORT risk 2).
4. **Re-verify every citation** tagged `[UNVERIFIED — from seed]` or `[V-listed]` in `research/day1/DAY1_REPORT.md` §3 before it enters the paper; drop what can't be verified.
5. Write the paper in `paper/` (AAAI-27 two-column, `aaai2027.sty`; get the author kit or fall back to a faithful .tex skeleton + build instructions). Include the AI-use documentation statement (mandatory: this project is AI-assisted research; human submitter is the author). Validate against the COMPLIANCE CHECKLIST in `research/day1/aaai_requirements.md` §8 before calling it done.
6. Commit per phase; final push to origin.

Useful commands: `python -m src.progress` (grid status), `scripts/pause_resume.ps1 -Action pause|resume` (freeze/unfreeze training).
