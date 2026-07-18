# DAY 1 REPORT — 2026-07-18

**Mission status: COMPLETE.** Topic locked, scaffolding built and tested, compute calibrated from measurement, Day 2 can launch training within 5 minutes of session start (`CLAUDE.md` §How Day 2 should start).

## 1. Deliverable checklist

| Deliverable | Path | Status |
|---|---|---|
| Environment facts (measured) | `ENVIRONMENT.md` | DONE |
| Project handbook / resume instructions | `CLAUDE.md` | DONE |
| Judgment-call log (D1–D10) | `decisions.md` | DONE |
| Pinned dependencies | `requirements.txt` | DONE (torch 2.13.0+cpu) |
| AAAI-27 requirements + compliance checklist | `research/day1/aaai_requirements.md` | DONE (web-verified) |
| Landscape survey + scoop check | `research/day1/landscape.md` | DONE (NOT SCOOPED) |
| ≥4 candidate problems | `research/day1/candidates.md` | DONE (4 candidates) |
| Binding research proposal, zero TBDs | `research/day1/research_proposal.md` | DONE (LOCKED) |
| Smoke test + measured calibration | `research/day1/smoke_test.md` | DONE |
| Data generator | `src/data.py` | DONE |
| Model (5 pluggable PEs, 1L + 2L configs) | `src/model.py` | DONE |
| Training loop (CSV logs, seeds, flags) | `src/train.py` | DONE |
| Day-2 grid launcher (8-worker pool) | `src/launch.py` | DONE (dry-run verified) |
| Test suite | `tests/` — **15 passed** | DONE |
| Git history | 6 commits, one per phase (final commit = this phase) | DONE |

## 2. Locked research problem

**"Positional Encoding as the Hidden Variable in Grokking"** — PE scheme {NoPE, learned, sinusoidal, RoPE, ALiBi} as the controlled variable for (H1) grokking onset/sharpness, (H2) circuit signatures (Fourier concentration, restricted/excluded loss, uniform-attention ablation), (H3) OOD prediction, on mod-113 addition, 1-layer Nanda-style transformer. Scoop check: 6 targeted queries + 3 abstract fetches, 11 nearest neighbors tabulated, none uses PE as the controlled variable for either outcome — **NOT SCOOPED** (landscape.md §2, decisions D5). Key adaptation: circuit identity treated as continuous signatures, not a clock/pizza binary, following the NeurIPS 2025 unification paper (D6).

## 3. Citation status table

| Source | Status |
|---|---|
| AAAI-27 dates/format/review/LLM-policy pages (aaai.org, 5 pages) | verified via search snippets, 2026-07-18 |
| aaai2027.sty author-kit contents | verified via third-party template mirrors, 2026-07-18 |
| AAAI-26 acceptance rate 17.6% (4,167/23,680) | verified, 2026-07-18 |
| Zhong et al. 2306.17844 (Clock & Pizza) | verified (abstract page + NeurIPS PDF listing), 2026-07-18 |
| Moisescu-Pareja et al., NeurIPS 2025 unification | verified (neurips.cc abstract fetched), 2026-07-18 |
| Xu et al. 2407.17963 | verified (abstract fetched), 2026-07-18 |
| 2506.23679 (modular exponentiation) | verified (abstract fetched), 2026-07-18 |
| 2605.20441 (weight-decay regimes, 2026) | verified (arXiv listing + summary), 2026-07-18 |
| Nanda 2301.05217, Power 2201.02177, Kazemnejad 2305.19466, Furuta 2402.16726, 2505.18266, 2606.17399, 2601.09049, 2606.12966, 2602.16746, 2603.23784, 2607.06639, 2511.23443, 2505.13027, 2501.18795, 2512.12167, 2512.25060, 2412.19442, 2503.12491, 2510.00636 | verified as existing (titles+IDs in search results), 2026-07-18 — **re-open before citing in the paper** |
| $300/page purchase, Type-1-font rule, contemporaneous-2-month rule, prompt-injection ethics clause | `[UNVERIFIED — from seed]` — re-verify Day 3 |
| 2605.12697, 2602.14050, 2410.02140, 2402.01032, 2305.07759, 2205.13504, 2201.02177 details | `[UNVERIFIED — from seed]` where not listed above |

## 4. Calibrated Day-2 compute plan (all numbers measured on this machine)

- Throughput: **12.0 epochs/s aggregate** at the optimal config (8 single-thread workers, val eval every 5 epochs); planning basis 11 ep/s.
- **Pass A** (H1+H2): 20 runs (5 PE × seeds 0–3) × ≤15k epochs — worst 7.6 h, expected ~6.5 h.
- **Pass B** (H3): 10 OOD runs (5 PE × seeds 0–1) × ≤15k — worst 3.8 h.
- **Hard commitment A+B: worst 11.4 h ≤ 12 h budget.** Pass C (5th seed, 3rd OOD seed, depth-2, p=97, ops) fills slack/overnight.
- Constraints honored: 8-worker RAM ceiling (12 workers measured to OOM), ≤16 total threads.

## 5. Top 3 risks & pivot plans

1. **Grokking onset shifts beyond the 15k cap for some PEs** (e.g., NoPE/ALiBi may grok much later or never at frac 0.3). *Mitigation:* censoring is itself an H1 result (rank statistics handle it); overnight 40k extension pass reserved for critical censored conditions; if >half of all runs censor, drop frac to 0.4 via a decisions.md entry (30 min re-launch cost).
2. **PE effect too small (null H1/H2)** — all PEs grok alike. *Pivot (pre-authorized, proposal §6):* fallback C2 (NoPE position-circuit formation) reuses every line of infrastructure; the PE grid becomes its control arm; trigger decision by Day-2 hour 6.
3. **Timing drift under long-run conditions** (thermals, background load; calibration used 200–500-epoch runs). *Mitigation:* the hour-by-hour schedule has ~2 h of slack vs worst case; launcher prints live per-job starts; if Pass A pace falls >20% behind at hour 2, cut Pass A to 3 seeds mid-flight (jobs are independent processes — safe to re-plan between jobs).

## 6. Acceptance criteria self-check

- [x] `ENVIRONMENT.md`, `CLAUDE.md`, `decisions.md`, `requirements.txt` — exist, non-trivial
- [x] All six `research/day1/` deliverables present
- [x] `pytest` passes (15/15)
- [x] Smoke test ran; loss 4.91 → 0.006; timing measured; grid extrapolated
- [x] Scoop check documented, 11 named nearest neighbors with differentiation
- [x] `research_proposal.md` contains zero TBDs
- [x] Projected grid fits the measured budget via documented cuts (D10)
- [x] Every citation tagged verified-with-date or `[UNVERIFIED — from seed]`
- [x] Git history: one commit per phase (0–5)
