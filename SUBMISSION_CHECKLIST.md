# AAAI-27 Submission Checklist (human-only items)

Owner: the human submitter. Claude cannot register on OpenReview, build with the licensed kit, or take authorship responsibility for prose. Dates are AoE (UTC-12).

## NOW — abstract (due July 21 AoE) — does NOT depend on GPU results
- [ ] Register/log in on OpenReview for AAAI-27; start a Main Technical Track submission.
- [ ] Submit **title + abstract** by **July 21 AoE**. Use the current title and the abstract in `paper/main.tex` (revise the one delayed number if the GPU grid has already reported; otherwise the CPU abstract is accurate as written). **Every submission needs the abstract in by this date or the full paper cannot be submitted.**

## Full paper (due July 28 AoE)
- [ ] Download the official author kit from https://aaai.org/authorkit27/; drop `aaai2027.sty` + `aaai2027.bst` next to `paper/main.tex` (see `paper/README_BUILD.md`).
- [ ] Integrate GPU results per `paper/RESULTS_TODO.md` (or have the integration session do it), then rebuild.
- [ ] Build: `latexmk -pdf main` (or the 4-command sequence in README_BUILD.md). Confirm it compiles with no missing refs.
- [ ] **Page check:** ≤7 pages main content, references on pages 8–9, total ≤9. Trim if over.
- [ ] Fonts: Type-1/TrueType only (kit default); no Type-3.
- [ ] Anonymize: no author names/affiliations/acknowledgements/identifying URLs (incl. the GitHub link — replace with "code will be released" for review).
- [ ] Append the kit's `ReproducibilityChecklist.tex` after references (does not count against the page limit); fill it in.
- [ ] **Substantive human rewrite + ownership of the prose.** AAAI policy prohibits LLM-generated body text except as experimental artifact; the AI-use statement discloses the AI drafting but disclosure does NOT waive the policy. Read, revise, and take responsibility for every sentence before submission. This is a hard requirement, not optional polish.
- [ ] Verify every citation resolves to the correct paper (the 12 in `references.bib` were arXiv-API-verified 2026-07-20, but re-check after any edits).
- [ ] Upload the PDF to OpenReview by **July 28 AoE**.

## Supplementary / code (due July 31 AoE)
- [ ] Prepare an anonymized code release (strip the git history's author identity if needed) or a supplementary zip; reference it as "to be released" in the anonymous submission.
- [ ] Confirm `python -m src.analysis --all` + `python -m src.figures --all` reproduce every reported number/figure from the released runs.

## Later (if accepted)
- [ ] Camera-ready (due Dec 14): de-anonymize, add author/affiliation, acknowledgements, real repo URL.

## Standing integrity reminders
- GPU stats are primary; the 50 CPU runs are reported as independent cross-hardware replication — never pooled.
- Report all pre-registered outcomes including nulls (`research/week/strengthening_plan.md` criteria); no post-hoc relabeling of exploratory findings as confirmatory.
