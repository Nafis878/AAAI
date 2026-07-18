# AAAI-27 Submission Requirements

**Prepared:** 2026-07-18 (Day 1). **Verification method:** aaai.org blocks direct page fetches from this environment, so facts were verified via web-search results that quote the official AAAI pages (search snippets opened this session). Each fact is tagged:
- `[VERIFIED]` — confirmed this session via search results quoting the cited official page (access date 2026-07-18).
- `[VERIFIED-3P]` — confirmed via a third-party page (e.g., template mirrors), cited.
- `[UNVERIFIED — from seed]` — from the July-2026 seeded research appendix only; re-verify before relying on it in the final submission.

## 1. Venue and cycle

- 41st AAAI Conference on Artificial Intelligence (AAAI-27), **Montréal, Canada, February 16–23, 2027**. `[VERIFIED]` — aaai.org/conference/aaai/aaai-27/
- Currently open cycle: main technical track, submissions via **OpenReview**. `[VERIFIED]` — aaai.org/conference/aaai/aaai-27/submission-instructions/

## 2. Key dates (all paper deadlines 11:59 PM UTC-12, "anywhere on earth")

| Milestone | Date | Status |
|---|---|---|
| OpenReview opens for author registration | Jun 17, 2026 | `[VERIFIED]` |
| OpenReview opens for paper submission | Jun 30, 2026 | `[VERIFIED]` |
| **Abstracts due** | **Jul 21, 2026** | `[VERIFIED]` |
| **Full papers due** | **Jul 28, 2026** | `[VERIFIED]` |
| Supplementary material + code due | Jul 31, 2026 | `[VERIFIED]` |
| Phase-1 rejection notification | Sep 24, 2026 | `[VERIFIED]` |
| Author feedback window | Oct 19–25, 2026 | `[VERIFIED]` |
| Final notification | Nov 30, 2026 | `[VERIFIED]` |
| Camera-ready due | Dec 14, 2026 | `[VERIFIED]` |

Sources: aaai.org/conference/aaai/aaai-27/, .../submission-instructions/, .../review-process/ (via search snippets, 2026-07-18).

**Project implication:** this 3-day project (Jul 18–20) lands exactly on the Jul 21 abstract deadline, with full paper due Jul 28 and code Jul 31 — the timeline is real but tight.

## 3. Page limits & formatting

- **7 pages of main content maximum; up to 9 pages total, with pages 8–9 exclusively for references.** `[VERIFIED]` — submission-instructions page.
- Up to 2 extra proceedings pages purchasable at $300/page (camera-ready stage). `[UNVERIFIED — from seed]`
- Two-column AAAI Press style; official author kit at aaai.org/authorkit27/ contains `aaai2027.sty`, `aaai2027.bst`, `main.tex`, `AnonymousSubmission2027`, `CameraReady2027`, and `ReproducibilityChecklist.tex`. `[VERIFIED-3P]` — coauthor.thesify.ai/templates/aaai-2027-latex-template (kit contents list); style-file name corrects the seed's `aaai2026.sty`.
- US Letter, PDF only, Type 1 / TrueType fonts (no Type 3). `[UNVERIFIED — from seed]` (standard AAAI kit requirement; confirm from the kit itself on Day 3.)
- Reproducibility checklist is appended **after the references** in the main submission PDF and does **not** count toward the page limit. `[VERIFIED]` — submission-instructions pages.

## 4. Anonymity / double-blind

- Fully double-blind: remove all author/affiliation information; acknowledgements omitted at submission; may substitute paper number + keywords. `[VERIFIED]` — submission-instructions page.
- Use the kit's `AnonymousSubmission2027` variant for the review PDF. `[VERIFIED-3P]`

## 5. Review process & evaluation

- **Two-phase review.** Every paper gets 3 reviewers in Phase 1; sufficiently negative Phase-1 reviews ⇒ rejection without author feedback (notified Sep 24, 2026). Surviving papers proceed to Phase 2 with an author-feedback window (Oct 19–25). `[VERIFIED]` — review-process page.
- Phase-1 human reviews are complemented by a uniquely identified, **AI-generated "Supplementary First-Stage Review"** (non-decisional; humans make all decisions). `[VERIFIED]` — review-process page.
- Reviewers explicitly assess **reproducibility**, and that assessment is weighed in final decisions, alongside the standard criteria (novelty/significance, technical soundness, relevance, clarity). `[VERIFIED]` for reproducibility weighting; criteria list `[UNVERIFIED — from seed]` in exact wording.
- Reference acceptance rate: AAAI-26 accepted 4,167/23,680 = **17.6%** (down from 23.4% at AAAI-25). `[VERIFIED]` — csconfstats.xoveexu.com/conferences/aaai/2026/ and press coverage.
- Papers published < 2 months before the deadline count as contemporaneous work (no comparison required). `[UNVERIFIED — from seed]`

## 6. Reproducibility checklist

`[VERIFIED]` (AAAI-26/27 submission-instructions + AAAI-25/26 reproducibility-checklist pages):
- Completed at submission time; becomes part of the submission and is shared with reviewers.
- Covers, among other items: conceptual outline/pseudocode of introduced methods; documentation of all hyperparameter ranges tried; code for data pre-processing; all source code for conducting and analyzing experiments; and, for randomness-dependent algorithms, the exact seed-setting method enabling replication.
- **Project implication:** our deterministic-by-construction pipeline (explicit seeds everywhere, synthetic data, committed CSV logs) is designed to satisfy this checklist with no retrofitting.

## 7. AI / LLM authorship and use policy

Closely paraphrased from the AAAI Publication Policies & Guidelines and the Policies for AAAI-26 Authors pages (`[VERIFIED]` via search snippets, 2026-07-18; AAAI-27 carries the same publication policy — confirm the AAAI-27-specific policy page wording on Day 3):

1. **No AI authorship:** AI systems (including generative models) do not satisfy the criteria for authorship of papers published by AAAI, because authorship carries accountability, which cannot apply to an AI system.
2. **Not citable as a source:** AI systems also cannot be used as a citable source in papers published by AAAI.
3. **LLM-generated text prohibited in the paper body** unless the produced text is presented as part of the paper's experimental analysis. `[VERIFIED]` — Policies for AAAI-26 Authors.
4. **Editing/polishing of author-written text with LLMs is allowed.** `[VERIFIED]` — same page.
5. **Documentation required:** use of any AI system in developing the publication is allowed only if its role is properly documented in the manuscript. `[VERIFIED]` — AAAI Publication Policies & Guidelines.
6. **Full human responsibility:** authors remain fully responsible for all submitted material; plagiarism, non-existent references, or other violations are sanctionable. `[VERIFIED]`
7. Enumerated ethics violations include prompt-injection attacks on review and LLM-rewritten near-duplicate submissions. `[UNVERIFIED — from seed]`

**Project implication (binding for Day 3):** the human submitter authors the paper; this project's role (AI-assisted research/engineering) must be documented in the manuscript; no LLM-generated body text may be pasted as-is except as experimental artifact; every citation must be a real, verified source.

## 8. COMPLIANCE CHECKLIST — validate the Day-3 paper against this before calling it done

**Format**
- [ ] Uses official AAAI-27 author kit (`aaai2027.sty`, `aaai2027.bst`), two-column, US Letter, PDF
- [ ] ≤ 7 pages main content; references only on pages 8–9; total ≤ 9 pages
- [ ] Type 1/TrueType fonts only (no Type 3); passes the kit's formatting checks
- [ ] Anonymous submission variant: no author names, affiliations, acknowledgements, or identifying links (incl. repo URLs)
- [ ] Reproducibility checklist completed and appended after references

**Content & integrity**
- [ ] Every citation opened and verified (arXiv ID / DOI / URL checked); zero fabricated or "from memory" references
- [ ] No LLM-generated body text except text explicitly presented as experimental artifact
- [ ] AI-use documentation section/statement included in the manuscript
- [ ] No AI system listed as author or cited as a source
- [ ] All reported numbers traceable to committed CSV logs in `experiments/`; seeds and configs stated
- [ ] Claims scoped to what the experiments actually show; limitations section present

**Reproducibility package (due Jul 31, 2026 cycle-equivalent)**
- [ ] Code + deterministic data generation + exact commands to reproduce every figure/table
- [ ] Hyperparameters (including ranges tried) documented; seed policy stated
- [ ] Hardware/runtime disclosure (CPU-only, measured runtimes)

**Process**
- [ ] Abstract-quality title/abstract ready by the abstract-deadline analog (end of Day 3)
- [ ] Contemporaneous-work rule applied correctly when discussing 2026 preprints
