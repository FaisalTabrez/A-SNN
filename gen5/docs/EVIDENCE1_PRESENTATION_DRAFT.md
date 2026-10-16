# Evidence-1 presentation draft artifact

Presentation revision (2026-10-15): the current deck now covers the full
program and matches the IEEE paper. See
[A_SNN_FULL_PROGRAM_PRESENTATION.md](A_SNN_FULL_PROGRAM_PRESENTATION.md).
The earlier audio-focused artifacts below remain historical versions.

Manuscript revision (2026-10-15): the current paper is now the full-program
IEEE university presentation manuscript described in
[A_SNN_IEEE_PRESENTATION_PAPER.md](A_SNN_IEEE_PRESENTATION_PAPER.md).
It uses the standard two-column IEEEtran conference template and covers all
research approaches. The files below remain the previous audio-focused version
and its audit; the slide deck was not revised in this manuscript-only update.

Audit qualification (2026-10-14): include the correction text in
`EVIDENCE1_HARDENING_2026_10_14.md` when using the original paper/slides.
Historical official-test exposure limits untouched-holdout claims, and global
state exchange does not isolate within-class identity or spike-specific coding.
The original numerical values and failed gates remain unchanged. Qualified
copies now exist as `output/pdf/evidence1_paper_draft/evidence1_research_paper_audited.pdf`
and `output/presentations/evidence1/A_SNN_Evidence1_Research_Presentation_Audited.pptx`.
The paper adds audit front matter; the deck revises interpretation wording and
disclosures while preserving original native chart/workbook parts.

Prepared on 2026-10-14 at the user's request from the retained confirmation
and same-seed repeat bundles. This note records an artifact, not an active
program change or a formal research decision lock.

The manuscript source is
`output/pdf/evidence1_paper_draft/evidence1_research_paper_draft.tex`.
The same directory contains numerical evidence, four presentation figures,
speaking notes and a reproducible JSON/manifest audit helper.

The audit verifies the supplied outer-manifest hashes, all 80 unique trial
keys per execution, per-trial record equality, shared protocol/data provenance,
and recomputed Student-t statistics and registered gate values. It does not
load checkpoint pickles, perform new training or certify external execution
independence.

Primary residual-LIF means are 84.06360424% SHD and 56.47875576% SSC.
Full-minus-zero effects are 9.43462898 and 13.63752330 percentage points;
full-minus-shuffled effects are 4.04004711 and 8.86713767 points. All four
paired 95% intervals have positive lower bounds. The registered accuracy gate
fails on both benchmarks. The systems gate also fails because SHD seed 310
and SSC seed 311 exceed the 1e-4 maximum-logit-error tolerance, despite
identical validation class predictions. Both bundles return
`causal_audit_only`.

The two executions use the same ten seed identifiers; they must not be pooled
as twenty independent observations. Independent clean-clone provenance and a
human-reviewed decision lock remain pending. The current program stays
Evidence-1. No competitive-superiority, energy-efficiency, whole-fly,
biological-fidelity or online-learning finding is authorized by this draft.

Author names and departments were supplied by the user. Authorship order,
institutional wording, mentor review and venue-specific requirements remain
pre-submission checks. The primary evidence roadmap and audio preregistration
remain authoritative; Graphify is only a derived retrieval layer.
