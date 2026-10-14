# Evidence-1 audit hardening and remaining work

The v2 implementation fixes the locally reproducible guard defects in the
[2026-10-14 audit](PROJECT_AUDIT_2026_10_14.md). It does not replace the original
registered results or issue the human-reviewed decision lock.

## Fixed locally

- Nonfinite logits, activity, fidelity differences, training gradients and
  updated parameters fail closed. A compiler numerical failure cannot masquerade
  as zero logit error.
- Aggregate summaries reject duplicate/unexpected keys, nonfinite metrics and
  incomplete dataset coverage before any passing research decision.
- Resumed trial records and checkpoint bytes require integrity receipts.
  Checkpoints bind their original initialization, seed, model and protocol/data
  context. Completed records receive schema, range, hash and identity checks.
- Completed encoded caches bind their digests to raw files and encoder sources.
  Changed, deleted or unbound caches cannot be adopted by a new run.
- FiTS/fly reject unbound output and verify completed/checkpoint/output receipts.
  FiTS remains a native reference, not a matched Evidence-1 comparison.
- Notebook Colab detection handles a missing parent namespace. Both copies are
  rebuilt from source; runtime dependency/driver/cuDNN/backend receipts are saved.
- A tested tooling requirements file, package metadata, CPU CI definition and
  source/snapshot release checker are provided. CI execution on GitHub requires
  a separately authorized commit/push.
- Gen29 now links the actual Phase47-49 analyses and Gen18/19. It rejects empty
  placeholder sources and labels itself a historical source-integrity synthesis,
  never an independently validated complete research certificate.
- Current summaries qualify broader/biological/continual-learning claims,
  acknowledge historical benchmark-test exposure and later memory supersession,
  and repair the relocated early evidence links.
- The stronger Graphify corpus contract checks every canonical source, dangling
  edges and graph/source hashes. Markdown reviews classified as papers receive
  AST-only supplementation through Graphify's own libraries. This is structural
  recall repair, not semantic validation.

Receipts detect accidental changes and wrong-run reuse. They are not signed
authenticity proofs. An adversary able to rewrite artifacts and receipts can
still fabricate a self-consistent folder. A torn file/receipt update fails
closed rather than guessing how to recover it. Preserve that folder; do not
manually regenerate a receipt to bless unverified data.

## Preserve existing results and data

Do not run v2 inside the original result folders. Source/runtime/schema locks
deliberately reject them. The notebook uses `evidence1_v2_smoke` and
`evidence1_v2_confirmation` output folders. Do not rerun the 80-fit study merely
to obtain new receipts. Original ZIPs and their frozen protocol remain the
authority for the reported observations.

Old encoded caches do not receive automatic v2 certification. To use v2 later,
choose a fresh dataset directory and copy the verified raw `.h5` files into its
`shd/` and `ssc/` folders. Let the encoder create new bound caches. Raw datasets
do not need another download. Preserve the old caches and results unchanged.

Local CPU setup:

```text
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research.txt
python -m unittest discover -s gen5/tests -v
python gen5/tools/verify_research_notebook.py --work-dir tmp/notebook_v2_smoke
python gen5/tools/verify_research_release.py
```

Install the intended CUDA Torch build separately for future GPU diagnostics.
The CPU requirements are a tested tooling baseline, not a retroactive lock of
the original Colab Python 3.13.15/Torch 2.11.0+cu130 environment. Actual future
runtime versions and backend settings enter the protocol signature. Changing
them requires a new result folder.

## Still pending, not repaired by code

Publication/closure implementation update: the user has now authorized GitHub
publication. The [closure runbook](EVIDENCE1_CLOSURE_RUNBOOK.md) adds an exact
original-source package, publicly reviewable result metadata/recomputation,
validation-only CUDA diagnostic notebook, clean-checkout software verifier,
historical recovery scanner and dormant follow-up controls. These implement the
missing tooling, not the external review, GPU execution or missing raw evidence.

| Audit area | What remains |
| --- | --- |
| F06 public release | Authorized source/metadata publication and local clean-checkout verification are being executed; independent research execution remains separate |
| F07 original runtime | Full historic package/driver receipt cannot be reconstructed honestly from the limited returned metadata |
| F10 test exposure | Historical official-test reuse cannot be undone; disclose it and use a new untouched dataset only under a later protocol |
| F11 mechanism isolation | Within-class/component control primitives are implemented; retrained controls and new preregistered experiments still require the lock |
| F12 early plasticity | Capacity/noise/topology confounds cannot be retroactively repaired into proof; the claim is narrowed |
| F13 raw artifacts | Recover Gen19, N-MNIST accuracy, Gen26 and Gen30 originals from Drive or retain explicit log/analysis-only status |
| F14 full historical metric validation | Gen29 no longer falsely certifies it; raw/gate recomputation across every history remains separate work |
| Compiler discrepancy | Diagnose finite errors for SHD310 and SSC311 with frozen validation checkpoints on actual CUDA |
| Decision | Mentor/human review and formal Evidence-1 lock remain pending |
| Semantic graph | No approved semantic backend, so no complete semantic-freshness claim |

The accepted claim concerns dependence on the original sample-state pairing
inside the selected hybrid. Global shuffles usually change class too, and
zeroing deletes both spikes and analog final membrane. Unique identity beyond
class, timing-specific coding and a biological learning mechanism remain
unisolated. The original accuracy and systems failures remain in the paper.

## Verified local checks

The full CPU suite passes, including numerical, resume/cache, Gen29, Graphify
and adapter-specific regressions. The regenerated notebook executes all twelve
code cells offline and returns eight synthetic trials with `smoke_only`.
The distribution builds as a wheel. These checks are local; GitHub CI and CUDA
execution are not claimed.

## Paper/presentation correction text

The following qualification accompanies the revised presentation and the
paper's added audit front matter:

"SHD and SSC official test results informed earlier exploratory architecture
choices. The present fixed-protocol confidence intervals summarize training-seed
variation on these known benchmarks, not a previously untouched external
holdout. The intervention result establishes dependence on the original
sample-state pairing. Global derangement does not isolate identity within a
class, and jointly deleting spike traces and final membrane does not isolate
spikes from analog state. No numerical result or registered margin changed."

The original presentation/PDF files are preserved. Qualified copies are:

- `output/pdf/evidence1_paper_draft/evidence1_research_paper_audited.pdf`:
  one audit-qualification page followed by all ten original pages unchanged.
- `output/presentations/evidence1/A_SNN_Evidence1_Research_Presentation_Audited.pptx`:
  revised claim wording, visible audit limits and speaker-note disclosures.
  All three original native charts and their workbooks are preserved byte-for-byte.

All seventeen revised slides were visually checked. Package, native chart and
layout validators pass. No numerical result, failed gate or registered margin
was changed.
