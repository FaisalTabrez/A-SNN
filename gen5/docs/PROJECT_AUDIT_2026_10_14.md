# A-SNN research and implementation audit

Audit date: 2026-10-14. Scope: retained project work through Gen-30, the
Evidence-1 restart, current experiment archives, reproducibility workflow,
and the paper/presentation artifacts.

This is an audit report, not a new experimental result or a decision lock.
Evidence-1 remains active. No stopped mechanism is reopened. No production
code was repaired, no official GPU training was repeated, and no commit or
push was performed as part of this audit.

## Executive assessment

The current Evidence-1 archives are internally consistent. Their checksums,
trial keys, source fingerprints, statistics, and narrowed decision all pass
an independent read-only audit. The conclusion remains `causal_audit_only`:
state interventions pass, while accuracy and compiled systems gates fail.

However, the project is not yet an independently reproducible research
release. There are confirmed runner guard loopholes, missing historical raw
evidence, stale claims, exploratory reuse of official benchmark tests, and
an incomplete public source/environment release. Passing existing tests did
not detect several of these cases.

These findings do not demonstrate fabricated results, corrupted supplied
checkpoints, train/test-row overlap, or invalid current arithmetic. They do
limit stronger claims and identify concrete work before new training or
publication.

## Scope and checks completed

| Check | Outcome | Boundary |
| --- | --- | --- |
| Every root-level result ZIP | 51 archives, 321,921,812 compressed bytes, no CRC errors or duplicate members | Archive integrity, not scientific correctness |
| Evidence-1 primary/repeat outer manifests | All 170 / 168 manifested files verified | Self-contained checksums do not certify external independence |
| Trial identity and per-trial JSON | Exactly 80 expected unique keys in each execution, records agree with aggregate payload | Checkpoint tensors were not loaded |
| Current model/loader source provenance | All 73 package source fingerprints match both archives | Does not release the source to GitHub |
| Recomputed mean, sample SD, paired Student-t intervals | Maximum discrepancy from archived statistics 7.28e-13 | Conditional on these fixed datasets and seed observations |
| Paper numerical supplement | Recomputed statistics match exactly | New causal/novelty claims need separate justification |
| PowerPoint chart data | All three native charts match recomputed values rounded to two decimals | No native PowerPoint application execution in this audit |
| Full Gen-5 unit suite | 261 tests passed in 20.687 seconds, Python 3.13.2 / Torch 2.6.0+cpu | CPU contracts do not certify CUDA fidelity |
| Existing Evidence-1 tests in a second CPU runtime | 10 tests passed, Torch 2.14.0+cpu | Newly discovered edge cases are not covered |
| Portable notebook smoke | All 12 code cells, eight synthetic records, checksummed ZIP, `smoke_only` | Current local environment, not a clean install or optional reference run |
| Notebook copies | Byte-identical, SHA-256 9b113cdabf695096edd6264259fcc88d3baeb90121786910b8b843febcfd9712 | Snapshot equality does not prove release completeness |
| Historical manifest sample | Nine manifests, 42 named artifacts present, all 36 available SHA-256 checks passed | Gen-21's six names have no hashes |
| Graphify acceptance contract | Passed, 4,251 nodes at audit start | Weak anchors do not establish complete coverage |
| Canonical graph source census | 389 of 391 inventoried files represented at audit start | Two source files absent from graph node provenance |

Historical inventory includes 85 `gen5/docs` files, 70 retained `analysis.md`
files, 103 retained output directories, all nine Sprint42-50 runners, and all
25 Gen6-30 runners. Roadmaps, claims, decisive analyses, source contracts and
targeted implementations received substantive review. This is not a claim
that every historical source line was manually inspected or every experiment
was re-executed.

Machine-readable checks are retained in
`output/audits/2026_10_14/bundle_audit.json` and `archive_integrity.json`.
The independent audit scripts are under `tmp/audit_2026_10_14/`.

## Findings and required work

Priority meanings: P1 affects research/release readiness or can falsely
certify a gate. P2 is a confirmed guard, provenance or methodological gap.
P3 is lower-risk documentation/pinning maintenance. Priorities are not claims
that the supplied experiments actually encountered each failure.

### F01. P1: nonfinite outputs can falsely satisfy numerical fidelity

Source: [benchmark comparison](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:295>)
and [evaluation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:192>).

The maximum-error accumulator uses `max(previous, float(error))`. With a NaN
error, `max(0.0, NaN)` returns 0.0. A CPU mock of the compiled call returning
`[NaN, 0]` against eager `[1, 0]` produced `status=measured`, class agreement
1.0, and maximum logit difference 0.0. With a speedup above one this can pass
the present systems gate. Evaluation also accepts all-NaN logits and reports
an ordinary accuracy when activity remains finite.

Required: finite checks on eager/compiled logits and their differences before
argmax or reductions, finite evaluation activity/metrics, terminal failure
records, and regression tests. Consider post-update parameter/gradient checks.

Existing-result impact: no evidence of NaNs in supplied metrics. Their two
reported fidelity errors are finite. This probe does not explain their CUDA
root cause or prove that the current result changed.

### F02. P2: aggregate decision fails open with unexpected records

Source: [summary eligibility and decision](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:325>).

Completeness tests only whether every expected key exists. An extra seed makes
a dataset summary get skipped. Eligibility does not require every registered
dataset summary, and `all()` over an empty summary collection is true. A
synthetic input with 80 failing expected records plus two unexpected-seed
records returned `model_and_systems_gates_pass_replication_pending`,
`complete=True`, `datasets={}`.

Required: exact key-set equality, duplicate/extra-key rejection, strict
schema/range/finiteness checks, and equality between summarized and registered
dataset sets before any decision.

Existing-result impact: the normal fresh canonical runner emits 80 slots;
this is an imported/merged/direct-summary loophole. Both supplied archives
contain exactly the expected unique keys and are unaffected by this case.

### F03. P2: completed-record resumption silently trusts changed artifacts

Source: [completed-record fast path](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:416>)
and [bundle generation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:363>).

After a legitimate synthetic run, a record's accuracy was changed from
0.210526... to 0.99999, its initial hash became `not-a-hash`, and its checkpoint
was replaced with an invalid placeholder. Resumption accepted the changed
record and regenerated a manifest over the invalid checkpoint. The output
protocol and source/config/data locks still matched.

Required: authenticate a per-trial record/checkpoint binding, validate schema
and expected seed/arm/dataset, verify original digests before skipping work,
and fail on unauthenticated nonempty output folders. A freshly regenerated
manifest is not authentication of prior contents.

Existing-result impact: a reproduced accidental-edit/recovery loophole,
not evidence of editing in the supplied ZIPs. Self-consistent current
manifests alone cannot rule out changes before manifest regeneration.

### F04. P2: checkpoint seed and original initialization are not bound

Source: [checkpoint load](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:219>)
and [checkpoint save/metadata](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:256>).

A completed seed-310 checkpoint was accepted by `train()` for seed 311.
The loaded best state stayed identical, while the returned initial-parameter
hash described the newly initialized seed-311 model rather than the run that
produced that checkpoint.

Required: save/check seed, arm, dataset, protocol/data identity and original
initialization hash inside the checkpoint. Resume must return stored initial
provenance, not a new model's unused initialization.

Existing-result impact: needs a misplaced/copied checkpoint or helper misuse;
no such event was found in the supplied records. Ordinary same-seed resume
tests pass.

### F05. P2: a bound encoded cache can change before a new run accepts it

Source: [cache binding](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:151>)
and [SHD cache load](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/shd_benchmark.py:621>).

The cache-source binding records raw and encoder identities, not the completed
encoded tensor digests. A tiny HDF5/cache reproduction changed encoded events
to zero and called the actual loader again. The altered cache was accepted
with unchanged raw hashes and changed cache hashes.

Required: atomically bind completed cache digests to raw/encoder identities,
validate them before reuse, and reject unexpected/changed cache contents.

Existing-result impact: a fresh cache build derives data normally. A
same-output resume rejects changed metadata. A new output folder reusing the
already bound cache accepts it. No proof of incorrect encoding in the actual
returned data exists; matching observed hashes are provenance, not an
independent reconstruction of raw-to-tensor correctness.

### F06. P1: Evidence-1 is not a source-complete GitHub release

Local HEAD and live GitHub main both remain
`ee150bd1b0051830f8e3e959805374f4efac9deb`, the Obsidian-removal commit.
The Evidence-1 runner, CLI, preregistration, tests, notebook, builder and
reference helpers are still untracked. The notebook intentionally reconstructs
four embedded files on that old base. It can be used, but a Git-only clone
cannot reproduce the complete current source/test tree.

Source: [notebook base and snapshots](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/build_research_notebook.py:85>)
plus audit-time `git status`, `git ls-files`, and `git ls-remote origin`.

Required: after review and authorized fixes, commit/release the exact source,
tests, frozen specification and environment. Verify a fresh checkout, pin the
notebook to that release, and separately record a reviewed decision. Do not
silently turn this audit into permission to push unrelated changes.

### F07. P2: dependency and automation coverage are incomplete

Source: [notebook setup](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/build_research_notebook.py:65>)
and [runner environment lock](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:386>).

There is no tracked requirements/lock/pyproject/setup manifest or GitHub CI
workflow. Broad installation ranges and an existing Torch preserve usability,
but do not reconstruct the archived Python 3.13.15 / Torch 2.11.0+cu130 / CUDA
13.0 / NVIDIA L4 runtime. Resume signatures omit the full package freeze,
driver and cuDNN identities.

Required: a tested environment definition and frozen dependency receipt,
driver/cuDNN/backend flags, clean-install notebook smoke, CPU contracts and
snapshot verification in CI, and a separately bounded CUDA diagnostic.
Adding guard tests has priority over additional accuracy sweeps.

Lower-priority distinction: upstream Git references are commit-pinned, but
arXiv PDF URLs omit revision suffixes and only record a hash after download.
These papers have observed provenance, not immutable revision pinning.

### F08. P2: advertised local-Jupyter setup can fail before installation

Source: [Colab detection](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/build_research_notebook.py:61>).

`find_spec('google.colab')` raises `ModuleNotFoundError` if the parent `google`
namespace is absent. This was reproduced in isolated Python. Setup fails
before installing dependencies on an otherwise valid clean local machine.
The successful local smoke already had that namespace and masks the defect.

Required: guarded parent/module detection and an isolated non-Colab setup
test. Rebuild both notebook copies after the authorized source fix.

### F09. P2: optional reference helpers can adopt unbound old output

Source: [FiTS resume](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/fits_validation_reference.py:49>),
[FiTS protocol creation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/fits_validation_reference.py:119>),
and [fly execution](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/fly_reference.py:48>).

FiTS can create a missing protocol lock and then accept an existing summary
or checkpoint. Fly creates a missing lock and calls upstream with
`force_overwrite=False`, which skips already-present parquet outputs. A reused
unbound folder can acquire misleading current provenance.

Required: reject unauthenticated nonempty output folders or verify an existing
trusted artifact manifest before adoption. Bind summaries/checkpoints/output
parquets to their original protocol and runtime.

This is source-verified against the pinned upstream implementations, not a
claim that a FiTS/fly result was corrupted. Neither optional reference has a
completed research result here. FiTS's validation-only selection and one final
test evaluation are correctly wired.

### F10. P1: historical benchmark test reuse needs explicit disclosure

Source: [Phase41 redesign](<C:/Users/FAISAL TABREZ/Documents/A-SNN/research.md:4314>),
[Phase43-44 progression](<C:/Users/FAISAL TABREZ/Documents/A-SNN/research.md:4398>),
and [residual/SSC progression](<C:/Users/FAISAL TABREZ/Documents/A-SNN/research.md:4521>).

Observed official SHD/SSC test results repeatedly influenced subsequent
architecture choices. Validation-only checkpoint selection protects epoch
selection in the current runner, but does not undo project-level exploratory
test exposure. The same benchmark test sets are therefore not untouched
external holdouts for a newly chosen architecture.

Required: disclose the exploratory history in the manuscript/presentation,
describe current intervals as fixed-protocol seed summaries on known
benchmarks, preserve the frozen decision, and use a genuinely untouched
external dataset only under a later preregistered protocol.

This is not evidence of overlapping training/test rows. It does not invalidate
the arithmetic or automatically erase the within-model intervention result.

### F11. P2: current controls do not isolate within-class identity or spikes

Source: [residual interventions](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:96>)
and [global derangement](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evidence1_audio.py:183>).

Global shuffling generally changes class as well as sample. Zero-state deletes
both spike-trace and analog final-membrane features. There is no within-class
shuffle, analog-state comparator, separate spike/final-state deletion, or
independently retrained same-feature branch-deletion control in Evidence-1.

Supported wording: the trained hybrid model depends on its original
sample-state pairing. Unique identity beyond class, timing-specific coding,
spiking dynamics versus analog state, and retrained architecture necessity
remain unisolated. Parameter matching alone also does not optimize each
baseline or match FLOPs, especially given the weak SHD GRU recipe.

Required: clarify the wording now. After the decision lock, preregister the
specific additional controls relevant to the next claim. Do not retroactively
add them to the frozen confirmation or select favorable test outcomes.

Three shuffles correctly average within each seed, and intervals use ten
seed-level observations. No shuffle-repetition pseudoreplication was found.

### F12. P2: early structural-plasticity proof was multi-factor and unmatched

Source: [early proof wording and capacity](<C:/Users/FAISAL TABREZ/Documents/A-SNN/research.md:443>),
[ablation parameters](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/evaluation.py:891>),
and [historical proven label](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/outputs/phase11_uploaded_outputs_review_2026-06-26.md:704>).

The static arm had eight edges, full plasticity averaged 85.95, and gated
plasticity 67. The full arm also changed pruning, sprouting and LTW noise
together. The observed difference does not isolate structural plasticity
under matched capacity or update budgets.

Historical trained BPTT/PPO comparisons remain deferred. The dense LIF/MLP
entries were inference scaffolds, and PPO was skipped. AMMC's best population
fitness of 20 versus dense LIF's four must not hide means of -0.0589 versus
+0.125 or become an average policy-superiority claim.

Required: narrow current-facing wording to a multi-factor evolutionary-package
comparison. Later matched failures remain authoritative. Do not spend compute
trying to rescue the old claim unless a separate new study is authorized.

### F13. P2: four historical branches lack their canonical raw evidence

| Branch | Retained evidence | Still missing |
| --- | --- | --- |
| Gen19 | Explicit aggregate terminal-log recovery | Original per-seed/progress/CSV/plot/bundle |
| N-MNIST accuracy detour | Log recovery with reported three-seed statistics | Canonical JSON/CSVs/checkpoint/timing manifest and bundle |
| Gen26 | Numerical-fidelity analysis text | Raw diagnostic JSON/CSV/manifest/bundle |
| Gen30 | Ten-seed console summary and stop analysis | Original per-seed records and checksummed bundle |

Sources: [Gen19 limitation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/GEN19_NMNIST_STATE_REPLICATION_ANALYSIS.md:7>),
[N-MNIST limitation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/NMNIST_ACCURACY_BENCHMARK_ANALYSIS.md:51>),
[Gen26 analysis](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/GEN26_SPARSE_NUMERICAL_FIDELITY_ANALYSIS.md:5>),
and [Gen30 limitation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/GEN30_DENDRITIC_PREDICTIVE_CREDIT_ANALYSIS.md:11>).

Required: recover originals from the retained Drive runs if available, verify
their hashes and recompute reported metrics. Otherwise keep explicit
reported/log-only labels. No per-seed evidence should be invented. The
N-MNIST 99.4767% result belongs to a conventional CNN, not an SNN record.
Missing provenance does not authorize reopening their stop decisions.

### F14. P1/P2: Gen29's complete status is a static ledger, not metric validation

Source: [source map](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/gen29_program_closure.py:14>)
and [claim generation](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/ammc_gen5/gen29_program_closure.py:68>).

Phase47, 48 and 49 all point to `MILESTONE_ROADMAP.md`, not their actual
metric-bearing evidence. Claims cite Gen18/19 but those sources are absent
from the hashed source map. Twelve logical entries represent only ten unique
paths. File existence/hash checks precede copying hard-coded conclusions and
unconditionally declaring complete. Empty placeholder source documents can
therefore satisfy the implementation's source requirement.

Required: map claims to actual analyses and immutable result records, include
all cited sources, verify metrics/gates rather than copying static statuses,
and distinguish generated synthesis from independent validation. Existing
available manifests passing does not remedy this semantic weakness.

### F15. P2/P3: current summaries conflict with later evidence

Examples include research.md's October restart section still saying official
confirmation/CUDA results have not arrived, the architecture/Gen5 README's
biological-proof/production/continuous-learning language, the milestone
roadmap's Gen30-not-authorized status despite its later stop, and earlier
dual-memory interpretations without a visible later algebraic supersession.
Five old `outputs/...` research references also resolve only under
`gen5/outputs/...` after relocation.

Sources: [restart status](<C:/Users/FAISAL TABREZ/Documents/A-SNN/research.md:24>),
[architecture wording](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/GEN5_ARCHITECTURE.md:5>),
[Gen5 README](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/README.md:3>),
and [milestone status](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/docs/MILESTONE_ROADMAP.md:3>).

Required: add a concise current evidence status and supersession links while
preserving date-stamped history. Engineering mechanisms are implementations,
not proof of biological fidelity or autonomous lifelong learning. Repair the
five relocated links without changing historical measurements.

### F16. P2: Graphify's passing contract misses coverage and freshness gaps

Source: [acceptance anchors](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/verify_graphify_contract.py:15>)
and [pass rule](<C:/Users/FAISAL TABREZ/Documents/A-SNN/gen5/tools/verify_graphify_contract.py:69>).

At audit start, `LITERATURE_UNIQUENESS_REVIEW.md` and
`SNN_PROJECT_INFERENCES_2026-08-09.md` had manifest entries but no represented
graph source node. The gate checks three directory prefixes and three research
strings, not comprehensive per-file hashes, actual source provenance, dangling
relationships, source-line freshness or semantic extraction success.

Required: a full canonical-corpus coverage/hash contract and explicit
classification of heading-only versus richer semantic extraction. The
authoritative files still exist. Graphify remains a retrieval layer. No cloud
semantic backend is authorized by this audit, and a local structural update
must not be called a complete semantic refresh.

## Existing Evidence-1 conclusion and artifact review

Both supplied runs contain 80 complete fits. The primary residual means are
84.0636% SHD and 56.4788% SSC. Full-minus-zero effects are 9.4346 and 13.6375
percentage points. Full-minus-shuffle effects are 4.0400 and 8.8671 points.
All four paired intervals have positive lower bounds.

The registered accuracy gate fails on both datasets. In particular, SSC
residual-minus-TCN is -1.9620 points, 95% interval [-2.8219, -1.1022]. SHD
versus Conv1D has a positive mean but an interval lower bound below the -1
point margin, so failing that noninferiority test does not prove inferiority.

Systems fidelity fails at SHD seed310 (0.000315189 maximum logit error) and
SSC seed311 (0.097311020), versus the fixed tolerance 0.0001. All class
predictions agree, and 18 of 20 residual checkpoints pass the logit tolerance
in each execution. These speedups compare compilation with the same model's
eager inference, not with dense baselines or energy consumption.

The repeat uses the same ten seed IDs and splits. There are not twenty
independent observations. Train-time fields differ for all 80 trials and no
checkpoint archive member is byte-identical across runs. These facts are
compatible with rerunning the study, but do not certify independent execution.
Formal clean-clone replication and human-reviewed decision lock remain pending.

The 10-page paper and 17-slide deck preserve the failed gates and bounded
interpretation. Their numerical supplement/native chart values match the
recomputed data. Before presentation/submission, add historical test-exposure
disclosure and clarify original sample-state pairing versus within-class
identity/timing specificity. Broader novelty, authorship contributions, mentor
review, venue requirements and any AI-assistance disclosure are still human
submission checks. No paper/slide files were edited in this audit.

## Post-Sprint41 work ledger

| Work | Status supported by retained sources |
| --- | --- |
| 42 initialization robustness | Completed, sparse advantage fails |
| 43 validation-selected checkpoint audit | Completed, sparse branch closes |
| 44 calibrated temporal baselines | Completed, Conv1D becomes useful target |
| 45 spiking temporal convolution | Completed, replacement/state placement fails |
| 46 residual placement | Completed, feature preservation recovers accuracy |
| 47 SHD state intervention audit | Completed, cooperative state contribution |
| 48 SSC state replication | Completed, cooperative contribution repeats |
| 49 dense-baseline/efficiency audit | Completed, TCN wins under that protocol |
| 50 evidence synthesis | Implemented and retained, not a new independent experiment |
| Submitted Phase51 duplicate | Previously identified duplicate Phase49 evidence, not a distinct experiment |
| Milestone A, Gen6-8 | Stop outcomes, no qualified promoted causal architecture |
| Gen9-13 | Tested adaptation/memory/local-rule branches stop |
| Gen14 | Stop, followed by reward-protocol redesign |
| Gen15-16 | Narrow reward-protocol/exact-analog-gradient passes |
| Gen17-18 | Translation/held-out robustness fails, reliable local-credit claim closes |
| Gen19 / N-MNIST accuracy detour | Reported outcomes, canonical raw evidence pending |
| Gen20 | Screen stops, downstream confirmation/causal stages intentionally not run |
| Gen21 | Narrow result subsequently limited by Gen22's memory algebra |
| Gen22-23 | Stops, tested dual-memory rules close |
| Gen24 | Narrow compiled-systems pass for its frozen workload |
| Gen25 | Generic COO fails |
| Gen26 | Reported diagnostic stop, raw provenance missing |
| Gen27 | Scope-limited sparse behavioral-semantic pass |
| Gen28 | Triton throughput/fidelity fails |
| Gen29 | Historical static synthesis, source/validation flaw described above |
| Gen30 | Reported stop, canonical bundle pending |
| Evidence-1 | Two audited runs, narrowed decision, reviewed lock/release still pending |

Stopped screening followed by absent confirmation rows is proper gate
behavior, not unfinished experimentation. The old Obsidian specification is
superseded by the user's Graphify-only instruction; its absence is intentional.

## Already corrected and intentionally deferred work

Do not count these as new defects requiring another rescue sweep:

- Sprint16 action-opportunity confound received Sprint17 action controls.
- Champion eager/compiled topology mismatch received a fingerprinted 86-edge
  same-topology comparison.
- Phase40/41 initialization/readout RNG confounds received Phase42 controls
  and Phase43 closure.
- Gen22's algebraic STW/LTW relabeling received the boundary-only Gen23
  redesign, which also stopped.
- Gen28's reused compiled-output buffer issue received a clone regression
  guard. The same retained-buffer problem was not demonstrated in Evidence-1,
  which consumes its eager/compiled outputs before the next compiled call.

Timing/rate-preserving controls, matched FiTS/readout studies, explicit-context
supermask routing, fly topology priors, energy measurement and hardware-sparse
systems remain gated/deferred. The first four follow the Evidence-1 lock;
they are not completed findings. Whole-fly simulation is not the next audio
training backend. No accepted CUDA sparse-speed or biological-learning claim
can be inferred merely from implemented scaffolds.

## Recommended order of work

1. Harden finite-output, record/schema, checkpoint identity, cache and
   aggregate-decision guards. Add reproductions as regression tests. Label this
   as a new implementation version; preserve original records and protocol.
2. Repair clean-Jupyter detection and reject unbound optional-reference output.
   Rebuild notebooks and test clean setup paths with no cached Google namespace.
3. Reconcile current documentation and historical supersession labels. Add the
   test-exposure/pairing limitations to the paper and speaking notes.
4. Recover Gen19/N-MNIST/Gen26/Gen30 raw bundles where available and replace
   static Gen29 attestation with evidence-driven claim/source validation.
5. Publish an authorized exact-source/environment release with CPU CI and a
   clean-checkout reproduction record. Then obtain the reviewed Evidence-1
   decision lock. Do not require every originally ambitious gate to pass in
   order to lock a narrowed negative result.
6. Diagnose the two finite compiler discrepancies with frozen validation
   checkpoints and intermediate tensors. No full retraining is needed for
   the initial diagnostic. Actual CUDA investigation remains pending.
7. Only after the lock, register the specific next methodology/control study.
   Use an untouched external dataset for a later generalization claim and
   matched baseline tuning if competitive performance becomes the question.

Most immediate work is CPU/source/documentation work, not another 80-fit GPU
run. Do not loosen the original tolerance or select a new accuracy margin
after looking at these outcomes.

## Verification limits and external technical references

No GPU was available locally. Original `.pt` files from user archives were
not loaded, and official-data training, CUDA fidelity/throughput, real FiTS,
Brian2 fly simulation and external replication were not repeated. CRC/hashes
and internally matching summaries establish consistency, not independent
authenticity or exact raw-to-tensor re-encoding.

PyTorch does not promise bitwise identical floating-point results across
implementations, versions or platforms. That general limitation does not
relax our frozen tolerance or identify the cause of these particular errors.
See [PyTorch 2.11 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).

Pinned reference code checked for optional helper semantics:
[FiTS training](https://raw.githubusercontent.com/kaistmm/FiTS/8fe7e376bae13fe75f357dcfbff6adf9c45f1077/common/training.py)
and [fly model](https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py).

This audit creates reports and isolated diagnostics only. Repairs, publication,
new experiments and GitHub changes remain separate authorized work.
