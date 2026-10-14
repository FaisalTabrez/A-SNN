# October 2026 research restart plan

Current update (2026-10-14): the original two confirmation bundles have been
audited; both return `causal_audit_only`. Their original records remain frozen.
The v2 notebook/runner adds audit guards and new result-folder names. Official
FiTS/fly runs, CUDA diagnosis, independent source-release reproduction and the
reviewed decision lock remain pending. The dated implementation checklist below
is historical, not a claim that the returned audio results are still unrun.

Approved by the user on 2026-10-13: incorporate the literature suggestions and
deliver a runnable, self-contained download/setup/experiment notebook.
This is implementation authorization, not validation of any paper's claims.
Evidence-1 remains the active program until its registered decision is locked.

## Work now

The portable notebook is `gen5/notebooks/evidence1_research_restart.ipynb`.
It embeds SHA-256-checked snapshots of the new runner/adapters and checks out
the prior public A-SNN base at `ee150bd1b0051830f8e3e959805374f4efac9deb` when
no local checkout exists. Local checkouts are preserved; conflicting embedded
files are not overwritten. No Git reset or cloud semantic backend is used.

Implemented scope:

- One validation-selected, resumable SHD/SSC four-arm runner, ten paired seeds
  in confirmation; fixed-checkpoint zero/shuffled state controls; confidence
  intervals; compiled fidelity/throughput and memory recording; hashed ZIP.
- Separate synthetic smoke profile that cannot yield a research decision.
- Dependency installation, optional Colab Drive persistence, official audio
  downloads with archive/raw checksums, result charts and return-ZIP cells.
- Pinned upstream code and open-paper downloads; timing-only Zenodo downloads
  verify metadata checksums and stop at an explicit size budget.
- FiTS native-recipe adapter refuses test-only selection and evaluates test
  only once after validation checkpoint selection. It keeps the upstream
  data/model recipe separate from E1 and supports epoch resumption.
- Fly reference helper uses the author's V630 data and sugar-input examples,
  compares MN9 readout descriptively to retained author examples, and records
  source/data/environment hashes. Its short simulation is not a reproduction
  claim; the full reference profile uses 30 one-second stochastic trials.

Local validation currently covers synthetic CPU execution and contracts.
Official data confirmation, CUDA compilation, FiTS training, full-fly simulation
and independent reproduction require their actual runs; no numerical result
from these unrun stages is implied.

## Notebook instructions

1. Open the notebook in Colab; run default smoke top-to-bottom first.
2. For the registered study choose GPU; set `PROFILE='confirmation'`,
   `CONFIRM_FULL_RUN=True`, preferably `USE_DRIVE=True`; then Run all.
3. Keep the output directory on Drive. Resume by repeating the execution cell
   after reconnecting. Do not change configuration/runtime/source mid-run.
4. Download `research_restart_results.zip` from the last cell or Files sidebar.
   It excludes raw data/papers/checkouts but includes protocol, checkpoints,
   per-seed records, charts and checksums. Return it for evidence review.
5. Reference downloads/runs are opt-in flags. They can add substantial CPU/GPU
   time. E1 performs 80 training fits plus compiler/probe work; no runtime or
   accuracy promise is made.

## Literature-informed queue

The following priorities are our interpretation, not findings about A-SNN.

| Input | Integration decision | Required evidence before expanding claims |
|---|---|---|
| [FiTS](https://arxiv.org/abs/2605.13071), [author code](https://github.com/kaistmm/FiTS) | Download/native validation reference now; matched port after E1 lock | Match binning, capacity, optimizer/search budget, validation and seeds; do not compare published 250-bin/140-channel results directly to E1 |
| [Beyond Rate Coding](https://arxiv.org/abs/2507.16043), [timing-only data](https://zenodo.org/records/16153275) | Post-lock temporal diagnostic | Rate-preserving time reversal/jitter/shuffle and count controls; distinguish total rate from within-sample timing; no automatic reopening of rejected learned delays |
| [Sliding-window readout](https://www.nature.com/articles/s44335-026-00093-1) | Post-lock matched readout ablation | Frozen backbone, capacity-matched pooled versus sliding readout; validation-selected windows, paired seeds; no test-selected window |
| [BulletTrains](https://arxiv.org/abs/2603.13283), [code](https://github.com/ToddMorrill/snn-bullet-trains) | Conditional systems follow-up | Exact supported recurrence, forward/gradient fidelity, reset semantics, end-to-end training time against our compiled implementation |
| [BrainTrace](https://doi.org/10.1038/s41467-026-68453-w), [code](https://github.com/chaobrain/braintrace) | Later learning-method comparator | Match gradient/credit storage, capacity, compute, retention and recurrent stability; do not relabel failed local-credit results |
| [Dual-memory pathways](https://doi.org/10.1038/s42256-026-01255-3), [code](https://github.com/sunpengfei1122/Dual_memory_pathways) | Reproduction/reference only | Distinguish that mechanism from our rejected STW/LTW claim; new preregistered mechanism and memory-matched controls required |
| [AGMP](https://doi.org/10.3389/fnins.2025.1768235), [DendriCL preprint](https://arxiv.org/abs/2607.02283) | Deferred explicit-context mask program | Existing R1 controls: frozen backbone, shared/random masks, shuffled context, replay and regularization; no projected-energy claim |
| [Shiu fly LIF model](https://github.com/philshiu/Drosophila_brain_model) | Author-reference diagnostic now; small fixed topology prior after E1 lock | Known input/readout behavior first, then degree/sign/weight-matched rewiring and equal neuron/synapse counts, paired seeds and causal circuit removal |
| [Male-CNS](https://male-cns.janelia.org/), [Eon embodiment](https://eon.systems/updates/embodied-brain-emulation), [ConnectomeX preprint](https://arxiv.org/abs/2610.08418) | Background only | Anatomy, brain dynamics, learned body controller and vision architecture are different artifacts; none establish A-SNN learning/causal benefits |

## Fly pilot boundary

After the E1 decision lock, preregister one small named circuit/input/readout.
Use fixed connectome adjacency first, then trained weights only in a separate
arm. Compare biological prior, degree-preserving signed rewiring, random sparse
wiring and a capacity-matched dense control; preserve weight/sign distributions
and the same preprocessing, readout, train budget and seeds. Include circuit
removal and input/readout controls. The claim can concern a useful topology
prior, not whole-fly intelligence or validated embodied learning. Do not replace
the working audio runner with a 100k-neuron whole-brain simulation.

## Milestones

1. E1 implementation + notebook smoke (this delivery).
2. Official paired GPU confirmation + separate clean-clone repeat.
3. Review/lock the registered decision, including failures; choose the narrowed
   paper/library/methodology scope the evidence supports.
4. Timing/readout and matched FiTS protocol, then only gated systems/context/fly
   follow-ups. Closed claims stay closed until independent new registered tests.

The notebook is rebuilt with `python gen5/tools/build_research_notebook.py`.
Tests protect snapshot hashes, matching, singleton-safe derangements,
validation-only selection, resumption and decision gating. Graphify indexes
these sources as a retrieval layer; source documents remain authoritative.
