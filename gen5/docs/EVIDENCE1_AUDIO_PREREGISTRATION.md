# Evidence-1 canonical audio preregistration

Frozen implementation specification: 2026-10-13, before official confirmation.
Original specification is retained below. Update (2026-10-14): two returned
full runs were audited and both return `causal_audit_only`. The decision lock
and independent replication remain pending. v2 adds integrity guards, with
unchanged architecture/training/decision rules, and must use new output folders.
See `EVIDENCE1_HARDENING_2026_10_14.md`; do not edit original run artifacts.
Authority: PRIMARY_EVIDENCE_TRACK_ROADMAP.md and retained Gen-29/30 decisions.

## Registered configuration

- Datasets: complete official SHD and SSC; no sample limits. SHD: 700 channels,
  64 bins across 1.4 seconds, 20 classes; SSC: 700 channels, 64 bins across
  1.0 second, 35 classes. SHD stratified 20% training holdout, split seed 2026;
  SSC official validation. Final tests are not selection data.
- Arms: residual temporal-convolution LIF, direct Conv1D, dilated TCN, GRU.
  Trainable-parameter ceiling 133,631; each model must use at least 95% of it.
  All parameters are dense/active; no sparse-compute accounting is asserted.
- Seeds 310–319, paired between architectures. Identical per-epoch sample order
  across arms; seeded initializations are not identical tensors across different
  architectures. Initial parameter hashes are recorded.
- AdamW: learning rate 0.003, weight decay 0.0001, batch 256, 15 fixed epochs,
  no early stopping, earliest highest validation accuracy checkpoint.
- Temporal pyramid levels 1,2,4,8; convolution input kernel 5; TCN hidden
  kernel 3/dilation 2; LIF surrogate slope 10. Existing model dynamics remain
  unchanged. No per-arm tuning budget or SSM implementation is added.
- Frozen residual-checkpoint interventions: full; zero both state trace and
  final state (direct-only); global sample-specific state derangement repeated
  three times, partner seeds `training_seed + 80000 + repeat_index`.
  Preserve each sample's direct path. Shuffle may cross minibatches and has no
  fixed points, including a singleton final batch. No ablation is retrained.
- Raw files and encoded caches have SHA-256 provenance; notebook download
  checks official gzip MD5 and raw-file equality. Model/loader sources, Python,
  Torch and execution device are locked for resumption. A new source/runtime,
  protocol or dataset requires a new results directory. Do not load third-party
  pickle/checkpoint files into this runner.

## Gates and reports

Each estimate reports per-seed records, mean, sample standard deviation and
two-sided 95% paired Student-t interval. Marginal intervals are not a
familywise-confidence claim; no exploratory selection is allowed. Every
registered condition must pass (intersection decision).

1. Causal gate: on each dataset the lower bound of full-minus-zero and
   full-minus-average-shuffle accuracy must exceed zero.
2. Accuracy gate: on each dataset lower bound of full-minus-each-dense-control
   must exceed -0.01 (one absolute percentage point). This is a preregistered
   noninferiority rule, not a claim of strict superiority or two-sided equivalence.
3. Systems gate: CUDA fullgraph compilation must have exactly identical
   validation predictions, max absolute logit error <= 1e-4, and speedup > 1 on
   every residual checkpoint. Median of three timing repeats, 20 iterations
   after five warmups, batch 256 (or available validation size). Record first
   shape compile latency, train/inference CUDA peak allocated memory, hardware
   and per-model throughput. Tail shapes are checked for fidelity but their
   recompilation is not included in the first-shape compile latency field.
4. Independent clean-clone repetition of the same protocol and an aggregate
   reviewed decision lock. The runner always marks this pending; it cannot
   certify its own independent replication.

Failing causal gate: stop causal claim. Passing causal/failing accuracy:
causal-audit-only. Passing model gates/failing or missing compiler gate:
systems unconfirmed. Passing all automated gates: replication still pending.
The active program does not change without reviewed decision lock. Energy,
online learning, biological fidelity and hardware-sparse execution are untested.

## Execution and outputs

Use `gen5/notebooks/evidence1_research_restart.ipynb`, or:

```text
python gen5/examples/evidence1_audio.py --profile confirmation --device cuda --data-root research_restart/data --output-dir research_restart/evidence1_confirmation
```

Repeat the command to resume. Epoch checkpoints, selected checkpoint history,
protocol/data/environment metadata, per-trial records, CSV, confidence/gate
summary and checksummed ZIP are retained. An interrupted epoch restarts from
the last complete checkpoint with its deterministic epoch seed/order.

The default smoke profile is two epochs, one seed and synthetic 12-channel,
three-class data. It is only a workflow test and cannot produce an eligible
confirmation decision. Modifying the confirmation dataclass also makes a run
ineligible for the registered decision. Test-derived tuning would require a
new holdout/protocol, not rerunning this one and selecting the favorable result.
