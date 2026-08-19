# A-SNN / AMMC

Adaptive spiking-neural-network research workspace for the AMMC line: a
browser-proven biological sandbox plus a headless Gen-5 tensor framework for
large-scale embodied evolution.

The short version:

- `index.html` is the Gen-4 browser sandbox for visual inspection.
- `gen5/` is the Python framework for headless tensor evolution and benchmarks.
- `research.md` is the living project memory and decision log.
- `gen5/outputs/` contains retained experiment evidence.

## Start here

If you are new to the repo, read these in order:

1. [Repository structure](docs/PROJECT_STRUCTURE.md)
2. [Gen-5 architecture](gen5/docs/GEN5_ARCHITECTURE.md)
3. [Current research log](research.md)
4. [Gen-5 README](gen5/README.md)
5. [Phase 11 Colab runbook](gen5/docs/PHASE11_COLAB_RUNBOOK.md)
6. [Harder bot-world benchmarks](gen5/docs/HARDER_WORLDS.md)
7. [Frozen embodied readout benchmark](gen5/docs/FROZEN_EMBODIED_ADAPTER.md)
8. [Embodied action controls](gen5/docs/EMBODIED_ACTION_CONTROLS.md)
9. [Frozen event-coded MNIST](gen5/docs/EVENT_MNIST.md)
10. [Event representation decomposition](gen5/docs/EVENT_REPRESENTATION_DECOMPOSITION.md)
11. [Temporal-state MNIST](gen5/docs/TEMPORAL_STATE_MNIST.md)
12. [Fixed-topology LTW training](gen5/docs/TRAINABLE_TEMPORAL_MNIST.md)
13. [LTW optimization diagnostic](gen5/docs/LTW_OPTIMIZATION_DIAGNOSTIC.md)
14. [Causal recurrence ablation](gen5/docs/RECURRENCE_ABLATION.md)
15. [Streaming row-sequential MNIST](gen5/docs/SEQUENTIAL_MNIST.md)
16. [Sequential LTW training](gen5/docs/TRAINABLE_SEQUENTIAL_MNIST.md)
17. [Targeted sequential synaptogenesis](gen5/docs/STRUCTURAL_SEQUENTIAL_MNIST.md)
18. [Adaptive-neuron sequential ablation](gen5/docs/ADAPTIVE_SEQUENTIAL_MNIST.md)
19. [Executable-delay sequential ablation](gen5/docs/DELAYED_SEQUENTIAL_MNIST.md)
20. [Trainable delay assignment](gen5/docs/TRAINABLE_DELAYS_MNIST.md)
21. [SHD temporal-pyramid readout](gen5/docs/SHD_TEMPORAL_PYRAMID.md)
22. [SHD temporal-control decomposition](gen5/docs/SHD_TEMPORAL_CONTROLS.md)
23. [SHD matched baselines](gen5/docs/SHD_MATCHED_BASELINES.md)
24. [Gen-6 successor preregistration and result](gen5/docs/GEN6_SUCCESSOR_PREREGISTRATION.md)
25. [Gen-7 predictive-state preregistration](gen5/docs/GEN7_PREDICTIVE_STATE_PREREGISTRATION.md)
26. [Gen-8 time-local binding preregistration](gen5/docs/GEN8_TEMPORAL_BINDING_PREREGISTRATION.md)
27. [Gen-9 continual-adaptation preregistration](gen5/docs/GEN9_CONTINUAL_ADAPTATION_PREREGISTRATION.md)
28. [Gen-10 robust-representation preregistration](gen5/docs/GEN10_ROBUST_REPRESENTATION_PREREGISTRATION.md)
29. [Gen-11 plastic-adapter preregistration](gen5/docs/GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md)
30. [Gen-11 plastic-adapter analysis](gen5/docs/GEN11_PLASTIC_ADAPTER_ANALYSIS.md)
31. [Gen-12 associative-memory preregistration](gen5/docs/GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md)
32. [Gen-12 associative-memory analysis](gen5/docs/GEN12_ASSOCIATIVE_MEMORY_ANALYSIS.md)
33. [Gen-13 local-plasticity preregistration](gen5/docs/GEN13_LOCAL_PLASTICITY_PREREGISTRATION.md)
34. [Gen-13 local-plasticity analysis](gen5/docs/GEN13_LOCAL_PLASTICITY_ANALYSIS.md)
35. [Continual-adaptation program closeout](gen5/docs/CONTINUAL_ADAPTATION_PROGRAM_CLOSEOUT.md)
36. [Gen-14 reward-eligibility preregistration](gen5/docs/GEN14_REWARD_ELIGIBILITY_PREREGISTRATION.md)
37. [Gen-14 reward-eligibility analysis](gen5/docs/GEN14_REWARD_ELIGIBILITY_ANALYSIS.md)
38. [Research evidence freeze](gen5/docs/RESEARCH_EVIDENCE_FREEZE.md)
39. [Gen-15 matched reward-baseline preregistration](gen5/docs/GEN15_REWARD_BASELINE_PREREGISTRATION.md)
40. [Gen-15 matched reward-baseline analysis](gen5/docs/GEN15_REWARD_BASELINE_ANALYSIS.md)
41. [Gen-16 local score-credit preregistration](gen5/docs/GEN16_LOCAL_SCORE_CREDIT_PREREGISTRATION.md)
42. [Gen-16 local score-credit analysis](gen5/docs/GEN16_LOCAL_SCORE_CREDIT_ANALYSIS.md)
43. [Gen-17 sparse-spiking credit preregistration](gen5/docs/GEN17_SPARSE_SPIKING_CREDIT_PREREGISTRATION.md)
44. [Gen-17 sparse-spiking credit analysis](gen5/docs/GEN17_SPARSE_SPIKING_CREDIT_ANALYSIS.md)
45. [Gen-18 held-out local-credit preregistration](gen5/docs/GEN18_LOCAL_CREDIT_REPLICATION_PREREGISTRATION.md)
46. [Gen-18 held-out local-credit analysis](gen5/docs/GEN18_LOCAL_CREDIT_REPLICATION_ANALYSIS.md)
47. [Gen-19 N-MNIST state-replication preregistration](gen5/docs/GEN19_NMNIST_STATE_REPLICATION_PREREGISTRATION.md)
48. [Gen-19 N-MNIST state-replication analysis](gen5/docs/GEN19_NMNIST_STATE_REPLICATION_ANALYSIS.md)
49. [Full-resolution N-MNIST accuracy benchmark](gen5/docs/NMNIST_ACCURACY_BENCHMARK_PREREGISTRATION.md)

## Repository map

```text
.
|-- index.html                 # Gen-4 browser sandbox / visual connectome lab
|-- research.md                # living findings, decisions, and next steps
|-- docs/                      # newcomer navigation and repo-level docs
|-- assets/design/             # concept and UI reference images
`-- gen5/
    |-- ammc_gen5/             # Python package: sparse brains, environment, evolution
    |-- examples/              # runnable sprint/evaluation scripts
    |-- benchmarks/            # throughput and baseline comparisons
    |-- tests/                 # unit/contract tests
    |-- tools/                 # verification utilities
    |-- docs/                  # Gen-5 technical runbooks and architecture notes
    `-- outputs/               # retained experiment bundles and analyses
```

## Current research baseline

For the first 2D bot-world cycle, sparse-efficiency tuning is now frozen:

- Default raw-survival baseline: `low_ltw_pruning`, `32` neurons.
- Sparse-efficiency baseline: `gentle_ltw_scheduled`, `32` neurons.
- Phase 17 activity-matched controls validate frozen-trace action decoding.
- Phase 20 established that full temporal AMMC state is informative.
- Phase 22 stabilized LTW training but found no practically meaningful gain.
- Phase 23 showed that 256 recurrent edges add only `+0.107` accuracy points
  to the full linear representation and reduce MLP accuracy by `0.053` points.
- Phase 24 showed that recurrence adds `+11.673` linear and `+17.240` MLP
  points when input rows must be remembered from final hidden state.
- Phase 25 showed that all-edge LTW learning adds `+2.113` linear and `+0.893`
  MLP points, with useful weight movement concentrated in sensor projections.
- Phase 26 found a conditional `+0.767`-point linear gain from 48 random sensor
  sprouts, but no mean MLP benefit; recurrent growth was weaker.
- Phase 27 rejected one-shot absolute-gradient sprouting: every guided arm
  lost to paired random growth on every seed, although peripheral pruning
  recovered part of the deficit without touching the recurrent core.
- Phase 28 rejected the tested adaptive-threshold rule: accuracy declined as
  adaptive coverage increased, with stable LTWs but progressively suppressed
  event flow.
- Phase 29 produced the strongest conventional temporal-mechanism result so
  far: heterogeneous recurrent delays gained `+8.087` linear and `+7.593` MLP
  points across all seeds with the same sparse topology.
- Phase 34 established reproducible SHD capacity scaling to `60.615%` at 512
  hidden neurons, though parameter efficiency declined with width.
- Phase 35 reproduced the 512/no-delay baseline at `60.704%` and rejected a
  robust cross-capacity delay effect; delays also cost roughly 40% throughput.
- Phase 36 raises the 512-neuron mean to `80.065%`, beating global pooling by
  `+19.287` points and fixed-shuffled timing by `+6.257` points.
- Phase 37 finds only `+0.648` points from recurrence over feedforward AMMC;
  the tested recurrent topology fails its causal gate.
- Gen-7 learned strongly future-aligned state but failed identity/order
  specificity. Gen-8's analog binder gained replicated order sensitivity,
  while the paired local LIF collapsed during screening and identity remained
  non-causal. Gen-9 validated a sensor-damage adaptation task, but predictive
  LIF missed the source-competence gate by `6.467` points. Gen-9 returned
  `stop`; memory mechanisms and hardware optimization remain closed.

The final machine-readable claim ledger and report are retained in
[`gen5/outputs/gen9_research_closeout_2026-08-10/`](gen5/outputs/gen9_research_closeout_2026-08-10/).

See [research.md](research.md) for the evidence trail.

The focused post-Phase-26 literature interpretation is documented in
[SNN project inferences](gen5/docs/SNN_PROJECT_INFERENCES_2026-08-09.md).

## Browser sandbox quick start

From the repository root, serve the static sandbox:

```powershell
python -m http.server 4173
```

Then open:

```text
http://127.0.0.1:4173/
```

The root `index.html` remains intentionally stable because it is the visual QA
entry point for connectome import/export and Gen-5 champion replay.

## Gen-5 quick validation

From the repository root:

```powershell
python -m compileall gen5
python -m unittest discover -s gen5\tests -v
```

Torch-dependent tests require a Python environment with PyTorch installed.

## Common Gen-5 commands

List sparse-efficiency groups:

```powershell
python gen5/examples/sprint13_sparse_efficiency_ablation.py --list-groups
```

Run a small local smoke test:

```powershell
python gen5/examples/sprint1_smoke.py
python gen5/examples/sprint4_5_vectorized_loop.py
```

Run the Colab/XLA throughput benchmark on a TPU runtime:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

Run the harder-world benchmark on Colab CUDA/T4:

```python
!python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds simple moving_toxins delayed_reward gauntlet \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/harder_worlds_cuda
```

Run the Phase 18 event-coded MNIST benchmark:

```python
!python gen5/examples/sprint18_event_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_cuda
```

Diagnose the Phase 18 representation loss:

```python
!python gen5/examples/sprint19_event_representation_decomposition.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_decomposition_cuda
```

Preserve the full temporal neuron state:

```python
!python gen5/examples/sprint20_temporal_state_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_temporal_cuda
```

Train active LTWs while keeping the sparse topology fixed:

```python
!python gen5/examples/sprint21_trainable_temporal_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_temporal_mnist_cuda
```

Diagnose LTW schedule, rate, slope, and edge scope:

```python
!python gen5/examples/sprint22_ltw_optimization_diagnostic.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/ltw_optimization_diagnostic_cuda
```

Test recurrence against paired feedforward expansion:

```python
!python gen5/examples/sprint23_recurrence_ablation.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/recurrence_ablation_cuda
```

Test final-state memory on row-sequential MNIST:

```python
!python gen5/examples/sprint24_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/sequential_mnist_cuda
```

Train LTWs on the fixed recurrent sequential topology:

```python
!python gen5/examples/sprint25_trainable_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_sequential_mnist_cuda
```

Test targeted synaptogenesis on the sequential topology:

```python
!python gen5/examples/sprint26_structural_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/structural_sequential_mnist_cuda
```

Test gradient-ranked growth and conservative peripheral pruning:

```python
!python gen5/examples/sprint27_utility_gated_structural_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --scoring-batches 4 \
  --prune-after-epochs 3 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/utility_gated_structural_mnist_cuda
```

Test fixed adaptive thresholds on the proven sequential topology:

```python
!python gen5/examples/sprint28_adaptive_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --adaptation-decay 0.95 \
  --adaptation-strength 0.5 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/adaptive_sequential_mnist_cuda
```

Test executable recurrent delay buckets:

```python
!python gen5/examples/sprint29_delayed_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/delayed_sequential_mnist_cuda
```

Optimize delay assignments without changing topology:

```python
!python gen5/examples/sprint30_trainable_delays_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --delay-learning-rate 0.003 \
  --entropy-regularization 0.001 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_delays_mnist_cuda
```

Transfer the retained fixed heterogeneous delays to the official Spiking
Heidelberg Digits dataset:

```python
!pip -q install h5py
!python gen5/examples/sprint31_shd_benchmark.py \
  --device cuda \
  --seeds 42 43 44 \
  --timesteps 64 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_benchmark_cuda
```

The first run downloads and MD5-verifies the official `shd_train.h5.gz` and
`shd_test.h5.gz` files, then caches deterministic 64-bin uint8 event tensors in
`gen5_data/shd`. Later runs reuse that cache. A quick plumbing screen can use
`--seeds 42 --train-samples 1000 --test-samples 500 --epochs 2`; it is not the
registered result. Dataset specifications and licensing are maintained by the
[Zenke Lab SHD resource](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

Diagnose the sparse SHD representation after the fixed-delay transfer run:

```python
!python gen5/examples/sprint32_shd_representation.py \
  --device cuda \
  --seeds 42 43 44 \
  --timesteps 64 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_representation_cuda
```

This registered diagnostic separates four hypotheses: linear-readout
underfitting, delay transfer under an MLP readout, insufficient hidden capacity,
and excessive firing activity. It reuses the cached official SHD tensors.

Validate the no-delay capacity curve and locate its efficiency knee:

```python
!python gen5/examples/sprint34_shd_capacity_scaling.py \
  --device cuda \
  --seeds 42 43 44 \
  --hidden-neuron-counts 128 192 256 384 512 \
  --delay-anchor 256 \
  --timesteps 64 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_capacity_cuda
```

The primary comparison is no-delay 256 versus no-delay 128 neurons. The sole
delay arm is a paired 256-neuron falsification control; the remaining scales
measure accuracy, activity, parameter efficiency, and throughput without delay
overhead.

Test whether heterogeneous delays interact reproducibly with hidden capacity:

```python
!python gen5/examples/sprint35_shd_delay_interaction.py \
  --device cuda \
  --seeds 42 43 44 \
  --hidden-neuron-counts 256 512 \
  --timesteps 64 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_delay_interaction_cuda
```

At each width, this runs paired no-delay, uniform-delay, hash-heterogeneous,
and distance-heterogeneous arms. A heterogeneous pattern must gain at least two
mean points with at least two one-point seed gains and stable dynamics to pass.

Phase 35 did not pass across widths. Test whether preserving temporal order at
the readout is the missing bottleneck:

```python
!python gen5/examples/sprint36_shd_temporal_pyramid.py \
  --device cuda \
  --seeds 42 43 44 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --projection-dim 32 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_temporal_pyramid_cuda
```

The pyramid arms preserve coarse timing at four scales while matching the
global MLP's readout parameter budget. The 512-neuron shuffled-time arm has the
same graph and decoder shape, so only ordered-over-shuffled improvement counts
as evidence that natural SHD timing is causally useful.

Phase 36 passed both gates. Run the causal readout/reservoir decomposition:

```python
!python gen5/examples/sprint37_shd_temporal_controls.py \
  --device cuda \
  --seeds 42 43 44 \
  --hidden-neurons 512 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --projection-dim 32 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_temporal_controls_cuda
```

This compares event counts, a parameter-matched temporal model over raw events,
global AMMC pooling, feedforward AMMC temporal features, and recurrent AMMC
temporal features. It determines whether the new 80.1% result belongs mainly
to the readout or also requires recurrent sparse computation.

Phase 37 found no practical random-recurrence advantage. Run the matched
standard-model comparison:

```python
!python gen5/examples/sprint38_shd_matched_baselines.py \
  --device cuda \
  --seeds 42 43 44 \
  --sparse-hidden-neurons 512 \
  --dense-lif-hidden-neurons 128 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_matched_baselines_cuda
```

Phase 38 passed the matched dense-LIF gate, but feedforward and recurrent sparse
models were nearly tied. Run the causal sparse-mechanism ablation:

```python
!python gen5/examples/sprint39_shd_sparse_mechanisms.py \
  --device cuda \
  --seeds 42 43 44 \
  --hidden-neurons 512 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_sparse_mechanisms_cuda
```

This keeps the temporal decoder and sparse sensor graph fixed while crossing
hard LIF versus analog leaky dynamics with frozen versus trainable LTWs. It is
the mechanism checkpoint before any structural-plasticity or broader claims.

Phase 39 falsified both the hard-spiking and LTW-learning hypotheses. Run the
analog/topology decomposition before deciding whether the sparse expansion is
a useful architecture or merely an allocation artifact:

```python
!python gen5/examples/sprint40_shd_analog_topology.py \
  --device cuda \
  --seeds 42 43 44 \
  --sparse-hidden-neurons 512 \
  --dense-hidden-neurons 128 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_analog_topology_cuda
```

This compares dense LIF, dense analog feedforward/recurrent, and sparse analog
instant/leaky models under the same effective parameter budget.

Phase 40 identified sparse width plus temporal leak as the surviving mechanism.
Run the fixed-budget width scaling experiment:

```python
!python gen5/examples/sprint41_shd_sparse_width.py \
  --device cuda \
  --seeds 42 43 44 \
  --widths 128 256 512 1024 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_sparse_width_cuda
```

Every sparse arm keeps 700 frozen sensor edges and the same total effective
parameter target. The output reports connected-node occupancy and fan-in so
width gains cannot be confused with simply allocating more parameters.

Phase 41 found strong width scaling up to 512 nodes but did not reproduce the
Phase 40 absolute advantage. Separate topology and readout/optimization seeds:

```python
!python gen5/examples/sprint42_shd_initialization_robustness.py \
  --device cuda \
  --topology-seeds 42 43 44 \
  --readout-seeds 142 143 144 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_initialization_robustness_cuda
```

This runs three raw readout seeds and a 3x3 topology/readout seed matrix for
both 512 and 1024 sparse nodes, explicitly reseeding the final readout after
topology construction.

Phase 42 found no robust sparse advantage. Run the final validation-selected
audit before freezing the SHD sparse branch:

```python
!python gen5/examples/sprint43_shd_validation_checkpoint.py \
  --device cuda \
  --topology-seeds 42 43 44 \
  --readout-seeds 142 143 144 \
  --validation-fraction 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_validation_checkpoint_cuda
```

The runner uses a fixed stratified validation split and reports both final-epoch
and best-validation test accuracy. Sparse topology and LTWs remain frozen.

Phase 43 closed the current sparse SHD branch. Calibrate the reproducible raw
temporal decoder against matched conventional temporal baselines:

```python
!python gen5/examples/sprint44_shd_calibrated_baselines.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --validation-fraction 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_calibrated_baselines_cuda
```

This compares validation-selected raw temporal pyramid, temporal Conv1D, GRU,
and dense recurrent LIF models at approximately 133,631 trainable parameters.

Phase 44 establishes temporal Conv1D as the calibrated target. Test whether its
learned local temporal filters can be retained in a leaky spiking core:

```python
!python gen5/examples/sprint45_shd_spiking_temporal_conv.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --validation-fraction 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_spiking_temporal_conv_cuda
```

The five matched arms separate the raw temporal decoder, conventional Conv1D,
leaky analog convolution, leaky LIF convolution, and dense recurrent LIF. The
primary gate is whether the convolutional LIF comes within two test points of
both analog temporal controls while maintaining non-degenerate spike activity.

Phase 45 shows that replacing direct Conv1D features with temporal state causes
the main loss before spiking. Diagnose whether preserving the direct features
beside the state recovers performance:

```python
!python gen5/examples/sprint46_shd_state_placement_diagnostic.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --validation-fraction 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_state_placement_diagnostic_cuda
```

This is a branch-closing diagnostic, not a threshold sweep. Residual analog and
LIF arms retain pooled ReLU convolution features beside their state traces.
They must recover at least four points over the corresponding state-only arm
and finish within two points of Conv1D to justify another mechanism phase.

Phase 46 passes the recovery gate, but the direct path could explain the whole
gain. Ablate each trained residual model without retraining to test whether its
state trace carries sample-specific information:

```python
!python gen5/examples/sprint47_shd_residual_state_contribution.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --validation-fraction 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_residual_state_contribution_cuda
```

The runner evaluates each best-validation checkpoint in full, direct-only,
state-only, and batch-shuffled-state modes. A credible state contribution must
lose at least one mean accuracy point when the state is removed and when its
sample identity is shuffled, with both effects replicated on two seeds.

Phase 47 establishes a causal SHD state contribution. Replicate the fixed
protocol on the official 35-class Spiking Speech Commands train/validation/test
splits:

```python
!python gen5/examples/sprint48_ssc_residual_lif_replication.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/ssc_residual_lif_replication_cuda
```

The first full run downloads roughly 1.6 GB of compressed official data and
creates uint8 temporal caches. It uses all 75,466 training, 9,981 validation,
and 20,382 test samples by default. `--train-samples`, `--validation-samples`,
and `--test-samples` are available only for explicit screening runs; publishable
replication evidence must use the complete official splits.

Phase 48 replicates the causal state effect on SSC. Audit predictive and
computational competitiveness against a matched two-layer dilated TCN:

```python
!python gen5/examples/sprint49_ssc_efficiency_baselines.py \
  --device cuda \
  --readout-seeds 142 143 144 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/ssc_efficiency_baselines_cuda
```

This uses the Phase 48 cache and compares matched Conv1D, dilated TCN, and
residual LIF checkpoints. It reports accuracy, measured T4 throughput, dense
MAC proxies, state updates, and spike-event counts. MACs are architecture
proxies—not hardware energy measurements—and the current dense PyTorch
implementation is not event-driven.

Phase 49 closes the empirical SSC audit. Generate the final machine-readable
claim ledger, milestone report, and summary figure from committed evidence:

```powershell
python gen5/examples/sprint50_evidence_synthesis.py `
  --evidence-root gen5/outputs `
  --output-dir gen5/outputs/gen5_evidence_synthesis_2026-08-10
```

The synthesis distinguishes supported mechanism claims from rejected
standalone, predictive-superiority, and software-efficiency claims. It also
defines the next work as compiled event-driven execution, accuracy scaling,
non-audio replication, and only then continual-plasticity reintegration.

## Evidence discipline

Experiment outputs kept in the repository should live under `gen5/outputs/`
and include an `analysis.md` whenever possible. New research decisions should
update [research.md](research.md) in the same change set.

The repo is deliberately part lab notebook, part framework. The code tells us
what can run; the evidence folders tell us what we have actually observed.

## Consolidated milestone workflow

The numbered experiment sequence ended with the Phase 50 evidence synthesis.
New work follows the three decision milestones in
[gen5/docs/MILESTONE_ROADMAP.md](gen5/docs/MILESTONE_ROADMAP.md): architecture,
hardware, and generalization/continual learning. A milestone performs its own
cheap screen, promotes only qualifying arms, runs confirmation and causal
controls, and emits an explicit stop/go decision.

Milestone A can be launched in one Colab cell after pulling the latest commit:

```python
!python gen5/examples/milestone_a_architecture.py \
  --device cuda \
  --screen-seed 142 \
  --confirm-seeds 142 143 144 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 4 \
  --confirm-epochs 15 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/milestone_a_architecture_cuda
```

The cached official SSC tensors from Phases 48–49 are reused. The runner first
tests all five arms on reduced subsets, then automatically runs the full
official train/validation/test confirmation only for the best conventional
control and qualifying causal candidates. Upload the resulting output folder
or ZIP; there is no separate screen-result round trip. Progress is atomically
checkpointed after every arm/seed run in `milestone_a_progress.json`. Repeating
the identical command resumes completed work after a Colab interruption.

### Milestone A result

Milestone A returned `stop`: only the dilated TCN passed screening and no
causal LIF arm qualified for confirmation. The current Gen-5 architecture
branch is closed, and the hardware and continual-learning milestones are
deferred rather than run against an architecture that failed its predictive
gate. The retained contribution is the narrower causal residual-state finding
on SHD and SSC. See the committed
[`analysis.md`](gen5/outputs/milestone_a_architecture_cuda_2026-08-10/analysis.md)
and final architecture-closeout evidence report.

## Gen-6 successor experiment

Gen-6 is a separately preregistered successor, not another Gen-5 rescue sweep.
It keeps the complete dilated-TCN direct predictor and adds a zero-initialized,
weight-shared residual state correction. The LIF arm adds only 99 trainable
values at the default width and cannot perturb TCN logits at initialization.
The full hypothesis and terminal gates are frozen in
[gen5/docs/GEN6_SUCCESSOR_PREREGISTRATION.md](gen5/docs/GEN6_SUCCESSOR_PREREGISTRATION.md).

Run the consolidated screen and automatic confirmation in Colab:

```python
!python gen5/examples/gen6_successor.py \
  --device cuda \
  --screen-seed 142 \
  --confirm-seeds 142 143 144 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 4 \
  --confirm-epochs 15 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen6_successor_cuda
```

The runner atomically checkpoints after every arm/seed pair. Repeating the
identical command resumes from `gen6_successor_progress.json`. It either emits
a qualified LIF successor and reopens hardware work, or permanently closes
this successor without a follow-up sweep.

## Gen-7 predictive-state experiment (completed: `stop`)

Gen-6 preserved TCN accuracy but its state correction was not beneficially
sample-specific. Gen-7 therefore gives state an explicit temporal objective:
early multi-timescale LIF activity predicts later encoder features, while a
zero-initialized sample-conditioned gate controls the residual correction.
Correctly paired targets are tested against identical no-predictive and
shuffled-target LIF controls. The frozen protocol is documented in
[gen5/docs/GEN7_PREDICTIVE_STATE_PREREGISTRATION.md](gen5/docs/GEN7_PREDICTIVE_STATE_PREREGISTRATION.md).

Run the consolidated Colab experiment:

```python
!python gen5/examples/gen7_predictive_state.py \
  --device cuda \
  --screen-seed 142 \
  --confirm-seeds 142 143 144 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 4 \
  --confirm-epochs 15 \
  --target-parameters 133631 \
  --timesteps 64 \
  --future-horizon 4 \
  --contrastive-temperature 0.10 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen7_predictive_state_cuda
```

The run checkpoints after every arm/seed pair and automatically retains the
no-predictive and shuffled-target controls whenever the paired LIF candidate
reaches confirmation.

Gen-7 completed with paired predictive LIF `+0.417` mean point over TCN and a
strong `0.2928` future-alignment margin. It nevertheless failed the terminal
identity/order gate: shuffled state improved accuracy by `1.022` points and
time reversal cost only `0.165` point. The result and final ledger are retained
in [`gen5/outputs/gen7_predictive_state_cuda_2026-08-10/`](gen5/outputs/gen7_predictive_state_cuda_2026-08-10/)
and [`gen5/outputs/gen7_research_closeout_2026-08-10/`](gen5/outputs/gen7_research_closeout_2026-08-10/).

## Gen-8 time-local predictive binding (completed: `stop`)

Gen-8 tests whether Gen-7's failure came from pooling away sample/time identity
before state affected the output. Its candidate predicts each aligned future
timestep and computes a zero-initialized class correction from
`direct[t] * state[t]` before temporal aggregation. A pooled Gen-7 reference,
an analog control, shuffled-target LIF, and the matched TCN run in the same
resumable screen/confirmation protocol. The complete frozen contract is in
[gen5/docs/GEN8_TEMPORAL_BINDING_PREREGISTRATION.md](gen5/docs/GEN8_TEMPORAL_BINDING_PREREGISTRATION.md).

Colab execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen8_temporal_binding.py \
  --device cuda \
  --screen-seed 145 \
  --confirm-seeds 145 146 147 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 4 \
  --confirm-epochs 15 \
  --target-parameters 133631 \
  --timesteps 64 \
  --future-horizon 4 \
  --contrastive-temperature 0.10 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen8_temporal_binding_cuda
```

The runner checkpoints after every arm/seed pair. Re-running the same command
resumes from `gen8_temporal_binding_progress.json`.

Gen-8 completed with the paired local LIF candidate failing screening at
`7.267%` validation accuracy and `50.656%` spike activity. The analog local
binder remained within `0.456` TCN point and showed a replicated `0.561`-point
time-reversal cost, but state shuffling cost only `0.118` point with 0/3 seeds
passing. The result and ten-source closeout are retained in
[`gen5/outputs/gen8_temporal_binding_cuda_2026-08-10/`](gen5/outputs/gen8_temporal_binding_cuda_2026-08-10/)
and [`gen5/outputs/gen8_research_closeout_2026-08-10/`](gen5/outputs/gen8_research_closeout_2026-08-10/).

## Gen-9 continual adaptation (completed: `stop`)

Gen-9 begins a new task-level program rather than rescuing the closed static
classifier branch. A fixed seed removes 35% of SSC sensor channels after source
training. TCN and predictive-LIF representations then adapt sequentially from
`0, 64, 256, 1024, 4096` validation-stream examples while the runner measures
damaged accuracy, source retention, activity, adaptation time, and throughput.
The frozen protocol is in
[gen5/docs/GEN9_CONTINUAL_ADAPTATION_PREREGISTRATION.md](gen5/docs/GEN9_CONTINUAL_ADAPTATION_PREREGISTRATION.md).

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen9_continual_adaptation.py \
  --device cuda \
  --screen-seed 148 \
  --confirm-seeds 148 149 150 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 4 \
  --confirm-epochs 15 \
  --adaptation-budgets 0 64 256 1024 4096 \
  --adaptation-epochs-per-block 3 \
  --adaptation-learning-rate 0.001 \
  --damage-fraction 0.35 \
  --damage-seed 909 \
  --target-parameters 133631 \
  --timesteps 64 \
  --future-horizon 4 \
  --contrastive-temperature 0.10 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen9_continual_adaptation_cuda
```

The runner resumes from `gen9_continual_adaptation_progress.json`. A pass opens
STW/LTW memory; a stop prevents automatic memory, replay, or plasticity work.

Gen-9 completed with a valid 9.364-point sensor-damage shift. Frozen-readout
adaptation recovered 5.601 points and full fine-tuning recovered 8.462 points,
but predictive LIF screened at 25.100% validation accuracy versus 31.567% for
TCN. Its healthy 8.964% spike rate shows that the failure was source competence,
not silent activity. Only TCN was promoted; the terminal decision is `stop`
with zero qualified arms. Results and the eleven-source evidence closeout are
retained in
[`gen5/outputs/gen9_continual_adaptation_cuda_2026-08-10/`](gen5/outputs/gen9_continual_adaptation_cuda_2026-08-10/)
and
[`gen5/outputs/gen9_research_closeout_2026-08-10/`](gen5/outputs/gen9_research_closeout_2026-08-10/).

## Gen-10 masked-sensor representation reset (completed: `stop`)

Gen-10 tests a new representation rather than tuning the failed Gen-9 pooled
predictive LIF. Random sensor masking trains a residual analog or LIF state to
align with the clean direct representation. Ordinary and sensor-dropout TCNs
control for architecture and augmentation effects. The frozen protocol is in
[gen5/docs/GEN10_ROBUST_REPRESENTATION_PREREGISTRATION.md](gen5/docs/GEN10_ROBUST_REPRESENTATION_PREREGISTRATION.md).

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen10_robust_representation.py \
  --device cuda \
  --screen-seed 151 \
  --confirm-seeds 151 152 153 \
  --screen-train-samples 15000 \
  --screen-validation-samples 3000 \
  --screen-test-samples 3000 \
  --screen-epochs 5 \
  --confirm-epochs 15 \
  --training-mask-fraction 0.20 \
  --damage-fraction 0.35 \
  --damage-seed 909 \
  --alignment-weight 0.10 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen10_robust_representation_cuda
```

The runner resumes from `gen10_robust_representation_progress.json`. A pass
opens only a separately preregistered Gen-11 continual-adaptation test. A stop
closes the representation without automatic mask, loss, leak, threshold, or
gate sweeps. Memory and structural-plasticity mechanisms remain gated.

Gen-10 promoted only the conventional controls. Sensor dropout improved
confirmed clean and damaged TCN accuracy by 2.684 and 8.763 points, but the
residual LIF arm missed clean/damaged screening by 9.200/6.733 points despite
healthy spikes. The retained result and twelve-source closeout are in
[`gen5/outputs/gen10_robust_representation_cuda_2026-08-10/`](gen5/outputs/gen10_robust_representation_cuda_2026-08-10/)
and
[`gen5/outputs/gen10_research_closeout_2026-08-10/`](gen5/outputs/gen10_research_closeout_2026-08-10/).

## Gen-11 frozen sensory backbone plus plastic state adapter (completed: `stop`)

Gen-11 preserves the proven sensor-dropout TCN as a frozen sensory backbone.
Matched analog and LIF adapters learn only bounded correction signals during
fixed sensor-damage adaptation. This tests functional separation rather than
forcing the spiking subsystem to relearn the full classifier. The protocol is
in [gen5/docs/GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md](gen5/docs/GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md).

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen11_plastic_adapter.py \
  --device cuda \
  --seeds 154 155 156 \
  --source-epochs 15 \
  --source-mask-fraction 0.20 \
  --damage-fraction 0.35 \
  --damage-seed 909 \
  --adaptation-budgets 0 64 256 1024 4096 \
  --adaptation-epochs-per-block 3 \
  --adaptation-learning-rate 0.001 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen11_plastic_adapter_cuda
```

The runner checkpoints after each seed/strategy curve. Only a causally useful
LIF adapter that matches readout adaptation and retention can open STW/LTW.

Gen-11 did not pass. Full fine-tuning and readout adaptation recovered 3.295
and 2.330 points, while analog and LIF adapters recovered 1.353 and 0.783.
Removing LIF state erased its gain, but shuffling sample identity cost only
0.011 point, so the correction was not sample-specific. Results and the
13-source closeout are retained in
[`gen5/outputs/gen11_plastic_adapter_cuda_2026-08-10/`](gen5/outputs/gen11_plastic_adapter_cuda_2026-08-10/)
and
[`gen5/outputs/gen11_research_closeout_2026-08-10/`](gen5/outputs/gen11_research_closeout_2026-08-10/).
Synaptic STW/LTW remains closed.

## Gen-12 frozen-backbone associative memory (completed: `stop`)

Gen-12 replaces the failed generic adapter with an explicitly sample-keyed
fast-memory hypothesis. The robust sensor-dropout TCN stays frozen. Dense and
sparse rank-order prototype memories accumulate damaged-stream class
associations without gradient updates and are compared with static, readout,
and full-fine-tuning controls. Memory removal and class-association shuffling
must both destroy any claimed gain. The protocol is in
[gen5/docs/GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md](gen5/docs/GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md).

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen12_associative_memory.py \
  --device cuda \
  --seeds 157 158 159 \
  --source-epochs 15 \
  --source-mask-fraction 0.20 \
  --damage-fraction 0.35 \
  --damage-seed 909 \
  --adaptation-budgets 0 64 256 1024 4096 \
  --adaptation-epochs-per-block 3 \
  --adaptation-learning-rate 0.001 \
  --memory-mix 0.50 \
  --memory-temperature 0.10 \
  --spike-fraction 0.20 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen12_associative_memory_cuda
```

The memory is deliberately active only in the known damaged context. A pass
therefore opens context-discovery and consolidation testing—not an immediate
continuous-learning or synaptic-memory claim.

Gen-12 did not pass. Readout and full fine-tuning recovered 3.564 and 4.767
points, while dense and spiking prototypes recovered only 0.250 and 0.278.
The spiking code remained healthy at 20% activity and zero context-gated
forgetting, but memory removal and class shuffling cost only 0.278 and 0.417
point. Results and the 14-source closeout are retained in
[`gen5/outputs/gen12_associative_memory_cuda_2026-08-10/`](gen5/outputs/gen12_associative_memory_cuda_2026-08-10/)
and
[`gen5/outputs/gen12_research_closeout_2026-08-10/`](gen5/outputs/gen12_research_closeout_2026-08-10/).

## Gen-13 local three-factor plasticity (completed: `stop`)

Gen-13 tests whether the successful output-layer credit assignment can be
implemented with explicit local synaptic updates rather than autograd through
the frozen sensory backbone. Analog and 20%-dense spiking presynaptic traces
are compared with static, autograd-readout, and full-fine-tuning controls.
Fast-weight removal and output-class shuffling are mandatory causal controls.
The protocol is in
[gen5/docs/GEN13_LOCAL_PLASTICITY_PREREGISTRATION.md](gen5/docs/GEN13_LOCAL_PLASTICITY_PREREGISTRATION.md).

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen13_local_plasticity.py \
  --device cuda \
  --seeds 160 161 162 \
  --source-epochs 15 \
  --source-mask-fraction 0.20 \
  --damage-fraction 0.35 \
  --damage-seed 909 \
  --adaptation-budgets 0 64 256 1024 4096 \
  --adaptation-epochs-per-block 3 \
  --adaptation-learning-rate 0.001 \
  --local-learning-rate 0.50 \
  --local-weight-decay 0.0001 \
  --spike-fraction 0.20 \
  --target-parameters 133631 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --no-download \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/ssc \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen13_local_plasticity_cuda
```

Gen-13 did not pass. Full fine-tuning and autograd readout adaptation recovered
3.269 and 2.049 points, while analog and spiking local rules recovered only
0.420 and 0.410 point. Spiking activity was healthy at 20%, source forgetting
was zero, and about 16,768 fast synapses were active, but removal and
class-shuffle controls cost only 0.410 and 0.468 point. The failure is not an
activity or capacity collapse; the tested rule did not reproduce useful
output credit assignment. Results and the 15-source closeout are retained in
[`gen5/outputs/gen13_local_plasticity_cuda_2026-08-10/`](gen5/outputs/gen13_local_plasticity_cuda_2026-08-10/)
and
[`gen5/outputs/gen13_research_closeout_2026-08-10/`](gen5/outputs/gen13_research_closeout_2026-08-10/).

The Gen-9–13 supervised continual-adaptation branch is closed. The recommended
new program is a separately preregistered reward-modulated embodied-learning
test, not a Gen-13 hyperparameter rescue. See
[the program closeout](gen5/docs/CONTINUAL_ADAPTATION_PROGRAM_CLOSEOUT.md).

## Gen-14 reward-modulated embodied eligibility (completed: `stop`)

Gen-14 starts a genuinely new program after the Gen-13 branch closure. Local
sensor/action eligibility traces are reinforced by delayed scalar food/toxin
reward in the tensorized 2D world. No class label, target action, or autograd
gradient reaches the plastic fast weights. A shuffled-reward arm tests whether
any gain depends on assigning reward to the correct agent.

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen14_reward_eligibility.py \
  --device cuda \
  --seeds 163 164 165 \
  --agent-count 5000 \
  --food-count 64 \
  --toxin-count 64 \
  --baseline-steps 300 \
  --training-steps 1800 \
  --evaluation-steps 300 \
  --reward-delay-steps 12 \
  --eligibility-decay 0.95 \
  --trace-decay 0.90 \
  --reward-baseline-decay 0.99 \
  --local-learning-rate 0.02 \
  --fast-weight-decay 0.0001 \
  --progress-reward-scale 0.05 \
  --temperature 0.50 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen14_reward_eligibility_cuda
```

The progress file is saved after every completed seed. A pass opens only a
separately preregistered causal confirmation; it does not open STW/LTW or
structural-plasticity claims automatically.

Gen-14 did not pass. The oracle reached +8.381 net fitness per 1,000 steps,
validating the world and sensor-action mapping. Spiking eligibility finished
at -0.109, below static behavior (+0.641) and shuffled reward (+0.052).
Although its own cold-start-to-evaluation score rose, static behavior rose
more; that phase comparison is confounded by world evolution and reward-buffer
warm-up. Healthy 20.04% spiking and zero weight saturation rule out an
execution collapse. Results are retained in
[`gen5/outputs/gen14_reward_eligibility_cuda_2026-08-10/`](gen5/outputs/gen14_reward_eligibility_cuda_2026-08-10/).
The corresponding 16-source ledger is in
[`gen5/outputs/gen14_research_closeout_2026-08-10/`](gen5/outputs/gen14_research_closeout_2026-08-10/).

Gen-14 triggered a 16-source evidence freeze and theory/baseline reset rather
than another automatic local-rule phase. Gen-15 has now completed the required
stationary reward diagnostic. See
[the evidence-freeze decision](gen5/docs/RESEARCH_EVIDENCE_FREEZE.md).

## Gen-15 matched reward-learning baseline (completed: `pass`)

Gen-15 executes the diagnostic required by the evidence freeze. It replaces
Gen-14's non-stationary phase comparison with identical seeded baseline/final
world resets and tests a conventional shared-policy REINFORCE learner against
static, oracle, and shuffled-reward controls. It does not introduce another
local-plasticity rule.

Colab T4 execution cell:

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen15_reward_baseline.py \
  --device cuda \
  --seeds 166 167 168 \
  --agent-count 1000 \
  --food-count 64 \
  --toxin-count 64 \
  --evaluation-steps 300 \
  --training-steps 1800 \
  --rollout-steps 30 \
  --reward-delay-steps 12 \
  --progress-reward-scale 0.05 \
  --hidden-units 32 \
  --learning-rate 0.003 \
  --weight-decay 0.0001 \
  --discount 0.99 \
  --entropy-weight 0.01 \
  --gradient-clip 1.0 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen15_reward_baseline_cuda
```

Gen-15 passed all registered gates. Correct-reward REINFORCE improved by
+0.992 fitness per 1,000 steps, finished +1.267 above shuffled reward, and the
static reset reproduced exactly. The effect remains weak and seed-sensitive:
the final mean was -0.271 and most of the gain came from one seed. This proves
that the reward protocol carries usable identity-specific credit; it does not
validate Gen-14 or any AMMC local mechanism. See
[the Gen-15 analysis](gen5/docs/GEN15_REWARD_BASELINE_ANALYSIS.md).

## Gen-16 local score-function credit (completed: `pass`)

Gen-16 now isolates credit assignment on the smallest matched system. An
8-to-4 linear policy is trained either by autograd REINFORCE or by the explicit
local rule `return × sensor × (chosen - probability)`. Static, oracle, and
agent-shuffled-reward controls use identical seeded resets. A pass requires
gradient equivalence, matched behavioral learning, and replicated reward
identity before the rule can be translated into sparse spikes.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen16_local_score_credit.py \
  --device cuda \
  --seeds 169 170 171 \
  --agent-count 1000 \
  --food-count 64 \
  --toxin-count 64 \
  --evaluation-steps 300 \
  --training-steps 1800 \
  --rollout-steps 30 \
  --reward-delay-steps 12 \
  --progress-reward-scale 0.05 \
  --learning-rate 0.02 \
  --weight-decay 0.0001 \
  --discount 0.99 \
  --gradient-clip 1.0 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen16_local_score_credit_cuda
```

Gen-16 passed every frozen gate. The manual gradient matched autograd within
`2.79e-9`, both policies produced identical final fitness, and correctly
assigned reward beat the shuffled control on all three seeds. The mean gain
was only +0.183, so this validates exact analog linear credit assignment—not
a capable standalone SNN. See
[the Gen-16 analysis](gen5/docs/GEN16_LOCAL_SCORE_CREDIT_ANALYSIS.md).

## Gen-17 sparse-spiking local credit (completed: `stop`)

Gen-17 replaces each analog sensor value with a parameter-matched Bernoulli
event while preserving the validated local update. Analog, static, oracle, and
agent-shuffled controls are included in the same frozen run.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen17_sparse_spiking_credit.py \
  --device cuda \
  --seeds 172 173 174 \
  --agent-count 1000 \
  --food-count 64 \
  --toxin-count 64 \
  --evaluation-steps 300 \
  --training-steps 1800 \
  --rollout-steps 30 \
  --reward-delay-steps 12 \
  --progress-reward-scale 0.05 \
  --learning-rate 0.02 \
  --weight-decay 0.0001 \
  --discount 0.99 \
  --gradient-clip 1.0 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen17_sparse_spiking_credit_cuda
```

Gen-17 generated healthy events (`6.369%` training, `12.078%` evaluation)
and preserved the exact manual gradient (`3.73e-9` maximum error), but the
correct-reward spiking learner lost `-0.391` fitness per 1,000 steps and
finished `-1.052` below shuffled reward. The analog reference also gained only
`+0.004` on the fresh seeds. The Bernoulli translation is rejected and analog
local-credit robustness is reopened. See
[the Gen-17 analysis](gen5/docs/GEN17_SPARSE_SPIKING_CREDIT_ANALYSIS.md).

## Gen-18 held-out local-credit replication (preregistered)

Gen-18 reruns the unchanged Gen-16 analog local rule on ten untouched seeds.
It adds no mechanism. A pass requires at least `7/10` seeds to clear the gain
and reward-identity thresholds and positive lower 95% confidence bounds.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!python gen5/examples/gen18_local_credit_replication.py \
  --device cuda \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen18_local_credit_replication_cuda
```

See the frozen
[Gen-18 preregistration](gen5/docs/GEN18_LOCAL_CREDIT_REPLICATION_PREREGISTRATION.md).

Gen-18 returned `stop`. Mean correct-reward gain was `+0.796`, but only `5/10`
seeds qualified and its lower 95% bound was `-0.016`. Correct reward finished
`+0.510` above shuffled reward on average, but only `6/10` seeds qualified and
the lower bound was `-0.013`. The local reward-credit program is closed under
its preregistered rule. See
[the Gen-18 analysis](gen5/docs/GEN18_LOCAL_CREDIT_REPLICATION_ANALYSIS.md).

## Gen-19 real event-vision state replication (completed: `stop`)

Gen-19 starts a separate external-generalization program. It tests the
established residual LIF state mechanism on N-MNIST using a matched temporal
Conv1D and full/direct-only/state-only/shuffled-state causal evaluations. It
does not reopen local reward credit.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!pip install -q tonic
!python gen5/examples/gen19_nmnist_state_replication.py \
  --device cuda \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen19_nmnist_state_replication_cuda
```

N-MNIST downloads automatically on the first run; no input file must be
uploaded. See the frozen
[Gen-19 preregistration](gen5/docs/GEN19_NMNIST_STATE_REPLICATION_PREREGISTRATION.md).

Gen-19 learned N-MNIST strongly: Conv1D reached `96.860%` and the matched
residual LIF reached `96.317%`. Removing LIF state cost `15.210` points and
activity was healthy at `17.052%`. The decisive identity control failed,
however: shuffling state between samples improved accuracy by `2.300` points,
with `0/3` qualifying seeds. The residual-state claim is therefore limited to
SHD/SSC event audio and is not extended to N-MNIST event vision. See the
[Gen-19 analysis](gen5/docs/GEN19_NMNIST_STATE_REPLICATION_ANALYSIS.md).

The runner now writes a checksummed manifest and
`gen19_nmnist_state_replication_bundle.zip` beside the individual artifacts,
so a complete result can be downloaded as one file after future verification
runs.

## Full-resolution N-MNIST accuracy benchmark (preregistered)

This bounded side track measures standard classification performance before
returning to Gen-20. It preserves Gen-19 unchanged and compares a full-frame
CNN, a spatial-temporal CNN, and a convolutional SNN with learnable membrane
dynamics. Screening uses validation accuracy only; at most two arms receive
three-seed full-data confirmation.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!pip install -q tonic
!python gen5/examples/nmnist_accuracy_benchmark.py \
  --device cuda \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/nmnist \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/nmnist_accuracy_benchmark_cuda
```

The first run builds an approximately 1.39 GB full-resolution cache in Drive.
The benchmark resumes after each completed arm or seed and finishes with a
checksummed single-file result bundle. The practical target is `99.0%`; the
stretch target is `99.4%`. Both pass and stop return to Gen-20. See the frozen
[benchmark preregistration](gen5/docs/NMNIST_ACCURACY_BENCHMARK_PREREGISTRATION.md).

The completed track passed: the full-resolution spatial-temporal CNN reached
**99.4767% mean N-MNIST test accuracy** over three seeds, while the frame CNN
reached 99.1233%. ConvPLIF screened at 93.07% and was not promoted, so the
99.4767% result is a conventional event-vision benchmark rather than an SNN
record. See the [benchmark analysis](gen5/docs/NMNIST_ACCURACY_BENCHMARK_ANALYSIS.md).
Gen-20 is frozen as one multi-timescale spiking translation package in the
[Gen-20 preregistration](gen5/docs/GEN20_SPIKING_SPATIOTEMPORAL_TRANSLATION_PREREGISTRATION.md).

## Gen-20 spiking spatial-temporal translation

Gen-20 is implemented as one bounded screen/confirmation run. It compares the
dense spatial-temporal teacher and frozen ConvPLIF baseline against direct and
teacher-distilled multi-timescale residual PLIF models. Only new spiking arms
above 97.5% screening validation accuracy with 1–30% activity can advance.
Confirmed arms must also pass state-removal, per-sample temporal-shuffle, and
activity-scaled operation-proxy gates.

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!pip install -q tonic
!python gen5/examples/gen20_spiking_spatiotemporal.py \
  --device cuda \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/nmnist \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/gen20_spiking_spatiotemporal_cuda
```

The runner resumes from its progress JSON and companion checkpoint directory.
Download `gen20_spiking_spatiotemporal_bundle.zip` when it finishes.

Gen-20 completed on 2026-08-20 with terminal status `stop`. The dense teacher
screened at 99.1165%, while ConvPLIF, multiscale residual PLIF, and its
distilled variant reached 96.2160%, 96.3661%, and 96.3327%. Neither new arm met
the frozen 97.5% promotion gate, so confirmation and causal temporal controls
did not run. The complete 22-source synthesis is archived in
[`gen20_evidence_synthesis_2026-08-20`](gen5/outputs/gen20_evidence_synthesis_2026-08-20/),
and the resulting program decision is documented in the
[post-Gen-20 sanity check](gen5/docs/PROGRAM_SANITY_CHECK_AFTER_GEN20.md).
