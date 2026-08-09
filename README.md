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
- Current scientific step: Phase 38 compares AMMC with parameter-matched dense
  recurrent LIF and GRU baselines before any recurrent redesign.

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
