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
- Current scientific step: Phase 36 tests a parameter-matched multi-scale
  temporal readout with a fixed shuffled-time control.

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

## Evidence discipline

Experiment outputs kept in the repository should live under `gen5/outputs/`
and include an `analysis.md` whenever possible. New research decisions should
update [research.md](research.md) in the same change set.

The repo is deliberately part lab notebook, part framework. The code tells us
what can run; the evidence folders tell us what we have actually observed.
