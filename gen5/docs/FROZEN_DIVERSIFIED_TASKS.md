# Frozen diversified task benchmarks

Sprint 15 freezes the current AMMC sparse substrate and evaluates it on
download-free synthetic tasks outside the embodied foraging loop.

No evolution, plasticity, optimizer update, sprouting, or pruning is allowed
during these tests. The goal is a reality check: does the current sparse
spiking substrate generalize beyond the sensor-motor world it was designed for?

## Why this exists

The delay-`2` neuron-scaling result showed that larger hidden-node pools do not
improve the current delayed-reward foraging task. Compact `16`-neuron models
win because the task still rewards direct sensor-motor loops.

The next question is not "can we make the brain bigger?" It is:

> What kinds of tasks make the current frozen substrate fail, and which failures
> require training, protected hidden expansion, or a new task transducer?

## Tasks

The first benchmark set keeps the existing 8-sensor / 4-motor convention:

- sensor channels `0:4`: directional food-like inputs
- sensor channels `4:8`: auxiliary toxin/cue inputs
- motor channels `8:12`: north/east/south/west decisions

| Task | Purpose |
|---|---|
| `direction_copy` | Sanity check for direct food-direction reflexes. |
| `anti_toxin` | Tests whether toxin-like channels can drive avoidance without learning. |
| `cue_switch` | Tests context-dependent reversal. |
| `delayed_recall` | Tests short temporal memory after inputs disappear. |
| `two_pulse_sum` | Tests simple sequence integration beyond reflex lookup. |

## Baselines reported

Each task reports:

- frozen AMMC accuracy,
- random 4-way accuracy,
- instant reflex accuracy,
- integrated reflex accuracy,
- oracle accuracy,
- inactive output rate,
- mean correct evidence margin.

The reflex baselines are intentionally simple. If they beat the frozen AMMC, the
task does not yet justify a complex sparse recurrent system. If the frozen AMMC
beats reflexes without learning, the wiring prior is doing useful computation.

## Colab command

```python
!python gen5/examples/sprint15_frozen_diversified_tasks.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_cuda
```

For a faster smoke test:

```python
!python gen5/examples/sprint15_frozen_diversified_tasks.py \
  --device cuda \
  --sample-count 512 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_smoke_cuda
```

## Expected outputs

- `frozen_diversified_tasks.json`
- `frozen_diversified_tasks_summary.csv`
- `frozen_diversified_tasks_summary.png`

## Interpretation rule

Use this benchmark as a diagnostic, not a leaderboard.

Possible outcomes:

- Frozen AMMC wins only on `direction_copy`: current model is mostly reflexive.
- Frozen AMMC beats instant reflex on `delayed_recall`: recurrent substrate is
  preserving useful temporal evidence.
- Frozen AMMC fails `cue_switch` and `two_pulse_sum`: we need training,
  protected hidden expansion, or a task-specific readout before broader
  cognitive claims.
- Larger future hidden models win these tasks: then scaling has a real purpose.

## First CUDA result

The first CUDA run is archived at:

- `gen5/outputs/frozen_diversified_tasks_cuda_2026-06-29/analysis.md`

Main conclusion:

- `direction_copy` reached `100%`, matching reflex baselines.
- `delayed_recall` reached `100%`, but integrated reflex also reached `100%`,
  so this is temporal evidence integration rather than proof of hidden-state
  memory.
- `anti_toxin` stayed at chance with `100%` inactive output, showing that the
  frozen toxin prior suppresses rather than generates active avoidance.
- `cue_switch` matched reflexes at roughly `50%`, meaning the context cue is
  ignored.
- `two_pulse_sum` stayed at chance, meaning the frozen circuit does not perform
  sequence composition.

Next recommendation: add a frozen representation probe with a trainable linear
readout while keeping the recurrent sparse AMMC substrate frozen.

## Frozen representation probe

The representation probe keeps the sparse AMMC recurrent substrate frozen, then
trains only a small linear classifier over final membrane and spike-count
features.

This distinguishes two failure modes:

- If the linear probe solves a task that the frozen motor readout failed, the
  representation exists and the motor readout/transducer is the weak link.
- If the linear probe also fails, the frozen substrate lacks the task-relevant
  representation.

Colab command:

```python
!python gen5/examples/sprint15_frozen_representation_probe.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_representation_probe_cuda
```

Expected outputs:

- `frozen_representation_probe.json`
- `frozen_representation_probe_summary.csv`
- `frozen_representation_probe_summary.png`

## First representation-probe result

The first CUDA probe run is archived at:

- `gen5/outputs/frozen_representation_probe_cuda_2026-06-29/analysis.md`

Main conclusion:

- `anti_toxin` jumped from `25%` frozen motor accuracy to `100%` linear-probe
  accuracy. This is a readout/transducer failure, not a representation failure.
- `cue_switch` improved from roughly `50%` to `85.76%`, so context information
  is partially present but not perfectly linearly disentangled.
- `two_pulse_sum` only improved from `25%` to `31.81%`, so modular sequence
  composition is still missing from the frozen substrate.

Next recommendation: implement a minimal trainable readout/transducer adapter
while keeping recurrent sparse AMMC weights frozen. Use that as the bridge
between frozen probing and full recurrent/plastic training.

## Frozen readout/transducer adapter

The readout adapter is the deployable version of the representation probe. It
still freezes the recurrent sparse AMMC substrate, but now treats the trained
readout as an adapter we may actually keep around the substrate.

Supported feature modes:

- `full_trace`: final membrane plus spike counts for every neuron. This should
  reproduce the linear-probe ceiling when `--adapter-kind linear` is used.
- `motor_trace`: final membrane plus spike counts for only the motor neurons.
  This is the stricter diagnostic. If it fails while `full_trace` succeeds, the
  useful state is distributed across the brain and the fixed motor pathway is
  the bottleneck.

Colab command:

```python
!python gen5/examples/sprint15_frozen_readout_adapter.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --adapter-kind linear \
  --feature-mode full_trace \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_cuda
```

Strict motor-pathway diagnostic:

```python
!python gen5/examples/sprint15_frozen_readout_adapter.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --adapter-kind linear \
  --feature-mode motor_trace \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_motor_trace_cuda
```

Optional nonlinear readout check:

```python
!python gen5/examples/sprint15_frozen_readout_adapter.py \
  --device cuda \
  --adapter-kind mlp \
  --hidden-units 32 \
  --feature-mode full_trace \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_mlp_cuda
```

Expected outputs:

- `frozen_readout_adapter.json`
- `frozen_readout_adapter_summary.csv`
- `frozen_readout_adapter_summary.png`

Interpretation:

- If `linear/full_trace` matches the representation probe, the probe result is
  reproducible as an adapter.
- If `motor_trace` underperforms `full_trace`, the next engineering target is a
  better sensor-to-motor transducer/readout map.
- If `mlp/full_trace` beats `linear/full_trace`, the representation exists but
  is not linearly separated.
- If all adapter modes fail on `two_pulse_sum`, the recurrent substrate itself
  needs temporal/compositional learning rather than readout tuning.
