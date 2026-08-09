# Phase 22: LTW Optimization Diagnostic

Phase 21 established that surrogate gradients move active LTWs, but it did not
meet the pre-registered gate for structural plasticity. The linear head lost
`0.31` accuracy points and the MLP gained only `0.13` points relative to paired
frozen controls. Phase 22 therefore keeps topology fixed and diagnoses the
optimization path before introducing sprouting or pruning.

## Paired interventions

Every arm within a seed/classifier block receives the same sparse topology and
readout initialization. The default sweep includes:

- frozen LTWs;
- the Phase 21 joint-training setting (`1e-3`, slope `10`);
- a `10`-epoch readout warmup followed by `5` LTW fine-tuning epochs;
- LTW rates `1e-4` and `3e-4` crossed with surrogate slopes `5` and `10`;
- sensor-edge-only and recurrent-edge-only fine-tuning at `3e-4`, slope `10`.

The runner records paired accuracy gain, hidden-event-rate drift, LTW movement
by edge type, boundary saturation, parameter counts, time, and throughput.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint22_ltw_optimization_diagnostic.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/ltw_optimization_diagnostic_cuda
```

List or restrict arms with:

```python
!python gen5/examples/sprint22_ltw_optimization_diagnostic.py --list-arms
```

## Decision rule

Do not inspect the reserved final-test complement during this sweep. Select an
LTW intervention only if it:

- gains at least `0.5` accuracy points on average over its paired frozen arm;
- improves at least two of three seeds;
- keeps the hidden event-rate ratio between `0.5` and `2.0`;
- avoids material LTW boundary saturation.

If no arm passes, stop supervised LTW tuning and reassess the neuron/event
dynamics before topology mutation. If an arm passes, replicate it with more
seeds before considering a later structural-plasticity phase.

## First CUDA result

No arm passes. Warmup stabilizes activity and removes LTW saturation, but the
best linear gain is only `0.087` points and the best MLP result is tied with its
frozen control. Recurrent-only updates are neutral or negative.

Supervised LTW tuning on static MNIST is now frozen. Phase 23 tests whether
recurrent edges themselves add causal value beyond sensor-to-hidden feedforward
expansion. See
`gen5/outputs/ltw_optimization_diagnostic_cuda_2026-08-09/analysis.md`.
