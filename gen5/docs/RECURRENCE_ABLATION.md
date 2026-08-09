# Phase 23: Causal Recurrence Ablation

Phase 22 found stable LTW update schedules but no practically meaningful
accuracy gain. Further supervised LTW tuning on static MNIST is frozen. Phase
23 asks a more fundamental question: do recurrent hidden-to-hidden edges
causally improve the temporal representation, or is sensor-to-hidden random
expansion sufficient?

## Causal comparison

For each seed, two reservoirs begin with the same generated topology and
weights:

- **feedforward expansion:** retain the 128 sensor-to-hidden edges and disable
  all 256 hidden-to-hidden recurrent edges;
- **recurrent expansion:** retain the full 384-edge topology.

The benchmark compares raw pixels, sensor temporal state, hidden/full
feedforward temporal state, and hidden/full recurrent temporal state. Linear
and parameter-budget-matched MLP readouts use identical initialization and
minibatch order for equal-dimensional causal pairs.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint23_recurrence_ablation.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/recurrence_ablation_cuda
```

## Decision rule

- If full recurrent state beats paired full feedforward state by at least
  `0.5` points on average and improves at least two of three seeds, recurrence
  has a useful causal role on this task.
- If feedforward expansion captures nearly all of the full-state gain, stop
  using static MNIST to justify recurrent plasticity and move to a task with
  genuine temporal dependence.
- Keep the reserved final-test complement untouched until this diagnostic
  selects the next task/model intervention.

## First CUDA result (2026-08-09)

The three-seed engineering-validation run rejected the practical recurrence
gate. Relative to the paired 128-edge feedforward expansion, the 384-edge
recurrent graph changed full-state accuracy by only `+0.107` percentage points
with a linear readout and `-0.053` points with an MLP. The linear delta was
positive in all three seeds, but it was far below the pre-registered `0.5`
point effect threshold; the MLP delta was negative in every seed.

The feedforward full-state representation already reached `91.407%` linear
and `92.613%` MLP accuracy. Adding 256 recurrent edges raised hidden event rate
by about `11.7%` without a practical accuracy return. Static MNIST therefore
does not support a claim that AMMC recurrence or structural plasticity is
useful. Phase 24 moves to row-sequential MNIST and permits the readout to see
only the reservoir's final hidden state.

See the retained [Phase 23 analysis](../outputs/recurrence_ablation_cuda_2026-08-09/analysis.md).
