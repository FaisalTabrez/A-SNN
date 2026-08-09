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
