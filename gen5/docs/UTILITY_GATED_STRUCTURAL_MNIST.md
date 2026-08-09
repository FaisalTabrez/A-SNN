# Phase 27: Utility-Gated Structural Plasticity

Phase 26 found a small linear-readout benefit from random sensor growth, but
the effect was seed-dependent and did not transfer to the MLP readout. Phase 27
tests whether structural selection quality is the missing ingredient.

## Experimental arms

- `raw`: raw-pixel upper baseline.
- `frozen_recurrent`: fixed recurrent representation.
- `fixed_warm_all`: Phase 25 fixed-topology LTW control.
- `random_sensor_48`: paired replication of random 48-edge sensor growth.
- `gradient_sensor_16`: top 16 edges from a 192-candidate gradient-ranked pool.
- `gradient_sensor_48`: top 48 edges from the same ranked pool.
- `gradient_sensor_48_prune`: the 48-edge guided arm plus conservative
  peripheral pruning.

At epoch 10, the guided arms temporarily place 192 inactive sensor-to-hidden
candidate edges at zero LTW. Four deterministic training batches estimate the
absolute task-loss gradient for each edge. Candidate slots are then cleared and
only the highest-ranked 16 or 48 routes are born at LTW 0.1. This scoring pass
does not perform an optimizer step.

The pruning arm waits three training epochs after growth. It can remove at
most half of the new edges, and only sprouts whose LTW has fallen below 95% of
its birth value. The original 272-edge graph is never eligible for pruning.

## Colab run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint27_utility_gated_structural_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --scoring-batches 4 \
  --prune-after-epochs 3 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/utility_gated_structural_mnist_cuda
```

## Decision gate

Accept gradient-gated growth only if an arm:

1. beats the paired `random_sensor_48` arm by at least 0.5 percentage points
   on average;
2. improves at least two of three paired seeds;
3. keeps the event-rate ratio between 0.5 and 2.0;
4. avoids material LTW saturation; and
5. preferably benefits both readouts rather than only the linear decoder.

Accept peripheral pruning only if it retains the guided arm's accuracy within
0.25 points while removing edges. The outcome does not authorize pruning of
the protected seed graph.
