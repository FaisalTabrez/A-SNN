# Phase 26 targeted-synaptogenesis analysis

Archive: `structural_sequential_mnist_cuda.zip`

Archive SHA-256: `A8C67E584D704A168DC32CF7CBB09B8D392C9F4D625DDEDDCAEE0E6296C81216`

Run configuration: CUDA, seeds 42/43/44, 20,000 training examples, 5,000
test examples, 15 epochs, LTW warm-up through epoch 10, 64 hidden neurons,
272 protected seed edges, and a 512-slot edge pool.

## Main result

Random sensor growth produced a small, readout-dependent benefit. The
48-sensor-edge arm reached 46.73% mean linear accuracy versus 45.97% for the
paired fixed-topology warm-all control, a gain of 0.77 percentage points. It
improved two of three seeds by at least 0.5 points and therefore passes the
predeclared linear-readout gate. Its paired seed gains were -0.96, +1.20, and
+2.06 points, so the effect is not yet robust to initialization.

The 16-edge sensor arm gained 0.49 points with the linear readout. All three
seeds improved, but only one exceeded the 0.5-point practical threshold. The
64-edge recurrent-growth arm gained 0.21 points and was weaker than sensor
growth.

No structural arm improved the MLP readout on average:

| Arm | Linear gain vs fixed | MLP gain vs fixed |
| --- | ---: | ---: |
| Sensor sprout 16 | +0.49 points | -0.23 points |
| Sensor sprout 48 | +0.77 points | -0.21 points |
| Recurrent sprout 64 | +0.21 points | -0.08 points |

## Structural diagnostics

- Linear sensor sprouts moved substantially from their 0.1 birth LTW:
  0.0705 mean absolute movement for 16 edges and 0.0579 for 48 edges.
- Linear recurrent sprouts moved only 0.0127, reinforcing the Phase 25 finding
  that the useful adaptation bottleneck is the sensor projection.
- Hidden-event ratios remained bounded: 1.25-1.38 for linear arms and
  1.07-1.17 for MLP arms.
- LTW saturation stayed minor. Mean lower saturation was at most 0.31% and
  mean upper saturation was at most 0.58% among the structural arms.
- The 48-edge sensor arm increased the active graph from 272 to 320 edges, an
  17.6% topology expansion for a 0.77-point linear gain.

## Sanity check against project goals

Supported by this run:

1. Adding sensor-to-hidden routes can improve linear separability on the
   sequential task.
2. Newly grown sensor edges receive meaningful LTW gradients without causing
   firing or weight saturation.
3. Sensor growth is more promising than indiscriminate recurrent growth in
   the current architecture.

Not supported by this run:

1. Synaptogenesis is not yet a generally beneficial learning mechanism: the
   MLP readout did not benefit.
2. Random growth is not seed-robust; one of three linear runs became worse.
3. More edges are not automatically better, and the run does not justify
   pruning the protected 272-edge core.
4. The recurrent AMMC representation remains far below the raw-pixel MLP
   baseline (56.42% fixed warm-all versus 94.21% raw MLP).
5. There is still no evidence for competitive MNIST performance, continual
   learning, or broad superiority over established SNN/ANN approaches.

## Decision

Proceed to Phase 27: utility-gated structural plasticity.

- Rank a deterministic pool of inactive sensor edges using task-loss
  gradients after the readout warm-up.
- Compare 16 and 48 gradient-selected edges against a paired random 48-edge
  control and the fixed warm-all topology.
- Keep the original 272-edge graph permanently protected.
- In one conservative arm, prune only newly grown edges that decay below 95%
  of their birth LTW, with a maximum removal of 50% of the sprouts.

Gradient-gated growth should only be accepted if it beats the paired random
growth control by at least 0.5 percentage points on average, improves at least
two of three seeds, and remains stable for both event rate and LTW saturation.
