# Phase 19 event representation decomposition: analysis

Date: 2026-08-09

Bundle SHA-256: `17F3102BB0A4944830C2D3BBB46FFACC6BACF7A04223CA96DFAE352BC80FBB52`

The bundle is complete: six feature families, two classifier families, three
seeds, 36 seed-level records, 12 aggregate rows, and a readable plot.

## Aggregate result

| Representation | Linear | Parameter-matched MLP |
|---|---:|---:|
| Raw intensity | 85.94% | 95.14% |
| Flattened latency | 88.11% | 91.40% |
| Sensor summary trace | 72.74% | 85.97% |
| Hidden summary trace | 73.03% | 83.43% |
| Full summary trace | 79.33% | 85.97% |
| Raw plus hidden summary | 86.92% | 93.50% |

## Paired findings

- Flattened latency minus raw intensity:
  - linear `+2.17` percentage points, positive for all seeds;
  - MLP `-3.74` points, negative for all seeds.
- Sensor summary minus flattened latency:
  - linear `-15.37` points;
  - MLP `-5.43` points.
- Hidden summary minus sensor summary:
  - linear `+0.29` points with inconsistent seed direction;
  - MLP `-2.54` points.
- Full summary minus sensor summary:
  - linear `+6.59` points;
  - MLP effectively unchanged at `-0.01` points.
- Raw plus hidden minus raw:
  - linear `+0.98` points, positive for all seeds;
  - MLP `-1.64` points, negative for all seeds.

Exploratory paired t intervals exclude zero for all comparisons above except
hidden-versus-sensor linear and full-versus-sensor MLP. With only three seeds,
the intervals are diagnostic rather than publication-grade.

## Interpretation

1. One-spike latency coding is not the main linear-information bottleneck. Its
   explicit time bins are more linearly separable than the raw `8x8` values.
2. Final count-plus-membrane pooling is the dominant loss. It discards the
   useful time structure preserved by flattened latency events.
3. The current recurrent hidden summary does not independently improve the
   sensor summary. Recurrence is weak, destructive, or simply poorly pooled.
4. Hidden activity contains a small complementary linear signal: concatenating
   it with raw input improves every seed. It does not beat the raw MLP under the
   current parameter-budget match, where the higher-dimensional input forces a
   narrower hidden layer.
5. The MLP latency deficit may combine temporal quantization and the narrower
   parameter-matched head. It should not be interpreted as proof that latency
   events are intrinsically inferior.

## Decision

Implement Phase 20 as temporal-state preservation with a residual raw pathway.
Record the pre-reset neuron state at every timestep instead of reducing the
episode to spike counts plus final membrane.

Compare:

- raw intensity;
- flattened latency;
- Phase 19 full summary;
- sensor temporal state;
- hidden temporal state;
- full temporal state;
- raw plus hidden temporal state.

Use both linear and approximately parameter-budget-matched MLP heads. Keep the
reservoir frozen. If hidden temporal state still adds no value, Phase 21 should
tune or train the substrate rather than increase its size.

## Validation boundary

This phase reused the 5,000-image engineering validation subset selected in
Phase 18. Preserve the unused 5,000-image complement of the official MNIST
test set until the temporal intervention is fixed.
