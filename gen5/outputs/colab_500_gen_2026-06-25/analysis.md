# AMMC Gen-5 Colab Run Analysis — 500 Generations

Date: 2026-06-25

Source files:

- `evolution_telemetry.json`
- `evolution_telemetry.csv`
- `evolution_telemetry.png`
- `champion_sparse_adjacency.json`
- `champion_connectome.json`
- `colab_weights.json`

## Summary

The 500-generation Colab run successfully produced a complete champion export
bundle. The telemetry shows rapid early selection, then a noisy plateau rather
than monotonic improvement. Structural evolution strongly increased topology
density, with mean active synapses rising from `9.16` to `86.03`.

The best recorded epoch fitness was `24`, reached at generations `236` and
`450`. The exported champion has fitness `24`, `16` neurons, and `88` active
synapses.

## Telemetry findings

| Metric | Value |
| --- | ---: |
| Generations | 500 |
| First max fitness | 12 |
| Final max fitness | 19 |
| All-time max fitness | 24 |
| All-time max fitness generations | 236, 450 |
| Mean max fitness across run | 18.408 |
| First 100-generation mean max fitness | 17.770 |
| Last 100-generation mean max fitness | 18.420 |
| Mean population fitness across run | -0.00748 |
| Mean active synapses start | 9.156 |
| Mean active synapses final | 86.033 |
| Mean active synapses last 100 generations | 85.957 |

Mean active synapse thresholds:

| Threshold | First generation reached |
| --- | ---: |
| 50 | 52 |
| 75 | 131 |
| 80 | 175 |
| 85 | 287 |

Mutation totals across the 500 epochs:

| Operation | Count |
| --- | ---: |
| Sprouts | 2,662,175 |
| Prunes | 1,888,501 |
| LTW mutations | 188,807,816 |

## Phase behavior

| Phase | Generations | Mean max fitness | Phase max fitness | Mean population fitness | Mean active synapses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 1-50 | 16.720 | 21 | -0.0784 | 31.441 |
| Early selection | 51-150 | 18.690 | 22 | 0.00252 | 66.909 |
| Mid run | 151-300 | 18.547 | 24 | -0.0225 | 82.734 |
| Late run | 301-500 | 18.585 | 24 | 0.0165 | 85.829 |

Interpretation:

- The evolutionary loop learns quickly in the first ~150 generations.
- After topology density saturates, fitness becomes noisy and mostly plateaus.
- Mean population fitness staying near zero suggests the champion tail improves
  while the bulk of the swarm remains noisy or fragile.
- The current mutation settings may be too topology-expansive; active synapses
  saturated near the edge-pool ceiling.

## Champion topology

Champion sparse adjacency:

| Metric | Value |
| --- | ---: |
| Fitness | 24 |
| Neurons | 16 |
| Active synapses | 88 |
| Excitatory edges | 66 |
| Inhibitory edges | 22 |
| Mean LTW | 0.09585 |
| LTW sum | 8.43490 |
| Max LTW | 0.53722 |
| Mean STW | 0 |
| Mean delay steps | 31.42 |
| Delay range | 0-64 |

LTW threshold distribution:

| Threshold | Edge count |
| --- | ---: |
| LTW = 0 | 8 |
| LTW < 0.001 | 8 |
| LTW < 0.01 | 15 |
| LTW < 0.05 | 37 |
| LTW >= 0.1 | 34 |
| LTW >= 0.2 | 11 |
| LTW >= 0.3 | 3 |
| LTW >= 0.5 | 1 |

Role-level edge routing:

| Route | Count |
| --- | ---: |
| food_sensor -> hidden | 7 |
| food_sensor -> motor | 9 |
| food_sensor -> food_sensor | 3 |
| food_sensor -> toxin_sensor | 7 |
| toxin_sensor -> hidden | 12 |
| toxin_sensor -> motor | 9 |
| toxin_sensor -> food_sensor | 3 |
| toxin_sensor -> toxin_sensor | 3 |
| motor -> hidden | 3 |
| motor -> motor | 5 |
| motor -> food_sensor | 3 |
| motor -> toxin_sensor | 3 |
| hidden -> hidden | 4 |
| hidden -> motor | 8 |
| hidden -> food_sensor | 3 |
| hidden -> toxin_sensor | 6 |

## Bundle validation

The export bundle is internally consistent:

- `champion_sparse_adjacency.json`: 88 rows.
- `champion_connectome.json`: 16 neurons, 88 synapses.
- `colab_weights.json`: 88 edge records.
- Every weight edge identity matches the corresponding browser synapse:
  `edge_index`, `source_id`, `target_id`, and `dendrite_id`.
- No mismatches were detected between adjacency LTWs, connectome LTWs, and
  importable Colab weights.

## Immediate implications

1. The champion is valid for browser injection.
2. The run demonstrates fast early evolutionary gains, but not sustained
   long-run improvement after structural saturation.
3. Next runs should test lower sprout probability, stronger low-LTW pruning, or
   a topology budget pressure term.
4. A controlled browser replay should compare:
   - seed genome
   - exported champion connectome
   - champion with plasticity off
   - champion with plasticity on

