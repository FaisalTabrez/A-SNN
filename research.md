# AMMC Research Log

Living research memory for the AMMC-SNN / AMMC Gen-5 project.

Update rule: before making a meaningful project decision, revisit this file.
After making a meaningful decision, experiment, import/export, monitoring pass,
architecture pivot, or result interpretation, revise this file with:

- date/time when relevant
- decision or finding
- evidence/source file
- implication for the next sprint
- open questions or risks

## Current thesis

AMMC is moving from a browser-based biological proof-of-concept into a
production-grade neuromorphic framework. Gen-4 validated the core mechanics in
JavaScript. Gen-5 should become a mathematical, hardware-accelerated Python/C++
runtime for continuous-learning agents.

The central bet is that sparse dynamic topology, dual memory timescales
(`STW + LTW`), astrocyte-like low-frequency modulation, and spatiotemporal
delays can replace parts of static ANN infrastructure for embodied,
continuous-learning systems.

## Core findings

### 1. LTW/STW separation works as a bridge between learning and persistence

Finding: splitting synaptic state into short-term weight (STW) and long-term
weight (LTW) gives the system a useful working-memory/permanent-memory
distinction.

Evidence:

- Gen-4 implemented STW/LTW in the browser simulator.
- Colab-imported LTW weights survived injection and remained stable during
  behavior monitoring.
- Corrected behavior observation showed mean LTW fixed at `0.86`, while mean
  STW fluctuated from `0.07` to `0.13`.

Source:

- `outputs/behavior_observation_corrected_2026-06-25T05-53-08-570Z.md`
- `outputs/behavior_observation_corrected_2026-06-25T05-53-08-570Z.json`

Implication:

- Gen-5 must treat STW and LTW as first-class memory tiers.
- Optimizers should be able to apply different learning rates/decay rules to
  each tier.
- Pruning should be based on LTW, not transient STW.

### 2. The PyTorch topology wall is real

Finding: standard PyTorch `nn.Parameter` tensors expect stable shape. True
structural plasticity changes edge count, so a naive dense/static tensor model
cannot represent physical synaptogenesis/pruning cleanly.

Evidence:

- The Gen-4 PyTorch exporter could export sparse edge lists and import updated
  weights, but Colab only changed weights, not topology.
- The generated `colab_weights.json` mapped edge index to updated weight.
- Import required topology identity checks to prevent mismatched edge injection.

Source:

- `outputs/evolved_model.py`
- `outputs/colab_weights.json`
- `outputs/evolved_connectome_for_colab_weights.json`

Implication:

- Gen-5 needs a dynamic sparse edge allocator.
- The Python prototype should keep optimizer-visible shapes stable while the
  future C++/CUDA backend manages allocation, pruning, and slot reuse.
- Do not claim PyTorch can safely resize trainable parameters mid-backward.

### 3. Astrocytes behave like a low-frequency spatial loss/modulation layer

Finding: the Gen-4 astrocyte overlay can be reframed as a dense, low-resolution
chemical grid that modulates the fast sparse electrical graph.

Evidence:

- During monitoring, local GABA modulation appeared in `17/19` samples.
- Regional spike load was visible and GABA suppression damped activity.
- Reward/punishment events in Gen-4 map naturally to dopamine/GABA fields.

Source:

- `outputs/behavior_observation_corrected_2026-06-25T05-53-08-570Z.md`

Implication:

- Gen-5 should compute two coupled systems:
  - high-frequency sparse electrical graph
  - low-frequency dense chemical tensor
- The chemical tensor can act as a spatial learning-rate/loss overlay rather
  than a single global scalar loss.

### 4. The Colab bi-directional bridge is viable

Finding: Gen-4 successfully exported an evolved sparse topology to PyTorch,
received trained weights back as JSON, and injected them into the live browser
brain without tearing down the animation loop.

Evidence:

- Exported model contained 21 sparse edges.
- Imported Colab payload updated 21 LTW values and cleared STW.
- Live UI success toast: `GPU memories injected - 21 LTW updated - STW cleared`.

Source:

- `outputs/evolved_model.py`
- `outputs/colab_weights.json`

Implication:

- Gen-5 serialization should keep edge identity stable:
  `edge_index`, `source_id`, `target_id`, `dendrite_id`.
- External training systems should update weights without silently changing
  topology unless using a topology-aware AMMC backend.

### 5. Post-injection behavior shows motor actuation, but reward performance is inconclusive

Finding: after Colab weight injection, the inspected organism drove motor output
and moved in the environment, but no food/toxin collisions happened in the
short corrected monitoring pass.

Evidence:

- Motor events observed: `O1 - Motor down`, `O1 - Motor up`.
- Non-zero velocity in `13/19` samples.
- Closest approach: food `16 px`, toxin `42 px`.
- Food hit delta `0`, toxin hit delta `0`, fitness delta `0`.

Source:

- `outputs/behavior_observation_corrected_2026-06-25T05-53-08-570Z.md`

Implication:

- Future evaluations need longer controlled trials.
- We need separate modes for:
  - isolated Colab-imported brain evaluation
  - plasticity-enabled adaptation
  - auto-evolution / mutation trials

### 6. The 500-generation Gen-5 Colab run found a champion, but topology saturated

Finding: the Gen-5 tensorized swarm completed 500 generations and produced a
valid champion export bundle. Fitness improved quickly early in the run, then
entered a noisy plateau while mean active synapses saturated near the edge-pool
ceiling.

Evidence:

- `500` telemetry records were produced.
- Best max fitness was `24`, reached at generations `236` and `450`.
- Mean max fitness was `17.77` over the first 100 generations and `18.42` over
  the final 100 generations.
- Mean active synapses rose from `9.16` to `86.03`, crossing `85` at generation
  `287`.
- The champion bundle contains `16` neurons and `88` active synapses.
- Bundle validation found no mismatches between sparse adjacency, browser
  connectome synapses, and importable Colab weight edge identities.

Source:

- `gen5/outputs/colab_500_gen_2026-06-25/evolution_telemetry.json`
- `gen5/outputs/colab_500_gen_2026-06-25/evolution_telemetry.csv`
- `gen5/outputs/colab_500_gen_2026-06-25/evolution_telemetry.png`
- `gen5/outputs/colab_500_gen_2026-06-25/champion_sparse_adjacency.json`
- `gen5/outputs/colab_500_gen_2026-06-25/champion_connectome.json`
- `gen5/outputs/colab_500_gen_2026-06-25/colab_weights.json`
- `gen5/outputs/colab_500_gen_2026-06-25/analysis.md`

Implication:

- The champion is ready for Gen-4 browser replay/injection.
- The evolutionary loop is working, but current mutation settings appear to
  favor topology expansion until saturation.
- Next experiments should test lower sprout probability, stronger low-LTW
  pruning, and/or an explicit active-edge budget pressure term.

### 7. Browser replay validates export integrity but reveals a transducer gap

Finding: the 500-generation champion bundle loaded into the Gen-4 browser and
accepted all `88` Colab LTW updates, but the first observed replay was
motor-silent. The displayed bot velocity stayed at `0.00, 0.00`; no food was
collected; toxin hits occurred when moving toxin objects reached the bot.

Evidence:

- Browser status toast: `GPU memories injected · 88 LTW updated · STW cleared`.
- Browser reported `16` neurons and `88` synapses.
- Inspector showed mean STW `0.00` and mean LTW `0.14`.
- A ~70 second monitor pass observed two sleep/offline-replay phases.
- Food hits: `0`; toxin hits: `2` across observed day/reset cycles.
- Synapses formed/pruned remained `0/0` with plasticity off.

Source:

- `gen5/outputs/colab_500_gen_2026-06-25/browser_champion_monitor_2026-06-25.md`

Implication:

- The exporter and browser importer are compatible.
- The champion's tensor-environment fitness does not yet translate into visible
  Gen-4 browser motor control.
- Next work should focus on the Gen-5 -> Gen-4 sensory/motor transducer mapping
  before using browser replay as a faithful behavioral demonstration.

### 8. Gen-5 bridge replay now actuates, but needs calibrated gain/seeded replay

Finding: after adding the Gen-5 browser transducer bridge and finite motor
guards, the champion replay produced visible motor events and finite movement.
The bot avoided toxin hits in the observed window and came much closer to food,
but still did not collect food.

Evidence:

- Browser status toast again confirmed `88` LTW updates and STW clearing.
- No browser console warnings/errors after the monitor pass.
- Velocity was finite in `15/15` samples.
- Maximum sampled speed was `1.05`.
- Food hits: `0`; toxin hits: `0`; net visible fitness: `0`.
- Closest food approach improved to `17 px`.
- Closest toxin approach stayed at `47 px` or farther.
- Visible motor events included `Motor ←` and `Motor →`.

Source:

- `gen5/outputs/colab_500_gen_2026-06-25/browser_champion_bridge_monitor_2026-06-25.md`

Implication:

- The original browser replay failure was partly a transducer/actuation bridge
  problem, not only a weak champion.
- The bridge now actuates safely, but needs deterministic replay and gain
  calibration before we can judge food-seeking behavior fairly.
- Next implementation should add fixed browser world seeds plus explicit Gen-5
  bridge sensor/motor gain controls or constants.

### 9. Browser replay must match Gen-5 tensor physics before judging champions

Finding: the 10,000-agent Gen-5 run used the default `TensorEnvironmentConfig`
physics constants unless count overrides were supplied by the Colab/example
loop. Browser replay now exposes those values as explicit calibration controls
so a champion can be evaluated against the same sensory and motor scale.

Evidence:

- `TensorEnvironmentConfig.sensor_radius = 0.35`
- `TensorEnvironmentConfig.friction = 0.985`
- `TensorEnvironmentConfig.action_gain = 0.05`
- The 10,000-agent example overrides `agent_count`, `food_count`, and
  `toxin_count`, but not these physics constants.
- Browser calibration sliders were added for sensor radius, drag multiplier,
  and spike-to-velocity multiplier with defaults `0.35`, `0.985`, and `0.05`.

Source:

- `gen5/ammc_gen5/tensor_environment.py`
- `gen5/examples/sprint8_evolving_headless_loop.py`
- `index.html`

Implication:

- Champion browser replay should be run with Gen-5 calibration defaults before
  claiming a behavior-transfer failure.
- The next replay experiment should import the champion connectome and Colab
  weights under these matched constants, then record food/toxin hits, speed,
  and nearest-object distances again.

### 10. Deterministic browser replay is required for champion comparison

Finding: the browser sandbox now includes a seeded replay harness. A replay seed
controls world respawns, neural noise, sleep spindle sampling, mutation choices,
and other practical random paths so repeated champion evaluations can be
compared under the same environment.

Evidence:

- Added a `Replay seed` field and `Seeded replay` button to the Gen-5 replay
  calibration panel.
- Added a deterministic PRNG initialized from the seed.
- Routed world respawns, object phases, neural noise, sleep replay, sprouting,
  and mutation calls through the simulator RNG.
- The loaded champion connectome was verified in the browser before the patch:
  `16` neurons, `88` synapses, mean LTW `0.14`, calibration
  `0.35 / 0.985 / 0.05`.

Source:

- `index.html`

Implication:

- Future browser champion reports should include the replay seed alongside
  sensor radius, drag, spike velocity, food hits, toxin hits, max speed, and
  nearest-object minima.
- The next browser pass should reload the updated app, import the champion
  connectome/weights, run `Seeded replay`, and monitor a fixed duration.

### 11. Browser monitoring requires a visible/foreground tab

Finding: seeded replay monitoring exposed a browser/tooling issue. The champion
run advanced when the user clicked the visible browser, but the automation
surface then reported `document.visibilityState = hidden`, which can freeze
`requestAnimationFrame` while Codex samples DOM metrics. The sandbox now has a
background tick fallback for hidden tabs so monitoring can continue during
agent-driven observations.

Evidence:

- Browser state before replay: `16` neurons, `88` synapses, mean LTW `0.14`.
- Import toast: `GPU memories injected · 88 LTW updated · STW cleared`.
- Calibration: `0.35 / 0.985 / 0.05`.
- Replay seed: `champion-001`.
- After the user clicked `Seeded replay`, sim time reached `23.798 s`, event
  showed `Motor ←`, and velocity was `0.11, 0.22`.
- During agent sampling, DOM visibility returned `hidden` and time stopped
  advancing.
- DOM visibility check returned `hidden`.
- A hidden-tab `setInterval` fallback was added to call the same frame update
  path while `document.visibilityState === "hidden"`.

Source:

- In-app browser monitor, 2026-06-26.
- `index.html`

Implication:

- Do not treat pre-fallback hidden-tab replay samples as behavioral evidence.
- After reload, champion replay monitoring can proceed even if the browser
  surface becomes hidden while Codex samples DOM metrics.

### 12. Seeded champion replay now transfers visible food-seeking behavior

Finding: after restarting the browser, reloading the champion connectome and
Colab weights, and running `Seeded replay` with seed `champion-001`, the Gen-5
champion produced visible motor output and collected food without toxin hits in
the monitored window. The circadian day/night transition and offline replay also
worked cleanly.

Evidence:

- Browser replay state: `16` neurons, `88` synapses, mean STW `0.00`, mean LTW
  `0.14`.
- Calibration matched Gen-5 defaults: sensor radius `0.35`, drag `0.985`, spike
  velocity `0.05`.
- Food was acquired in observed awake windows; toxins remained `0`.
- Sleep replay phases froze bot velocity at `0.00, 0.00` and logged
  `Offline replay - sensory channels muted`.
- No browser console warnings/errors appeared during the pass.
- Plasticity Mode was OFF, so formed/pruned counters stayed `0/0` and the run
  should be interpreted as bridge replay rather than online learning.

Source:

- `gen5/outputs/colab_500_gen_2026-06-25/browser_seeded_replay_monitor_2026-06-26.md`
- In-app browser monitor, 2026-06-26.

Implication:

- The Gen-5 -> Gen-4 champion bridge is no longer merely syntactically valid;
  it can express visible survival behavior under matched replay constants.
- The next controlled experiment should run the same seed with Plasticity Mode
  ON to test whether dopamine events create STW, consolidate LTW, and alter
  topology.
- The reward toast reported `0 eligible synapses reinforced` during this
  no-plasticity pass, so reinforcement eligibility should be checked explicitly
  during the plasticity-enabled run.

### 13. Plasticity replay closes the learning loop but destabilizes the champion

Finding: running the same `champion-001` seeded replay with Plasticity Mode ON
activated structural plasticity, dopamine/GABA modulation, and sleep
consolidation. The champion collected food and sprouted new synapses, but later
hit a toxin after topology changed, showing that unconstrained online
plasticity can perturb an evolved champion.

Evidence:

- Before the main sample window, synapses had already changed from the imported
  `88` to `82`, with `2` formed and `8` pruned.
- During the observed day, synapses increased from `82` to `85` through
  sprouting events including `N7 -> N6.D1`, `N4 -> N3.D3`, and `N4 -> N8.D2`.
- Food/dopamine was observed: fitness reached `+1`, and nearest astrocyte
  state reached `A5 +1.00`.
- Sleep replay pruned one synapse, leaving `84`.
- Dawn logs confirmed STW-to-LTW consolidation:
  `0.316 STW consolidated into LTW` and later
  `0.489 STW consolidated into LTW`.
- A later toxin event drove fitness to `-1.00`, nearest astrocyte state to
  `A5 -0.93`, and toast text to
  `O1 toxin - GABA -1.00 - plasticity suppressed`.
- No browser console warnings/errors appeared during the pass.

Source:

- `gen5/outputs/colab_500_gen_2026-06-25/browser_seeded_plasticity_replay_monitor_2026-06-26.md`
- In-app browser monitor, 2026-06-26.

Implication:

- The closed loop is now visible end-to-end: behavior -> reward/stress ->
  astrocyte modulation -> structural churn -> sleep consolidation.
- Plasticity is too aggressive for champion preservation. Imported champions
  need a stability gate, lower structural churn rates, or dopamine-gated
  sprouting before we claim reliable post-Colab adaptation.
- Diagnostics should expose unrounded STW/LTW changes and per-synapse
  reinforcement eligibility so small consolidation events are measurable.

### 14. Sprint 11 raw outputs verify convergence and plasticity benefit

Finding: raw Sprint 11 multi-seed and plasticity ablation outputs were found in
the user's Downloads folder and copied into the project. They verify that Gen-5
evolution converges consistently across 10 seeds and that structural plasticity
materially outperforms a static SNN under the perturbed foraging task. This was
the first quantitative proof step; retention, throughput, and baseline outputs
were analyzed later in finding 15.

Evidence:

- `1st run/evolution_telemetry.json` contains `500` generations.
- Best single-run max fitness was `24.0`, reached at generations `236` and
  `450`.
- Champion export integrity is valid:
  `16` neurons, `88` active edges, `88` connectome synapses, `88` sparse
  adjacency rows, and `88` Colab weight edges.
- Multi-seed statistical run:
  - `10` seeds: `42-51`.
  - `5,000` trial records.
  - Final mean all-time best fitness: `26.0 +/- 0.667`.
  - Final seed range: `25.0` to `27.0`.
  - Mean-best threshold crossing: `20` by generation `20`, `24` by generation
    `91`, `25` by generation `178`, and `26` by generation `446`.
- Plasticity ablation:
  - `static_snn`: final mean best `13.6 +/- 0.843`, active synapses `8.0`.
  - `full_plasticity_infant`: final mean best `25.9 +/- 0.994`, active
    synapses `85.95`.
  - `gated_plasticity_adult`: final mean best `24.6 +/- 1.075`, active
    synapses `67.0`.
  - Static SNN reached `>=20` fitness in `0/10` seeds.
  - Full plasticity reached `>=25` fitness in `10/10` seeds.
  - Gated adult plasticity reached `>=25` fitness in `5/10` seeds.
- At this stage, no raw `retention_ablation.*`, `throughput_results.*`, or
  `baseline_comparison.*` files were available yet.

Source:

- `gen5/outputs/phase11_uploaded_outputs_review_2026-06-26.md`
- `gen5/outputs/phase11_colab_2026-06-26/multi_seed_trials.json`
- `gen5/outputs/phase11_colab_2026-06-26/multi_seed_aggregate.csv`
- `gen5/outputs/phase11_colab_2026-06-26/plasticity_ablation.json`
- `gen5/outputs/phase11_colab_2026-06-26/plasticity_ablation_summary.csv`
- `1st run/evolution_telemetry.json`
- `1st run/champion_sparse_adjacency.json`
- `1st run/champion_connectome.json`
- `1st run/colab_weights.json`
- Uploaded images in `C:\Users\FAISAL TABREZ\.codex\attachments\f42f063e-171e-49e6-9543-fd631425666d\`

Implication:

- The convergence story is now statistically supported across 10 seeds.
- Plasticity benefit over static topology is raw-data supported.
- The current ablation does not prove the "adult/gated" thesis by raw fitness:
  full plasticity outperforms gated plasticity in final best fitness and speed
  to high thresholds.
- Gated adult plasticity is more synapse-efficient: it achieves `24.6` final
  mean best fitness with roughly `67` active synapses versus full plasticity's
  `25.9` with roughly `86` active synapses.
- Catastrophic forgetting still required an original -> perturbed -> original
  retention protocol, which was then run and analyzed in finding 15.

### 15. Phase 11 evidence bundle is complete; retention and throughput are now quantified

Finding: the uploaded `phase11_remaining_outputs.zip` completed the Phase 11
evidence bundle. The verifier now finds champion export integrity, multi-seed
convergence, plasticity ablation, retention ablation, throughput scaling, and
baseline comparison artifacts. The results strengthen the Gen-5 convergence and
throughput story, while also showing that the current gated/adult plasticity
policy is not yet superior to full plasticity on retention.

Evidence:

- Uploaded archive unpacked to:
  `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/`.
- `verify_phase11_outputs.py` reports all Phase 11 groups complete:
  `champion`, `multi_seed`, `plasticity_ablation`, `retention_ablation`,
  `throughput`, and `baselines`.
- Retention ablation:
  - `15,000` records.
  - `static_snn`: original final epoch best `8.3`, perturbation peak `13.3`,
    recovery final `8.7`, retention ratio `1.048`, active synapses `8.0`.
  - `full_plasticity_infant`: original final `19.6`, perturbation peak `25.3`,
    recovery final `18.9`, retention ratio `0.964`, active synapses `85.90`.
  - `gated_plasticity_adult`: original final `18.4`, perturbation peak `24.2`,
    recovery final `17.0`, retention ratio `0.924`, active synapses `67.54`.
- Throughput benchmark on CUDA with `torch.compile` active:
  - `1k` agents: `599,600` agent-steps/sec, `8.89 MB` CUDA max memory.
  - `10k` agents: `5.95M` agent-steps/sec, `53.47 MB`.
  - `50k` agents: `22.48M` agent-steps/sec, `249.14 MB`.
  - `100k` agents: `29.29M` agent-steps/sec, `486.65 MB`.
- Baseline comparison:
  - `ammc_sparse_evolver`: `3.70M` agent-steps/sec, max fitness `20`.
  - `dense_lif_snn`: `2.42M` agent-steps/sec, max fitness `4`.
  - `dense_mlp_policy`: `2.90M` agent-steps/sec, max fitness `3`.
  - `ppo_mlp_policy`: skipped because `stable-baselines3` was unavailable.

Source:

- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/retention_ablation/retention_ablation_summary.csv`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/throughput/throughput_results.json`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/baselines/baseline_comparison.json`
- `gen5/outputs/phase11_uploaded_outputs_review_2026-06-26.md`

Implication:

- Multi-seed convergence and plasticity benefit are now backed by a complete
  Phase 11 artifact set.
- Full plasticity currently beats gated/adult plasticity in raw adaptation and
  retention. Gated/adult remains useful as a sparsity/efficiency direction, but
  its current rule is too restrictive or is gating the wrong events.
- Static topology is not competitive; its high retention ratio is misleading
  because it starts from a low original-task score.
- Throughput at `100k` agents is strong enough to justify continuing the
  vectorized CUDA-first path.
- The baseline comparison is only partial: dense LIF/MLP inference scaffolds
  ran, but trained BPTT SNN and trained PPO remain future fair baselines.
- Next performance proof should benchmark saturated champion-like topologies,
  not only the `8` active-synapse benchmark prior.

### 16. Gen-5 must transition to TPU/XLA-first execution

Finding: the near-term accelerator constraint has changed the backend
priority. Gen-5 should be shaped around Colab TPU/XLA execution now, with T4
CUDA kept as a standard PyTorch fallback and custom CUDA kernels deferred.

Evidence:

- The existing Gen-5 code had CUDA/CPU assumptions in evaluation, examples,
  throughput benchmarks, and baseline scripts.
- TPU/XLA needs static-shape discipline:
  - avoid per-step `.item()` synchronization,
  - avoid `bool(mask.any())` branches in hot loops,
  - avoid dynamically sized tensors from `mask.sum()`,
  - prefer fixed-capacity pools and masked updates.
- The project already uses fixed-capacity sparse edge pools, which maps well to
  XLA compared with true runtime tensor resizing.

Changes made:

- Added `gen5/ammc_gen5/runtime.py` as the central backend abstraction:
  `resolve_device`, `mark_step`, `sync`, `seed_everything`, backend memory
  wrappers, and XLA detection.
- `--device auto` now chooses XLA first when PyTorch/XLA can acquire a device,
  then CUDA, then CPU.
- `TensorEnvironment2D._respawn(...)` now uses fixed-shape masked respawns.
- `TensorEvolver.mutate_children(...)` now generates fixed-shape sprout
  candidates and masks them in.
- `EvolvingHeadlessAMMCLoop` now uses host-side epoch/generation counters to
  avoid per-step `.item()` syncs.
- Headless loops call backend-aware XLA step markers.
- Throughput and baseline scripts now accept `--device xla`.
- Added `gen5/docs/TPU_XLA_MIGRATION.md`.

Implication:

- XLA is now the primary architecture target for Gen-5 validation.
- CUDA custom-kernel design should wait until the XLA-compatible algorithmic
  surface stabilizes.
- The next benchmark pass should rerun Phase 11 on `--device xla` and compare
  convergence, retention, and throughput against the existing CUDA/T4 evidence.

Runtime caveat added 2026-06-27:

- `--device xla` requires `torch_xla` to be importable in the active Colab
  runtime. If `torch_xla` is missing, the issue is a Colab TPU/PyTorch-XLA
  setup problem, not an AMMC graph error.
- `_XLAC` / `undefined symbol` import failures mean `torch_xla` is installed
  but binary-incompatible with the active `torch` wheel.
- AMMC now raises an actionable dependency message instead of a raw
  `ModuleNotFoundError` or binary loader traceback.
- T4/L4 notebooks should use `--device cuda`; TPU notebooks should pass the
  PyTorch/XLA preflight import check before running benchmarks.

### 17. Throughput benchmarks must report topology load

Finding: the first Phase 11 throughput result used the original `8` active-edge
foraging prior, while the evolved champion saturates near `86-88` active
synapses. Population size alone is therefore not enough to characterize Gen-5
runtime cost.

Evidence:

- Phase 11 throughput reached `29.29M` agent-steps/sec at `100k` agents, but
  the benchmark reported `8.0` mean active synapses.
- The exported champion bundle contains `88` active sparse edges.
- Earlier evolution telemetry shows mean active synapses rising into the high
  `80s`, so champion-like operation is roughly an order of magnitude denser
  than the original throughput benchmark.

Change made:

- `gen5/benchmarks/benchmark_throughput.py` now supports topology presets:
  - `foraging`: original `8`-edge seed prior,
  - `saturated`: synthetic champion-like fixed active edge count,
  - `champion`: load an exported `champion_sparse_adjacency.json`.
- Output rows now include:
  - `topology_preset`,
  - `requested_active_edges`,
  - `seeded_active_edges`,
  - `adjacency_json`.

Implication:

- Future throughput claims should always include both population size and
  active-edge load.
- The next Colab TPU/XLA benchmark should run at least:
  - `--topology-preset foraging`,
  - `--topology-preset saturated --active-edges 86`,
  - optionally `--topology-preset champion --adjacency-json ...`.

Follow-up result added 2026-06-27:

- Saturated CUDA benchmark with `86` active edges completed.
- At `100k` agents, saturated throughput was `4.64M` agent-steps/sec with
  `883.84 MB` CUDA max memory.
- Compared with the earlier `8`-edge CUDA benchmark, saturated throughput was:
  - `72.4%` at `1k`,
  - `68.6%` at `10k`,
  - `20.5%` at `50k`,
  - `15.9%` at `100k`.
- Saturated memory at `100k` was `1.82x` the `8`-edge benchmark.
- The saturated throughput curve plateaus from `50k` to `100k`, suggesting
  scatter/memory-bandwidth pressure under high active-edge load.

Implication:

- Champion-like topologies are still viable at large population size, but the
  performance story must be topology-aware.
- Future benchmark tables must include active-edge count, not just population
  size and device.
- Next benchmark: rerun saturated CUDA with `--compile`, then run exact
  `champion_sparse_adjacency.json` topology.

Artifact:

- `gen5/outputs/throughput_cuda_saturated_2026-06-27/analysis.md`

Compile follow-up added 2026-06-27:

- Saturated CUDA `--compile` run completed on the same `86` active-edge
  topology.
- At `100k` agents, compiled saturated throughput was `4.62M`
  agent-steps/sec with `883.84 MB` CUDA max memory.
- Compiled/eager throughput ratio was effectively flat:
  - `1.013x` at `1k`,
  - `0.994x` at `10k`,
  - `0.996x` at `50k`,
  - `0.995x` at `100k`.
- The run emitted a Torch Dynamo recompile-limit warning from
  `EvolvingHeadlessAMMCLoop.step()` because the full training step mutates the
  Python integer `_epoch_step_host`, which Dynamo treats as a static
  `nn.Module` guard.

Decision:

- Keep full `step()` semantics unchanged for real evolution and telemetry.
- Add a separate `benchmark_tick()` tensor hot path for throughput timing and
  `torch.compile`.
- Future throughput rows now record
  `tick_mode: tensor_hot_path_no_epoch_control` so compiler-hot-path results
  are not confused with full evolutionary training-step diagnostics.

Implication:

- The uploaded compiled saturated result is diagnostic evidence, not the final
  compiler-performance claim.
- The next publishable benchmark should rerun eager and compiled saturated
  throughput on the same `benchmark_tick()` path, then repeat with the exact
  champion adjacency.

Artifacts:

- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/analysis.md`
- `gen5/benchmarks/benchmark_throughput.py`
- `gen5/ammc_gen5/evolving_loop.py`

Hotpath rerun added 2026-06-27:

- The patched `benchmark_tick()` compiled CUDA run completed with
  `tick_mode: tensor_hot_path_no_epoch_control`.
- At `100k` agents and `86` active edges, throughput reached `39.15M`
  agent-steps/sec with `488.19 MB` CUDA max memory.
- Compared with the prior full-step compiled saturated run, hotpath throughput
  improved by:
  - `4.994x` at `1k`,
  - `5.448x` at `10k`,
  - `8.377x` at `50k`,
  - `8.476x` at `100k`.
- CUDA max memory at `100k` fell from `883.84 MB` to `488.19 MB`.

Interpretation:

- The earlier saturated `--compile` result did not show that the sparse AMMC
  math was compiler-resistant. It showed that full training-step diagnostics,
  return payloads, and Python host-control state were dominating the measured
  path.
- The clean compiled hotpath result is now the strongest throughput evidence
  for pure vectorized AMMC compute.
- Full-step benchmarks remain useful, but they should be labelled as
  training-loop overhead measurements rather than raw brain/environment
  throughput.

Next action:

- Run the same `benchmark_tick()` path without `--compile` to separate the
  compiler speedup from the no-telemetry/no-host-control speedup.
- The explicit no-telemetry environment step mode has been implemented:
  `TensorEnvironment2D.step(..., collect_telemetry=False)`.
- `benchmark_tick()` now uses `collect_telemetry=False`, so eager and compiled
  hotpaths can measure the same intended workload instead of depending on
  compiler dead-code elimination of unused diagnostics.

Artifact:

- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/analysis.md`
- `gen5/ammc_gen5/tensor_environment.py`

Eager hotpath control added 2026-06-27:

- Saturated CUDA eager hotpath completed with the same
  `tick_mode: tensor_hot_path_no_epoch_control`.
- At `100k` agents and `86` active edges, eager hotpath reached `4.71M`
  agent-steps/sec with `883.84 MB` CUDA max memory.
- Clean compiled/eager hotpath throughput ratios:
  - `5.053x` at `1k`,
  - `5.259x` at `10k`,
  - `8.242x` at `50k`,
  - `8.320x` at `100k`.
- Eager hotpath was only `1.001x` to `1.030x` faster than previous full-step
  eager runs, so the massive compiled hotpath improvement is not just a
  telemetry-removal artifact.

Implication:

- The publishable CUDA saturated-topology statement is now:
  `39.15M` compiled agent-steps/sec versus `4.71M` eager agent-steps/sec at
  `100k` agents on the same tensor hotpath.
- `torch.compile` is valuable for the CUDA path even before custom kernels,
  especially at large population sizes where it reduces intermediate allocation
  pressure.

Artifact:

- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/analysis.md`

Champion-path robustness added 2026-06-27:

- A CUDA champion benchmark attempt failed because the command used
  `gen5_outputs/champion/champion_sparse_adjacency.json`, but the repository
  archive stores the known champion at
  `gen5/outputs/colab_500_gen_2026-06-25/champion_sparse_adjacency.json`.
- `benchmark_throughput.py` now gives a candidate-discovery diagnostic for
  missing champion adjacency paths and tells Colab users to run:
  `find /content -name champion_sparse_adjacency.json -print`.
- The Colab runbook now includes an exact CUDA champion hotpath command with an
  `ADJ_PATH` variable.

Implication:

- Throughput artifacts should record the exact adjacency path used, because
  "champion" can refer to either the archived 500-generation champion or a
  freshly exported runtime champion.
- Future benchmark commands should either use the repository archive path or
  explicitly locate the current Colab export before running.

Artifacts:

- `gen5/benchmarks/benchmark_throughput.py`
- `gen5/docs/PHASE11_COLAB_RUNBOOK.md`

Champion compiled hotpath result added 2026-06-27:

- Exact current champion topology benchmark completed on CUDA with
  `torch.compile` active and `tick_mode: tensor_hot_path_no_epoch_control`.
- The run used `gen5_outputs/champion/champion_sparse_adjacency.json` and
  seeded `55` active edges, so it represents the fresh/current champion export,
  not the older archived `88`-synapse champion.
- At `100k` agents, the champion reached `37.25M` agent-steps/sec with
  `488.19 MB` CUDA max memory.
- Compared with the saturated `86`-edge compiled hotpath:
  - `63.8%` throughput at `1k`,
  - `35.2%` at `10k`,
  - `78.6%` at `50k`,
  - `95.2%` at `100k`.

Interpretation:

- The champion topology is highly scalable on the compiled CUDA hotpath.
- The result exposes an important backend truth: active-edge count is currently
  biological sparsity, not fully hardware sparsity. `TensorEvolver` stores
  genomes as fixed `[population, max_edges]` tensors, and the benchmark used
  `max_edges=128`, so the `55` active-edge champion still executes inside a
  128-slot edge pool.
- Future benchmark outputs now include `edge_pool_capacity` and
  `active_edge_utilization` so active-edge claims cannot be confused with
  physical kernel work.

Next action:

- Run champion eager hotpath for the exact-topology compiler control.
- Run a champion capacity sweep with `--max-edges 64`, `96`, and `128` to
  quantify fixed-pool overhead.

Artifact:

- `gen5/outputs/throughput_cuda_champion_compile_hotpath_2026-06-27/analysis.md`

Champion eager hotpath result added 2026-06-27:

- Champion CUDA eager hotpath completed with `torch.compile` disabled and
  `tick_mode: tensor_hot_path_no_epoch_control`.
- The run used the displayed path
  `gen5_outputs/champion/champion_sparse_adjacency.json`, but seeded `83`
  active edges, while the previous compiled champion run using the same
  displayed path seeded `55` active edges.
- At `100k` agents, the `83`-edge champion eager hotpath reached `5.69M`
  agent-steps/sec with `853.89 MB` CUDA max memory.
- Compared with the saturated `86`-edge eager hotpath:
  - `1.131x` throughput at `1k`,
  - `1.175x` at `10k`,
  - `1.209x` at `50k`,
  - `1.209x` at `100k`.

Interpretation:

- This is valid evidence for the fresh `83`-edge champion eager runtime.
- It is not a valid compiled/eager pair against the previous `55`-edge compiled
  champion run. The same Colab display path can point to different champion
  payloads across sessions or exports.
- Exact-topology benchmark comparisons now require a topology fingerprint.
  `benchmark_throughput.py` therefore records `resolved_adjacency_json` and
  `adjacency_sha256` for future champion runs.

Next action:

- Rerun both champion eager and champion compiled hotpath after the SHA-256
  schema patch, using the same printed `adjacency_sha256`.
- Only then report a champion-specific compiled/eager speedup.

Artifact:

- `gen5/outputs/throughput_cuda_champion_eager_hotpath_2026-06-27/analysis.md`
- `gen5/benchmarks/benchmark_throughput.py`

Fingerprint-matched champion compiled run added 2026-06-27:

- Corrected fingerprinted champion compiled hotpath files were uploaded and
  verified against JSON and CSV.
- Adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`.
- Active edges: `86`.
- Edge pool capacity: `128`.
- Active edge utilization: `67.19%`.
- At `100k` agents, compiled champion throughput reached `35.35M`
  agent-steps/sec with `488.19 MB` CUDA max memory.
- Compared with saturated 86-edge compiled hotpath, the fingerprinted champion
  achieved:
  - `75.7%` throughput at `1k`,
  - `96.3%` at `10k`,
  - `89.9%` at `50k`,
  - `90.3%` at `100k`.

Interpretation:

- This is now the strongest champion-specific compiled throughput artifact.
- The current champion topology is near saturated-topology throughput at scale,
  but not identical: exact source/target scatter structure matters even when
  active edge count and edge-pool capacity match.
- The old `55`-edge non-fingerprinted champion compiled run should be treated
  as historical diagnostic evidence, not the current publishable champion
  result.
- Memory equality with saturated compiled throughput reinforces that the
  current fixed-capacity edge pool controls memory footprint.

Next action:

- Rerun champion eager hotpath and require the same `adjacency_sha256` before
  reporting champion-specific compiled/eager speedup.

Artifact:

- `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/analysis.md`

### 18. Literature scan: AMMC is likely unique as an integration, not as individual mechanisms

Finding: a first-pass literature scan shows strong prior art for nearly every
individual AMMC ingredient: structural plasticity, dynamic sparse rewiring,
astrocyte-modulated SNNs, dopamine-STDP embodied robots, sleep/replay
consolidation, and neuroevolution. The strongest defensible novelty is the
system-level integration of these mechanisms into a benchmarkable, serialized,
embodied, tensorized organism framework.

Nearby prior art:

- STDP pruning and energy-efficient SNN compression: Rathi et al. 2017.
- Adaptive/evolved SNN structure with dopamine-modulated plasticity: Pan et
  al. 2023.
- Dynamic sparse rewiring: DEEP R, RigL, dynamic sparse RL.
- Astrocyte-modulated computation: Tewari and Majumdar 2011, Tang et al. 2019,
  Shen et al. 2023, Yang et al. 2025.
- Sleep/replay SNNs: Whelan et al. 2021, Pietras et al. 2022, Massey et al.
  2026.
- Neuroevolution/co-evolution: NEAT, competitive coevolution, TensorNEAT.

Implication:

- Do not claim AMMC is the first SNN with pruning, astrocytes, replay, or
  evolution.
- Claim AMMC as an integrated sparse spiking-organism framework for embodied
  continual learning, where structural plasticity, chemical modulation, sleep
  consolidation, and evolutionary populations are tested together.
- Publication strength will come from ablations showing that the integrated
  stack outperforms component-stripped variants and fair baselines.

Artifact:

- `gen5/docs/LITERATURE_UNIQUENESS_REVIEW.md`

## Project decisions

### Decision: Gen-5 is a backend framework, not another visual simulator

Date: 2026-06-25

Decision:

Gen-5 will be implemented as a Python/C++ mathematical framework designed for
GPU/TPU/neuromorphic hardware. The browser remains a Gen-4 proof and visual
debugging sandbox.

Rationale:

- Gen-4 proved the biological mechanics.
- Production scale requires vectorized tensor execution and custom sparse
  kernels.
- Visual rendering should not be in the core runtime path.

Artifacts:

- `gen5/README.md`
- `gen5/docs/GEN5_ARCHITECTURE.md`

### Decision: Start with fixed-capacity sparse edge pools in Python

Date: 2026-06-25

Decision:

The Sprint 1 prototype uses fixed-capacity edge slots plus active masks. This
preserves PyTorch optimizer compatibility while exposing sprouting and pruning
semantics.

Rationale:

- PyTorch parameter shapes cannot safely mutate mid-backward.
- Fixed-capacity pools let us test sparse dynamics today.
- C++/CUDA can later replace the storage layer with true memory allocation and
  compaction.

Artifacts:

- `gen5/ammc_gen5/dynamic_sparse.py`
- `gen5/tests/test_dynamic_sparse_contract.py`

### Decision: Keep dual-frequency processing explicit

Date: 2026-06-25

Decision:

Gen-5 will model fast electrical and slow chemical computation as separate but
coupled tensor systems.

Rationale:

- This preserves the biological separation observed in Gen-4.
- It provides an implementation boundary for future CUDA kernels.
- It avoids mixing high-rate event propagation with low-rate modulation logic.

Artifacts:

- `gen5/ammc_gen5/dual_tensor.py`

### Decision: Build vectorized embodiment before CUDA allocation

Date: 2026-06-25

Decision:

Sprint 4/5 adds a PyTorch-native 2D tensor environment and a vectorized
transducer before the custom C++/CUDA dynamic allocator.

Rationale:

- Gen-5 needs scalable embodied feedback to evaluate plastic brains.
- A 10,000-agent tensor environment lets us test swarm learning loops on CUDA
  even before custom kernels exist.
- The transducer establishes the environment-to-brain-to-action contract that
  future Isaac Gym integration can reuse.

Artifacts:

- `gen5/ammc_gen5/tensor_environment.py`
- `gen5/ammc_gen5/transducer.py`
- `gen5/examples/sprint4_5_vectorized_loop.py`
- `gen5/tests/test_tensor_environment_contract.py`

### Decision: Evolve per-agent sparse genomes as batched tensors

Date: 2026-06-25

Decision:

Sprint 6/7 adds `TensorEvolver`, a batched sparse genome pool shaped
`[population, max_edges]`. It performs culling, survivor broadcasting, LTW noise,
random pruning, and random sprouting with tensor operations.

Rationale:

- A 10,000-agent swarm cannot copy Python organism objects at epoch boundaries.
- The evolutionary loop must stay in VRAM with tensor indexing and masks.
- Per-agent sparse genomes are required for true co-evolution; a single shared
  `DynamicSparseLinear` is only a transitional vectorized baseline.

Artifacts:

- `gen5/ammc_gen5/evolver.py`
- `gen5/examples/sprint6_7_tensor_evolver.py`
- `gen5/tests/test_tensor_evolver_contract.py`

### Decision: Bind environment, transducer, and evolver into one epoch loop

Date: 2026-06-25

Decision:

Sprint 8 adds `EvolvingHeadlessAMMCLoop`, the central Gen-5 runtime cycle:

`TensorEnvironment2D` physics -> `VectorizedTransducer` sensors ->
`TensorEvolver` per-agent brains -> `VectorizedTransducer` motors ->
`TensorEnvironment2D` actions.

At `epoch_steps`, the loop reads environment fitness, calls
`TensorEvolver.evolve(fitness)`, resets environment positions/scores, clears
membrane state, and advances the generation counter.

Rationale:

- This is the first complete headless evolutionary organism cycle.
- Per-agent genome evolution now happens from actual environment fitness.
- The loop remains tensorized over agents; only the outer clock advances in
  Python.

Artifacts:

- `gen5/ammc_gen5/evolving_loop.py`
- `gen5/examples/sprint8_evolving_headless_loop.py`
- `gen5/tests/test_evolving_loop_contract.py`

### Decision: Treat epoch telemetry as the headless visual layer

Date: 2026-06-25

Decision:

Gen-5 will use `EvolutionTelemetryLogger` as the first observability layer for
headless evolution. It records max fitness, mean population fitness, mean active
synapses, sprout counts, prune counts, and LTW mutation counts each epoch, then
exports JSON/CSV and optional matplotlib plots.

Rationale:

- Gen-5 has no UI, so fitness and topology curves are the visual evidence.
- Epoch-level telemetry is compact enough for 500+ generation Colab runs.
- Mean active synapses tells us whether evolution is discovering efficient
  sparse topology or simply bloating the edge pool.

Artifacts:

- `gen5/ammc_gen5/telemetry.py`
- `gen5/tests/test_telemetry_contract.py`
- `gen5/examples/sprint8_evolving_headless_loop.py`

### Decision: Export champions as a three-file Gen-5 -> Gen-4 bridge

Date: 2026-06-25

Decision:

Gen-5 champion export will emit three synchronized artifacts:

- `champion_connectome.json` for loading the champion topology into the Gen-4
  browser sandbox.
- `colab_weights.json` for overwriting browser LTW values with the champion's
  long-term weights.
- `champion_sparse_adjacency.json` for analysis, audits, and future benchmark
  pipelines.

The export process snapshots the all-time best organism before culling/mutation
at epoch boundaries, then maps the sparse Gen-5 genome into Gen-4-compatible
neuron IDs, dendrite IDs, and synapse records.

Rationale:

- A weights-only payload cannot reconstruct topology.
- A connectome-only payload may not prove exact LTW injection semantics.
- The raw adjacency file keeps the mathematical champion available even if the
  browser compatibility layer changes later.
- Gen-5 and Gen-4 do not yet share identical sensor semantics, so the exporter
  explicitly records the bridge mapping.

Artifacts:

- `gen5/ammc_gen5/champion_export.py`
- `gen5/ammc_gen5/evolving_loop.py`
- `gen5/ammc_gen5/evolver.py`
- `gen5/tests/test_champion_export_contract.py`
- `gen5/examples/sprint8_evolving_headless_loop.py`

### Decision: Add explicit Gen-5 browser transducer bridge mode

Date: 2026-06-25

Decision:

Gen-4 browser replay now treats imported Gen-5 champion connectomes as a
compatibility mode rather than plain Gen-4 organisms. The browser preserves
separate food/toxin directional sensor channels and applies a scoped analog
motor-readout assist for imported Gen-5 organisms.

Rationale:

- Gen-5 tensor runs use eight sensor channels:
  `food north/east/south/west` and `toxin north/east/south/west`.
- The browser previously blended food attraction and toxin avoidance into only
  four directional channels.
- The first champion replay showed valid topology/weight injection but no
  visible motor movement, so the demonstration gap was in transduction rather
  than serialization.

Artifacts:

- `index.html`
- `gen5/ammc_gen5/champion_export.py`
- `gen5/tests/test_champion_export_contract.py`

Validation:

- `python -m compileall gen5` passed.
- `python -m unittest discover -s gen5\tests -v` passed.
- Browser reload of `http://127.0.0.1:4173/` produced no console errors.

### Decision: Advance to controlled evaluation and champion-stability phase

Date: 2026-06-26

Decision:

The project has enough foundational evidence to move beyond mechanism-building
into the next major phase: controlled evaluation, champion stability, and
benchmark preparation.

This is not yet the phase for claiming AMMC outperforms established SNNs or
Transformers. It is the phase for producing statistically defensible evidence,
stabilizing post-Colab plasticity, and creating comparison baselines.

Rationale:

- Gen-4 validated the biological pillars visually:
  structural plasticity, astrocyte modulation, embodiment, sleep replay,
  serialization, PyTorch export, and weight re-import.
- Gen-5 has the first headless mathematical scaffold:
  dynamic sparse pools, batched tensor environment, transducer, evolver,
  epoch loop, telemetry, and champion exporter.
- A 500-generation Colab run produced a valid champion bundle and telemetry.
- Seeded browser replay now shows visible champion behavior under matched
  Gen-5 constants.
- Plasticity-enabled replay closes the loop from behavior to dopamine/GABA,
  structural churn, and sleep consolidation.
- The main risk has shifted from "can the mechanism exist?" to "can we tune,
  measure, and compare it rigorously?"

Phase boundary:

- Proceed to evaluation/stability engineering.
- Do not yet claim state-of-the-art performance.
- Treat superiority claims as blocked until controlled baselines, repeated
  seeded trials, runtime benchmarks, and ablations are complete.

Immediate phase goals:

1. Add champion-stability controls for imported Gen-5 organisms.
2. Add deterministic multi-seed browser and Gen-5 evaluation harnesses.
3. Add baseline comparisons against random, non-plastic, plastic, and
   champion-stability variants.
4. Add performance benchmarks for environment steps/sec and epoch throughput.
5. Add unrounded memory/topology diagnostics so small STW/LTW changes are
   measurable.

Artifacts supporting the decision:

- `gen5/outputs/colab_500_gen_2026-06-25/evolution_telemetry.json`
- `gen5/outputs/colab_500_gen_2026-06-25/analysis.md`
- `gen5/outputs/colab_500_gen_2026-06-25/browser_seeded_replay_monitor_2026-06-26.md`
- `gen5/outputs/colab_500_gen_2026-06-25/browser_seeded_plasticity_replay_monitor_2026-06-26.md`

### Decision: Implement Sprint 11 quantitative proof harnesses

Date: 2026-06-26

Decision:

Sprint 11 will establish reproducible statistical and hardware benchmarking
pipelines before any AMMC performance claims are made. The implementation adds:

- `TrialRunner` for multi-seed evolutionary convergence trials.
- `PlasticityAblationRunner` for static/full/gated plasticity comparisons.
- Positive-reward gates in `TensorEvolver` so "adult" plasticity can suppress
  pruning and LTW noise unless parent fitness crosses a dopamine-like
  threshold.
- Colab-facing examples for 10-seed / 500-generation evaluation runs.
- Throughput benchmarks for 1k, 10k, 50k, and 100k population scaling.
- Baseline comparison scaffolds for AMMC sparse, dense LIF-style SNN, dense
  MLP, and dependency-gated PPO.

Rationale:

- A single champion is anecdotal; convergence must be measured across seeds.
- Plasticity must be tested as an ablation, not assumed beneficial.
- Hardware efficiency must be measured as ticks/sec and agent-steps/sec.
- External baseline claims require shared task, shared metrics, and dependency
  visibility.

Artifacts:

- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint11_statistical_evaluation.py`
- `gen5/examples/sprint11_plasticity_ablation.py`
- `gen5/benchmarks/benchmark_throughput.py`
- `gen5/benchmarks/comparison_baselines.py`
- `gen5/tests/test_evaluation_contract.py`

Validation:

- Bundled Python syntax compile passed: `python -m compileall gen5`.
- Unit suite passed locally with PyTorch-dependent tests skipped because the
  desktop bundled Python lacks PyTorch:
  `python -m unittest discover -s gen5\tests -v`.

### Decision: Add retention/forgetting ablation protocol

Date: 2026-06-26

Decision:

The existing plasticity ablation proves adaptation under a perturbed sensor
mapping, but it does not measure catastrophic forgetting. Add a three-phase
retention protocol:

1. original environment,
2. perturbed environment with food/toxin sensor inversion,
3. original environment again.

The new runner records phase-local best fitness, final recovery fitness,
retention ratio, forgetting delta, and final active synapse count.

Rationale:

- The raw ablation shows full plasticity has higher perturbed fitness than
  gated adult plasticity.
- The adult/gated hypothesis is not "highest raw adaptation at any cost"; it is
  "adapt while preserving useful prior structure."
- That claim requires an explicit return-to-original retention measurement.

Artifacts:

- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint11_retention_ablation.py`
- `gen5/tests/test_evaluation_contract.py`
- `gen5/README.md`

Validation target:

- Local syntax validation should pass.
- PyTorch contract test should run in Colab or another environment with
  PyTorch installed.

### Decision: Add Phase 11 evidence verifier

Date: 2026-06-26

Decision:

Add a standard-library verifier that scans output folders and reports which
Phase 11 artifact groups are complete. The verifier summarizes champion,
multi-seed, plasticity ablation, retention ablation, throughput, and baseline
outputs when their raw files are present.

Rationale:

- Manual inspection missed raw files that were in Downloads rather than the
  workspace.
- Completion claims now require a repeatable artifact audit.
- The verifier cleanly distinguishes proven groups from missing groups.

Current verifier result after `phase11_remaining_outputs.zip` upload:

- Complete: `champion`, `multi_seed`, `plasticity_ablation`,
  `retention_ablation`, `throughput`, `baselines`.
- Missing: none for the Phase 11 artifact checklist.

Artifact:

- `gen5/tools/verify_phase11_outputs.py`

### Decision: Close Phase 11 evidence gate and move to tuning plus fair baselines

Date: 2026-06-26

Decision:

Treat Phase 11 as evidence-complete for the first quantitative benchmark pass,
but do not claim final superiority over trained external methods yet. The next
major workstream should focus on:

- saturated-topology throughput,
- trained BPTT SNN and PPO baselines,
- gated/adult plasticity redesign,
- active-edge pressure and protected-core champion stability.

Rationale:

- Multi-seed convergence is proven at `26.0 +/- 0.667` final mean best fitness.
- Structural plasticity beats static topology by a large margin.
- Full plasticity beats current gated/adult plasticity on both perturbation
  adaptation and recovery retention.
- CUDA throughput reached `29.29M` agent-steps/sec at `100k` agents, but only
  under the low-active-edge benchmark prior.
- Baseline comparison ran scaffold LIF/MLP baselines, while PPO was skipped and
  trained dense baselines remain incomplete.

Artifacts:

- `gen5/outputs/phase11_uploaded_outputs_review_2026-06-26.md`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/`
- `gen5/tools/verify_phase11_outputs.py`

### Decision: Make TPU/XLA the near-term Gen-5 accelerator architecture

Date: 2026-06-26

Decision:

Move Gen-5 from CUDA-first assumptions to a TPU/XLA-first runtime architecture.
Keep standard PyTorch CUDA/T4 compatibility, but defer custom CUDA kernels until
after the XLA-compatible fixed-pool design and benchmark claims stabilize.

Rationale:

- The user's available near-term accelerator path is Colab TPU/XLA rather than
  custom CUDA kernel development.
- XLA rewards the same static-capacity sparse pools that Gen-5 already uses for
  dynamic topology.
- CUDA allocator work is still valuable, but it should not block statistical
  proof, retention studies, or trained baseline comparisons.

Implementation direction:

- Centralize backend behavior in `ammc_gen5.runtime`.
- Prefer explicit `--device xla` in Colab TPU commands.
- Treat `--device auto` as XLA -> CUDA -> CPU.
- Replace dynamic hot-loop allocation/branching with static-shape masked
  tensor updates.
- Separate accelerator-neutral algorithm claims from backend-specific kernel
  claims.

Artifacts:

- `gen5/ammc_gen5/runtime.py`
- `gen5/docs/TPU_XLA_MIGRATION.md`
- `gen5/README.md`
- `gen5/docs/PHASE11_COLAB_RUNBOOK.md`

### Decision: Publish the project to GitHub and push changes continuously

Date: 2026-06-26

Decision:

Initialize/publish this workspace to `FaisalTabrez/A-SNN` and keep the remote
repository updated whenever project changes are made.

Rationale:

- The project now has enough code, evidence, and research state that local-only
  storage is risky.
- Future benchmark runs, TPU/XLA migration changes, and research conclusions
  should be versioned with traceable commits.
- `research.md` remains the living project memory and should travel with every
  meaningful code or decision change.

Operational rule:

- For future implementation or research updates, stage the intended files,
  commit with a concise message, and push to the GitHub remote after validation.
- Do not stage generated caches such as `__pycache__`.
- If a future output bundle is very large, decide explicitly whether it belongs
  in git or should move to release/artifact storage.

Artifact:

- `README.md`
- Git remote target: `https://github.com/FaisalTabrez/A-SNN.git`

## Current implementation state

### Gen-4 browser sandbox

Status: functional proof-of-concept.

Capabilities implemented:

- structural plasticity
- astrocyte overlay
- embodied PIP world
- sleep/replay consolidation
- connectome save/load
- PyTorch export
- Colab weight import
- swarm/evolution scaffolding

### Gen-5 backend scaffold

Status: initial scaffold added.

Capabilities added:

- `DynamicSparseLinear`
- `DynamicSparseLinearFunction`
- `LTWSTWMemory`
- `DualTensorManager`
- `TensorEnvironment2D`
- `VectorizedTransducer`
- `HeadlessAMMCLoop`
- `TensorEvolver`
- `EvolvingHeadlessAMMCLoop`
- `EvolutionTelemetryLogger`
- `ChampionExporter`
- `TrialRunner`
- `PlasticityAblationRunner`
- architecture document
- smoke test and unittest scaffold

Validation:

- `python -m compileall gen5` passed.
- `python -m unittest discover -s gen5\tests -v` passed.
- PyTorch-dependent unit tests skip cleanly when PyTorch is not installed
  locally; pure-Python champion export and telemetry contracts pass.
- Sprint 11 syntax validation passes locally; Colab/PyTorch should run the
  torch-dependent evaluation contracts.

## Open questions

1. Dynamic CUDA allocator strategy
   - fixed-size pool with compaction?
   - slab allocator?
   - per-neuron adjacency pools?
   - global free-list?

2. Backpropagation across topology changes
   - should structural changes happen only between optimizer steps?
   - can we support surrogate gradients for sprouting/pruning decisions?
   - should topology mutation be reinforcement/evolution-driven rather than
     gradient-driven?

3. Delay-buffer implementation
   - ring buffers per edge?
   - grouped delay buckets?
   - event-driven sparse queues?

4. Evaluation protocol
   - what is the minimum trial length for meaningful behavior claims?
   - should food/toxin placement be seeded?
   - should comparison baselines include random, evolved-only, Colab-only, and
     Colab-plus-plasticity brains?

5. Isaac Gym / vectorized embodiment
   - initial target environment?
   - observation/action schema?
   - reward-to-astrocyte mapping?

6. Batched collision scale
   - current prototype computes all agent-object distances with broadcast
     tensors; this is simple and GPU-friendly but may need spatial hashing for
     very large food/toxin counts.
   - evaluate memory cost at 10,000 agents x object count on target GPUs.

7. Per-agent brain integration
   - `EvolvingHeadlessAMMCLoop` now uses `TensorEvolver.forward()` as the
     per-agent recurrent brain.
   - next question: should the shared-brain `HeadlessAMMCLoop` remain as a
     baseline or be moved under examples only?

8. Epoch telemetry semantics
   - Sprint 8 returns the pre-reset world telemetry alongside an epoch report
     when an epoch triggers.
   - decide whether future reporting should also include post-reset environment
     state in the same return payload.

9. Champion visualization fidelity
   - Gen-5 champion export now emits a compatible connectome and matching
     weights, but Gen-4 browser sensors are still a compatibility mapping.
   - Future work may need a native Gen-5 browser inspection mode so food/toxin
     channels and recurrent hidden state are represented without lossy mapping.
   - First browser replay of the champion was motor-silent despite valid
     topology/weight import, strengthening the case for a dedicated transducer
     compatibility sprint.

10. Evolution pressure tuning
   - The first 500-generation run saturated mean active synapses near `86`.
   - Open question: should topology pressure be implemented as a fitness
     penalty, mutation schedule, or hard pruning rule?
   - Open question: is plateauing caused by topology saturation, environment
     difficulty, or insufficient exploitative selection pressure?

11. Browser bridge calibration
   - Gen-5 bridge replay now moves safely but does not collect food yet.
   - Need deterministic browser world seeds to compare changes.
   - Need a controlled gain sweep for Gen-5 sensor gain and motor assist gain.

12. Champion plasticity stability
   - Plasticity-enabled seeded replay collected food and consolidated STW into
     LTW, but later hit a toxin after structural churn.
   - Open question: should imported champions default to reduced plasticity
     rates, reward-gated sprouting, or a protected core-connectome mask?

13. Evaluation runtime cost
   - 10 seeds x 500 generations x 10,000 agents is intentionally
     Colab-accelerator work, not local desktop work.
   - Open question: what epoch length gives enough behavioral signal while
     keeping trial cost manageable?
   - Open question: should trial reports use all-time best fitness, epoch best
     fitness, or both as the primary convergence curve?

14. Plasticity ablation semantics
   - Full/aggressive plasticity beats current gated adult plasticity on
     perturbed all-time best fitness and on recovery retention.
   - Gated adult plasticity remains more compact, so the next question is not
     whether gating matters, but which events should be gated and by what
     dopamine/fitness signal.
   - Need variants that separate sprouting gates, pruning gates, LTW decay
     gates, and protected-core masks.

## Next recommended steps

1. Run the Phase 11 benchmark suite on Colab TPU/XLA:
   - `--device xla` throughput,
   - `--device xla` multi-seed convergence,
   - `--device xla` plasticity and retention ablations,
   - compare against the existing CUDA/T4 evidence.
2. Complete topology-aware hotpath throughput coverage:
   - rerun champion eager and compiled hotpaths with matching
     `adjacency_sha256`,
   - sweep champion `--max-edges 64`, `96`, and `128` after fingerprinting,
   - compare eager vs `--compile` for the `foraging` 8-edge prior,
   - report `tick_mode`, active-edge count, edge-pool capacity, utilization,
     `adjacency_sha256`, memory, and agent-steps/sec at 1k/10k/50k/100k.
3. Redesign gated/adult plasticity:
   - test separate gates for sprouting, pruning, LTW decay, and LTW noise,
   - add protected-core champion masks,
   - tune dopamine/fitness thresholds from retention results.
4. Run fair trained baselines:
   - BPTT-trained static LIF SNN,
   - PPO-trained MLP after installing `stable-baselines3`,
   - report fitness, active parameters, memory, and inference speed.
5. Add active-edge pressure to evolution:
   - fitness penalty per active edge,
   - lower sprout probability,
   - stronger low-LTW pruning,
   - compare fitness-per-active-synapse.
6. Expand `DynamicSparseLinear` with delay buckets so polychronous timing can be
   benchmarked directly in Gen-5.
7. Add astrocyte reward/punishment coupling from `TensorEnvironment2D` into
   `DualTensorManager`.
8. Continue Gen-5 -> Gen-4 bridge calibration:
   - deterministic browser world seed,
   - Gen-5 sensor gain,
   - Gen-5 motor assist gain,
   - compare tensor-environment replay against browser replay for the same
     champion genome.
9. Use `gen5/tools/verify_phase11_outputs.py` after every future output upload
   to avoid ambiguous evidence status.
