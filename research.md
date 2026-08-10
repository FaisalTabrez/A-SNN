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

- `gen5/outputs/legacy_first_run_upload_2026-06-25/evolution_telemetry.json`
  contains `500` generations.
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
- `gen5/outputs/legacy_first_run_upload_2026-06-25/evolution_telemetry.json`
- `gen5/outputs/legacy_first_run_upload_2026-06-25/champion_sparse_adjacency.json`
- `gen5/outputs/legacy_first_run_upload_2026-06-25/champion_connectome.json`
- `gen5/outputs/legacy_first_run_upload_2026-06-25/colab_weights.json`
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

Fingerprint-matched champion eager run added 2026-06-27:

- Corrected fingerprinted champion eager hotpath files were uploaded and
  verified against JSON and CSV.
- Adjacency SHA-256 matched the compiled champion run:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`.
- Active edges: `86`.
- Edge pool capacity: `128`.
- Active edge utilization: `67.19%`.
- At `100k` agents, eager champion throughput reached `5.60M`
  agent-steps/sec with `853.89 MB` CUDA max memory.
- Clean matched-SHA compiled/eager throughput ratios:
  - `4.904x` at `1k`,
  - `5.791x` at `10k`,
  - `6.197x` at `50k`,
  - `6.311x` at `100k`.
- At `100k`, compiled memory was `57.2%` of eager memory.

Interpretation:

- This is the first publication-grade champion eager-vs-compiled comparison.
- The champion-specific CUDA compiler story is now:
  `35.35M` compiled agent-steps/sec versus `5.60M` eager agent-steps/sec at
  `100k` agents on the same adjacency fingerprint.
- This validates the earlier saturated-topology compile finding on a real
  evolved topology, not only a synthetic edge-load preset.

Next action:

- Run foraging 8-edge hotpath with and without `--compile`.
- Run champion capacity sweeps using the same adjacency SHA to quantify
  fixed-pool overhead.

Artifact:

- `gen5/outputs/throughput_cuda_champion_eager_hotpath_fingerprinted_2026-06-27/analysis.md`

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

### 19. Neuron scaling must be measured as capacity efficiency, not just larger topology

Decision: add a dedicated Sprint 12 `NeuronScalingRunner` to test whether
increasing hidden decision nodes improves embodied foraging performance after
Gen-5's base convergence and throughput evidence.

Rationale:

- The current default transducer reserves `8` sensory channels and `4` motor
  channels, so hidden decision capacity is `neuron_count - 12`.
- More neurons are only useful if evolution discovers behaviorally valuable
  intermediate structure. Otherwise, larger brains become a compute and memory
  liability rather than a capability gain.
- Raw best fitness is insufficient. The runner also records mean active
  synapses, active edge-pool utilization, fitness per active synapse, threshold
  success rate, and mean generation-to-threshold.

Default sweep:

- `16` neurons / `128` edge slots: current champion-class baseline.
- `32` neurons / `256` edge slots: first expanded decision layer.
- `64` neurons / `512` edge slots: larger Colab-scale decision layer.

Interpretation plan:

- If fitness rises while fitness-per-active-synapse stays stable or improves,
  larger decision capacity is likely buying real behavioral expressivity.
- If fitness plateaus while active synapses and utilization rise, topology is
  bloating and we need stronger active-edge pressure.
- If larger models converge slower but reach higher final fitness, the project
  should separate "sample efficiency" from "asymptotic capability" in future
  claims.

Artifacts:

- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint12_neuron_scaling.py`
- `gen5/tests/test_evaluation_contract.py`

### 20. Neuron scaling result: current Gen-5 is structure-efficiency limited, not neuron-capacity limited

Finding: the first Sprint 12 CUDA neuron-scaling run completed across `10`
seeds, `500` generations, and `10,000` agents at `16`, `32`, and `64`
neurons.

Results:

| Neurons | Hidden decision nodes | Edge slots | Final mean best fitness | Std | Mean active synapses | Utilization | Fitness / active synapse | Threshold success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 128 | 25.9 | 1.10 | 85.92 | 67.13% | 0.301 | 100% |
| 32 | 20 | 256 | 26.1 | 2.33 | 171.62 | 67.04% | 0.152 | 80% |
| 64 | 52 | 512 | 25.5 | 1.35 | 343.62 | 67.11% | 0.074 | 70% |

Interpretation:

- More neurons did not materially improve final fitness on the current
  foraging task.
- Active synapses scaled almost exactly with edge-pool capacity, while edge
  utilization stayed near `67%` across every condition.
- Fitness per active synapse approximately halved with each capacity doubling:
  `0.301 -> 0.152 -> 0.074`.
- The current task and mutation schedule are not exploiting extra hidden
  decision nodes. The system is filling available edge capacity rather than
  discovering more efficient sparse computation.

Decision:

- Do not increase neuron count further as the next default capability lever.
- Prioritize active-edge pressure, low-LTW pruning pressure, and
  fitness-per-active-synapse as first-class optimization targets.
- Treat `16` neurons / `128` edge slots as the current Pareto baseline for the
  simple 2D foraging task until a sparse-efficiency ablation proves otherwise.

Artifact:

- `gen5/outputs/neuron_scaling_cuda_2026-06-27/analysis.md`

### 21. Sparse-efficiency ablation is the next required gate before harder worlds or MNIST

Decision: implement a dedicated sparse-efficiency ablation before moving to
harder bot worlds or MNIST-style classification.

Rationale:

- The neuron-scaling run showed that larger brains fill edge capacity without
  improving behavior on the current foraging task.
- Moving to MNIST before solving this would risk importing the same topology
  bloat into a harder benchmark, making failures ambiguous.
- The next experiment must prove that AMMC can maintain fitness while reducing
  active synapses and improving fitness-per-active-synapse.

Implemented controls:

- Active-edge fitness penalty: rank organisms by raw fitness minus metabolic
  edge cost.
- Low-LTW pruning pressure: add a separate pruning channel for weak long-term
  connections.
- Sprouting schedule: reduce sprouting probability as edge-pool capacity grows.
- Protected core: preserve seeded/champion-like core pathways while allowing
  peripheral plasticity.
- Hidden-node diagnostics: record how much active topology actually touches
  hidden decision nodes versus direct sensor-to-motor reflex edges.

Success criterion:

- Preserve or improve final best fitness.
- Reduce mean active synapses and edge utilization.
- Improve fitness-per-active-synapse.
- Show whether larger hidden-node pools are actually used, rather than merely
  filled.

Artifacts:

- `gen5/ammc_gen5/evolver.py`
- `gen5/ammc_gen5/evolving_loop.py`
- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint13_sparse_efficiency_ablation.py`

### 22. Sparse-efficiency runs need staged execution and checkpoints

Finding: the first full sparse-efficiency command ran for more than `4` hours
in Colab. This is expected: the full matrix is `5` ablation groups x `3`
neuron scales x `10` seeds x `500` generations x `120` steps, making it about
`5x` heavier than the earlier neuron-scaling sweep.

Decision:

- Keep the full matrix available, but do not use it as the default exploratory
  workflow.
- Add `--groups` filtering so individual mechanisms can be run independently.
- Add checkpointing after each group/scale/seed trial so Colab interruptions do
  not destroy completed work.
- Add a recommended focused screen:
  `baseline_capacity_fill` vs `protected_sparse_core`, seeds `42 43 44`,
  `200` generations, full `16/32/64` scale points.

Operational note:

- The old no-checkpoint full run should be allowed to finish if it is still
  actively using the accelerator and Colab is stable.
- For future runs, prefer checkpointed screens and only expand to the full
  matrix after a promising group is identified.

Artifacts:

- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint13_sparse_efficiency_ablation.py`
- `gen5/docs/PHASE11_COLAB_RUNBOOK.md`

### 23. Sparse-efficiency screen: sparse pressure works, but the first protected core is too aggressive

Finding: the checkpointed sparse-efficiency screen completed on CUDA for
`baseline_capacity_fill` and `protected_sparse_core` across `3` seeds, `200`
generations, and the `16/32/64` neuron scale points.

Results:

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Hidden-edge fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_capacity_fill | 16 | 25.67 | 81.70 | 63.83% | 0.233 | 43.58% | 100% |
| baseline_capacity_fill | 32 | 25.00 | 163.45 | 63.85% | 0.108 | 85.11% | 66.67% |
| baseline_capacity_fill | 64 | 24.00 | 326.66 | 63.80% | 0.057 | 95.89% | 33.33% |
| protected_sparse_core | 16 | 24.33 | 41.81 | 32.66% | 0.462 | 36.11% | 66.67% |
| protected_sparse_core | 32 | 18.67 | 46.43 | 18.14% | 0.344 | 71.31% | 0% |
| protected_sparse_core | 64 | 15.00 | 44.00 | 8.59% | 0.220 | 78.86% | 0% |

Interpretation:

- The baseline repeated the neuron-scaling pattern: larger edge pools filled to
  about `64%` utilization without improving raw fitness.
- The protected sparse core sharply reduced active synapses and nearly doubled
  fitness-per-active-synapse at `16` neurons.
- At `32` and `64` neurons, the same pressure underfit badly. It kept active
  synapses near `42-46` across all scales, which is too restrictive for larger
  networks.
- Sparse pressure is therefore directionally useful, but it needs a gentler or
  capacity-aware schedule.

Decision:

- Treat `protected_sparse_core` as a successful proof of efficiency pressure at
  small scale, not as the final sparse rule.
- Run component ablations next: `active_edge_penalty`, `low_ltw_pruning`, and
  `scheduled_sprouting`.
- Add a future gentler protected-core variant with lower penalty, lower
  weak-edge prune probability, and/or a capacity-aware minimum active-edge
  target.
- Keep `16` neurons as the current simple-foraging Pareto baseline.

Artifact:

- `gen5/outputs/sparse_efficiency_screen_cuda_2026-06-28/analysis.md`

### 24. Sparse-efficiency component screen: low-LTW pruning is the best single lever

Finding: the component sparse-efficiency screen completed on CUDA for
`active_edge_penalty`, `low_ltw_pruning`, and `scheduled_sprouting` across `3`
seeds, `200` generations, and the `16/32/64` neuron scale points.

Results:

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Threshold success |
|---|---:|---:|---:|---:|---:|---:|
| active_edge_penalty | 16 | 24.67 | 61.14 | 47.77% | 0.273 | 33.33% |
| active_edge_penalty | 32 | 23.67 | 133.01 | 51.96% | 0.123 | 0% |
| active_edge_penalty | 64 | 23.67 | 274.02 | 53.52% | 0.073 | 33.33% |
| low_ltw_pruning | 16 | 23.67 | 49.46 | 38.64% | 0.364 | 0% |
| low_ltw_pruning | 32 | 26.00 | 98.00 | 38.28% | 0.191 | 100% |
| low_ltw_pruning | 64 | 24.67 | 194.11 | 37.91% | 0.098 | 66.67% |
| scheduled_sprouting | 16 | 24.00 | 82.21 | 64.23% | 0.243 | 0% |
| scheduled_sprouting | 32 | 24.67 | 111.31 | 43.48% | 0.162 | 33.33% |
| scheduled_sprouting | 64 | 23.33 | 134.13 | 26.20% | 0.142 | 33.33% |

Interpretation:

- `low_ltw_pruning` is the strongest single mechanism so far. At `32` neurons
  it achieved the best raw fitness (`26.0`) while cutting active synapses to
  `98`, versus `163.45` in the prior baseline screen.
- `low_ltw_pruning` also preserved threshold success better than the other
  component groups at `32` and `64` neurons.
- `active_edge_penalty` is too blunt at the current coefficient. It reduces
  wiring, but hurts threshold success and does not preserve raw fitness well.
- `scheduled_sprouting` is useful at larger capacities, especially `64`
  neurons, where it produces the best component-screen fitness per active
  synapse. Its raw fitness is weaker than `low_ltw_pruning`.

Decision:

- Keep low-LTW pruning as the primary sparse-efficiency mechanism.
- Combine low-LTW pruning with scheduled sprouting next.
- Avoid or substantially reduce active-edge fitness penalty until its
  coefficient is calibrated.
- Do not reuse the original aggressive `protected_sparse_core`; design a
  gentler capacity-aware variant.

Artifact:

- `gen5/outputs/sparse_efficiency_components_cuda_2026-06-28/analysis.md`

### 25. Implement gentler sparse-efficiency combination with an active-edge floor

Decision: add a new sparse-efficiency group, `gentle_ltw_scheduled`, combining
the two most promising component mechanisms:

- low-LTW pruning,
- capacity-scaled sprouting.

The group deliberately avoids the blunt active-edge fitness penalty and adds a
capacity-aware active-edge floor so larger networks cannot collapse to the
`40-50` active-edge regime observed in the original aggressive
`protected_sparse_core` screen.

Configuration:

- `low_ltw_prune_threshold`: `0.08`
- `low_ltw_prune_probability`: `0.03`
- `sprout_scale_by_capacity`: enabled
- `minimum_active_edge_fraction`: `0.25`
- `active_edge_fitness_penalty`: `0.0`
- protected core: disabled for this first gentle combination

Rationale:

- The component screen showed `low_ltw_pruning` best preserved raw fitness and
  threshold success.
- The same screen showed `scheduled_sprouting` provided useful edge control at
  larger capacities.
- The earlier `protected_sparse_core` screen proved sparse pressure can improve
  fitness-per-active-synapse, but its combined pressure was too aggressive.

Next test:

Run `low_ltw_pruning`, `scheduled_sprouting`, and `gentle_ltw_scheduled`
together on seeds `42 43 44`, `200` generations, and `16/32/64` neurons.

Artifacts:

- `gen5/ammc_gen5/evolver.py`
- `gen5/ammc_gen5/evaluation.py`
- `gen5/examples/sprint13_sparse_efficiency_ablation.py`

### 26. Gentle sparse combination avoids edge collapse, but low-LTW pruning remains the raw-fitness leader

Finding: the CUDA finalist screen completed for `gentle_ltw_scheduled`,
`low_ltw_pruning`, and `scheduled_sprouting` across `3` seeds, `200`
generations, and the `16/32/64` neuron scale points.

Results:

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Threshold success |
|---|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 24.33 | 59.13 | 46.20% | 0.338 | 66.67% |
| gentle_ltw_scheduled | 32 | 23.67 | 77.36 | 30.22% | 0.233 | 0% |
| gentle_ltw_scheduled | 64 | 24.00 | 131.01 | 25.59% | 0.137 | 33.33% |
| low_ltw_pruning | 16 | 24.33 | 49.54 | 38.71% | 0.404 | 66.67% |
| low_ltw_pruning | 32 | 26.00 | 97.52 | 38.09% | 0.191 | 100% |
| low_ltw_pruning | 64 | 23.00 | 193.42 | 37.78% | 0.100 | 0% |
| scheduled_sprouting | 16 | 24.67 | 81.84 | 63.93% | 0.212 | 66.67% |
| scheduled_sprouting | 32 | 24.67 | 111.81 | 43.68% | 0.164 | 66.67% |
| scheduled_sprouting | 64 | 23.67 | 134.96 | 26.36% | 0.138 | 33.33% |

Interpretation:

- `low_ltw_pruning` remains the best raw-fitness candidate. At `32` neurons it
  reached final mean best fitness `26.00` with `100%` threshold success.
- `gentle_ltw_scheduled` achieved its structural purpose: active synapses grew
  with capacity (`59.13 -> 77.36 -> 131.01`) instead of collapsing into the
  overly restrictive `40-50` active-edge regime seen in the first
  `protected_sparse_core` run.
- The gentle schedule did not preserve enough raw fitness at `32` neurons to
  replace `low_ltw_pruning` as the default rule.
- `scheduled_sprouting` remains useful as a stabilizer, but it is not the
  dominant mechanism on either raw fitness or sparse efficiency.
- Hidden-edge usage rises with neuron count, so larger brains are routing
  through hidden nodes; the current simple foraging world is not yet rewarding
  that extra depth.

Decision:

- Promote `low_ltw_pruning` as the raw-fitness finalist.
- Promote `gentle_ltw_scheduled` as the balanced sparse-efficiency finalist.
- Stop running broad sparse-efficiency matrices for now. The next expensive
  Colab run should compare only those two finalists across `10` seeds and
  `500` generations.

Artifact:

- `gen5/outputs/sparse_efficiency_gentle_combo_cuda_2026-06-28/analysis.md`

### 27. Sparse-efficiency finalists: low-LTW pruning is the default; gentle schedule is the efficiency baseline

Finding: the full CUDA finalist run completed for `low_ltw_pruning` and
`gentle_ltw_scheduled` across `10` seeds, `500` generations, and the
`16/32/64` neuron scale points.

Results:

| Group | Neurons | Final mean best fitness | Std | Active synapses | Utilization | Fitness / active synapse | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 26.00 | 1.41 | 61.08 | 47.72% | 0.306 | 90% |
| gentle_ltw_scheduled | 32 | 25.70 | 1.34 | 80.48 | 31.44% | 0.222 | 80% |
| gentle_ltw_scheduled | 64 | 24.40 | 0.97 | 131.30 | 25.65% | 0.133 | 50% |
| low_ltw_pruning | 16 | 25.40 | 0.70 | 51.64 | 40.34% | 0.350 | 90% |
| low_ltw_pruning | 32 | 26.00 | 1.33 | 103.54 | 40.45% | 0.177 | 100% |
| low_ltw_pruning | 64 | 25.40 | 1.07 | 206.55 | 40.34% | 0.085 | 80% |

Interpretation:

- `low_ltw_pruning` is the best default survival rule. Its `32`-neuron setting
  reached final mean best fitness `26.00` with `100%` threshold success.
- `gentle_ltw_scheduled` is a valid sparse-efficiency baseline. At `32`
  neurons it lost only `0.30` mean best fitness relative to `low_ltw_pruning`
  while using about `22%` fewer active synapses (`80.48` vs `103.54`).
- At `64` neurons, the gentle schedule used about `36%` fewer active synapses
  than low-LTW pruning (`131.30` vs `206.55`) and had much better
  fitness-per-active-synapse, but weaker survival reliability.
- The simple bot world still does not reward larger brains. Hidden-edge
  fraction rises strongly with neuron count, but final best fitness does not
  improve.
- The `16`-neuron gentle schedule remains surprisingly competitive, but
  `32`-neuron low-LTW pruning is more reliable.

Decision:

- Freeze sparse-efficiency tuning on the current simple foraging world.
- Use `low_ltw_pruning` at `32` neurons as the default raw-survival baseline.
- Use `gentle_ltw_scheduled` at `32` neurons as the sparse-efficiency baseline.
- Move next to harder bot worlds before expanding neuron count further.

Artifact:

- `gen5/outputs/sparse_efficiency_finalists_cuda_2026-06-28/analysis.md`

### 28. Repository reorganization for newcomer accessibility

Decision: reorganize the repository around stable newcomer entry points without
breaking the browser sandbox path.

Changes:

- Keep `index.html` at the repository root so `http://127.0.0.1:4173/`
  continues to open the visual sandbox.
- Move concept images from `design/` to `assets/design/`.
- Move the original root-level `1st run/` upload bundle to
  `gen5/outputs/legacy_first_run_upload_2026-06-25/`.
- Add repository-level documentation in `docs/`.
- Add output-bundle guidance in `gen5/outputs/README.md`.

Rationale:

- Newcomers should see code, docs, and the browser entry point at the root
  without raw experiment folders competing for attention.
- Gen-5 outputs should all live under `gen5/outputs/` so evidence is easy to
  audit.
- Design images are assets, not source or experiment outputs.

Artifacts:

- `README.md`
- `docs/README.md`
- `docs/PROJECT_STRUCTURE.md`
- `assets/README.md`
- `gen5/outputs/README.md`
- `gen5/outputs/legacy_first_run_upload_2026-06-25/README.md`

### 29. Add harder bot-world benchmark scaffolding

Decision: shift the next benchmark axis from larger neuron counts to harder
environment dynamics.

Implementation:

- Add named world presets to `TensorEnvironment2D`:
  - `simple`,
  - `wide_arena`,
  - `sparse_cues`,
  - `moving_toxins`,
  - `delayed_reward`,
  - `gauntlet`.
- Add optional moving food/toxin object dynamics.
- Add delayed reward and punishment buffers with static tensor shapes.
- Add `sprint14_harder_worlds.py` to compare the current two sparse-efficiency
  finalists across world presets.
- Keep `simple` world defaults backward-compatible with existing baselines.

Rationale:

- Sparse-efficiency and neuron-scaling results show the simple world does not
  reward more hidden decision nodes.
- Harder worlds can test whether hidden recurrent structure becomes useful
  when the agent needs search, memory, hazard tracking, or delayed-credit
  assignment.
- The current baselines remain:
  - raw-survival: `low_ltw_pruning`, `32` neurons,
  - sparse-efficiency: `gentle_ltw_scheduled`, `32` neurons.

Next recommended command:

```bash
python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds simple moving_toxins delayed_reward gauntlet \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/harder_worlds_cuda
```

Artifacts:

- `gen5/ammc_gen5/tensor_environment.py`
- `gen5/examples/sprint14_harder_worlds.py`
- `gen5/docs/HARDER_WORLDS.md`

### 30. Harder-world results: delayed reward is the real capability wall

Finding: the first Sprint 14 harder-world benchmark completed on CUDA for
`simple`, `moving_toxins`, `delayed_reward`, and `gauntlet` using the frozen
`32`-neuron sparse-efficiency baselines.

Results:

| World | Group | Final mean best fitness | Active synapses | Fitness / active synapse | Threshold success |
|---|---|---:|---:|---:|---:|
| simple | gentle_ltw_scheduled | 24.70 | 80.53 | 0.231 | 50% |
| simple | low_ltw_pruning | 26.40 | 103.25 | 0.173 | 80% |
| moving_toxins | gentle_ltw_scheduled | 25.60 | 79.82 | 0.218 | 90% |
| moving_toxins | low_ltw_pruning | 25.50 | 102.97 | 0.168 | 90% |
| delayed_reward | gentle_ltw_scheduled | 20.80 | 81.89 | 0.144 | 10% |
| delayed_reward | low_ltw_pruning | 21.40 | 105.30 | 0.131 | 0% |
| gauntlet | gentle_ltw_scheduled | 10.60 | 77.82 | 0.053 | 0% |
| gauntlet | low_ltw_pruning | 10.60 | 100.58 | 0.038 | 0% |

Interpretation:

- `moving_toxins` does not yet create a meaningful capability wall. Both groups
  reached `90%` threshold success; the gentle schedule was slightly better in
  raw mean fitness while retaining its sparse-edge advantage.
- `delayed_reward` is the first clear hard setting. Mean best fitness dropped
  to roughly `21`, and threshold success collapsed to `10%` or lower.
- `gauntlet` is too hard as a direct jump. It should be treated as a curriculum
  endpoint rather than the next pass/fail benchmark.
- `gentle_ltw_scheduled` keeps about a `22%` active-edge reduction across all
  worlds. Its efficiency advantage is robust, but efficiency alone does not
  solve delayed credit assignment.

Decision:

- Promote delayed reward to the next main benchmark axis.
- Run a delay-length curriculum or sweep before revisiting neuron scaling:
  `reward_delay_steps = 3`, `6`, and `12`.
- Re-run neuron scaling only on a delayed-reward setting that is hard but not
  collapsed.
- Keep gauntlet as a later curriculum endpoint.

Artifact:

- `gen5/outputs/harder_worlds_cuda_2026-06-29/analysis.md`

### 31. Delay-3 screen: less collapsed than delay-12, but not solved

Finding: the short delayed-reward delay-`3` screen completed on CUDA for
`3` seeds and `200` generations.

Important caveat:

The run used global `--reward-delay-steps 3`, so both requested world labels
(`simple` and `delayed_reward`) resolved to the same effective environment
configuration. Treat the two labels as replicate delay-`3` runs, not as
separate world conditions.

Results:

| World label | Group | Final mean best fitness | Active synapses | Threshold success |
|---|---|---:|---:|---:|
| simple | gentle_ltw_scheduled | 24.00 | 77.37 | 33.33% |
| simple | low_ltw_pruning | 23.67 | 98.09 | 33.33% |
| delayed_reward | gentle_ltw_scheduled | 22.67 | 77.83 | 0% |
| delayed_reward | low_ltw_pruning | 22.33 | 97.97 | 0% |

Combined across the two identical effective configurations:

| Group | Effective seeds | Mean best fitness | Active synapses | Threshold success |
|---|---:|---:|---:|---:|
| gentle_ltw_scheduled | 6 | 23.33 | 77.60 | 16.67% |
| low_ltw_pruning | 6 | 23.00 | 98.03 | 16.67% |

Interpretation:

- Delay `3` is less collapsed than the full delay-`12` result, but still not
  solved in the short screen.
- `gentle_ltw_scheduled` slightly beat `low_ltw_pruning` on raw mean fitness
  while using about `21%` fewer active synapses.
- Differences between the `simple` and `delayed_reward` labels are stochastic
  variance because their effective configs were identical.

Decision:

- Run a clean delay sweep using only `--worlds delayed_reward`.
- Sweep `reward_delay_steps = 1`, `2`, and `3` before running expensive
  `10`-seed, `500`-generation evaluations.
- Only rerun neuron scaling after identifying a delay setting that is hard but
  not collapsed.

Artifact:

- `gen5/outputs/delayed_reward_delay3_screen_cuda_2026-06-29/analysis.md`

### 32. Delay-1 screen: first near-threshold delayed-reward candidate

Finding: the clean delay-`1` delayed-reward screen completed on CUDA using only
`--worlds delayed_reward`.

Results:

| Group | Final mean best fitness | Active synapses | Fitness / active synapse | Threshold success |
|---|---:|---:|---:|---:|
| gentle_ltw_scheduled | 23.00 | 77.55 | 0.232 | 0% |
| low_ltw_pruning | 24.67 | 98.65 | 0.193 | 33.33% |

Interpretation:

- Delay `1` is the best delayed-reward screen so far. It is not solved, but
  `low_ltw_pruning` reached `24.67` mean best fitness and crossed threshold in
  `1 / 3` seeds.
- `low_ltw_pruning` reclaims the raw-survival lead at delay `1`.
- `gentle_ltw_scheduled` remains about `21%` sparser, but its survival score is
  weaker in this setting.
- Compared with delay `3` and delay `12`, delay `1` is the first plausible
  hard-but-not-collapsed candidate.

Decision:

- Run the clean delay-`2` screen before spending a full `10`-seed,
  `500`-generation evaluation.
- If delay `2` underperforms delay `1`, promote delay `1` to the next full
  statistical benchmark.
- After selecting the delay setting, rerun neuron scaling there.

Artifact:

- `gen5/outputs/delayed_reward_delay1_screen_cuda_2026-06-29/analysis.md`

### 33. Delay-2 screen: best hard-but-not-collapsed candidate so far

Finding: the clean delay-`2` delayed-reward screen completed on CUDA using only
`--worlds delayed_reward`.

Results:

| Group | Final mean best fitness | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 24.67 | 78.13 | 0.205 | 66.67% | 101 |
| low_ltw_pruning | 24.67 | 98.45 | 0.169 | 33.33% | 70 |

Comparison:

| Delay | Group | Mean best fitness | Active synapses | Threshold success |
|---:|---|---:|---:|---:|
| 1 | gentle_ltw_scheduled | 23.00 | 77.55 | 0% |
| 1 | low_ltw_pruning | 24.67 | 98.65 | 33.33% |
| 2 | gentle_ltw_scheduled | 24.67 | 78.13 | 66.67% |
| 2 | low_ltw_pruning | 24.67 | 98.45 | 33.33% |
| 3 | gentle_ltw_scheduled | 22.67 | 77.83 | 0% |
| 3 | low_ltw_pruning | 22.33 | 97.97 | 0% |

Interpretation:

- Delay `2` is now the strongest hard-but-not-collapsed delayed-reward
  candidate.
- `gentle_ltw_scheduled` matched `low_ltw_pruning` on raw mean best fitness
  while using about `20.6%` fewer active synapses and achieving higher
  threshold success.
- `low_ltw_pruning` still reaches threshold faster when it succeeds, but it
  succeeds in fewer seeds in this short screen.
- Delay `1` was slightly too favorable to raw pruning, while delay `3` was too
  unstable. Delay `2` is the first setting where sparse/gentle adult plasticity
  looks behaviorally competitive rather than merely efficient.

Decision:

- Promote `reward_delay_steps = 2` to the next full statistical benchmark.
- Run `10` seeds for `500` generations at `32` neurons and `256` edge slots.
- If delay `2` holds, rerun neuron scaling on this delayed-reward setting.
- Keep delay `1` as a fallback and delay `3` as the next curriculum target.

Artifact:

- `gen5/outputs/delayed_reward_delay2_screen_cuda_2026-06-29/analysis.md`

### 34. Delay-2 full run: sparse/gentle plasticity survives the 10-seed test

Finding: the full delay-`2` delayed-reward benchmark completed on CUDA for
`10` seeds and `500` generations, but the Colab session ended before the
downloadable artifacts were recovered. The final console JSON summary survived
and has been archived as summary-only evidence.

Important evidence caveat:

- This is console-log-only evidence.
- The final summary JSON survived.
- Per-generation records, full JSON/CSV outputs, and plots were not recovered.
- Use this result to guide roadmap decisions, but rerun with persistent output
  before using it as publication-grade evidence.

Results:

| Group | Final mean best fitness | Std | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 24.50 | 1.08 | 81.40 | 0.200 | 50% | 169.20 |
| low_ltw_pruning | 24.50 | 1.43 | 104.82 | 0.166 | 40% | 201.25 |

Interpretation:

- The full run preserves the core delay-`2` screen result:
  `gentle_ltw_scheduled` matches `low_ltw_pruning` on raw mean best fitness.
- `gentle_ltw_scheduled` used about `22.3%` fewer active synapses.
- It also had lower variance, higher threshold success, and earlier mean
  threshold crossing.
- `low_ltw_pruning` retained a higher mean selection-best fitness
  (`17.40` vs. `16.30`), so it remains useful as a raw-selection baseline.

Decision:

- Treat `gentle_ltw_scheduled` at `reward_delay_steps = 2` as the preferred
  delayed-reward sparse-efficiency baseline.
- The next major experiment should be delay-`2` neuron scaling, with persistent
  Colab output enabled from the beginning.
- If we need publication-quality evidence for the exact full delay-`2` run,
  rerun it with Drive-backed output or automatic zip export.

Artifact:

- `gen5/outputs/delayed_reward_delay2_full_cuda_console_2026-06-29/analysis.md`

### 35. Delay-2 neuron scaling: compact brains still win

Finding: the delay-`2` delayed-reward neuron-scaling run completed on CUDA for
`10` seeds and `500` generations across `16`, `32`, and `64` neurons. The run
wrote persistent Drive outputs, but only the console log was provided in this
turn, so this is currently summary-only evidence.

Important evidence caveat:

- This is console-log-only evidence until the Drive artifacts are uploaded.
- The final summary JSON survived in the console log.
- The reported Drive output folder is
  `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/`.

Results:

| Group | Neurons | Hidden | Final mean best fitness | Active synapses | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 4 | 25.40 | 62.10 | 0.274 | 44.25% | 14.78% | 80% |
| gentle_ltw_scheduled | 32 | 20 | 23.70 | 81.45 | 0.211 | 84.60% | 5.07% | 0% |
| gentle_ltw_scheduled | 64 | 52 | 23.50 | 131.33 | 0.117 | 94.08% | 3.04% | 20% |
| low_ltw_pruning | 16 | 4 | 25.00 | 52.58 | 0.331 | 43.51% | 15.12% | 70% |
| low_ltw_pruning | 32 | 20 | 24.40 | 104.72 | 0.158 | 85.49% | 4.40% | 40% |
| low_ltw_pruning | 64 | 52 | 24.60 | 207.71 | 0.080 | 96.05% | 1.39% | 60% |

Interpretation:

- Delay `2` did not unlock larger-brain scaling. The strongest points are still
  the compact `16`-neuron models.
- `gentle_ltw_scheduled` at `16` neurons has the best raw mean fitness
  (`25.40`) and threshold success (`80%`).
- `low_ltw_pruning` at `16` neurons has the best sparse efficiency
  (`0.331` fitness per active synapse) and uses the fewest active synapses
  (`52.58`).
- Larger networks route overwhelmingly through hidden nodes while losing direct
  sensor-motor pathways. Hidden-edge fraction rises to `85-96%`, while direct
  sensor-motor fraction collapses to `1-5%`.
- The current delayed-reward task still rewards compact sensor-motor loops more
  than deep latent decision chains. Extra hidden capacity may dilute credit
  assignment rather than improving memory.

Decision:

- Treat `16` neurons as the preferred topology for delay-`2` delayed reward.
- Use `gentle_ltw_scheduled/16` as the raw-fitness compact baseline.
- Use `low_ltw_pruning/16` as the sparse-efficiency compact baseline.
- Stop treating `32` neurons as the default delayed-reward baseline until a
  harder world proves hidden capacity is useful.
- Next experiments should either preserve direct sensor-motor cores while
  adding hidden nodes, or move to a harder task that genuinely requires hidden
  state.

Artifact:

- `gen5/outputs/neuron_scaling_delay2_cuda_console_2026-06-29/analysis.md`

### 36. Delay-2 neuron scaling artifacts recovered

Finding: the full Drive artifacts for the delay-`2` delayed-reward
neuron-scaling run were recovered and archived locally. This upgrades the
previous console-only evidence into full artifact evidence.

Archived files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_summary.png`
- `sparse_efficiency_progress.json`

Validation:

- `sparse_efficiency_progress.json` reports `60 / 60` completed trials.
- `sparse_efficiency_summary.csv` matches the previous console summary.
- The result remains unchanged: compact `16`-neuron brains outperform larger
  `32`- and `64`-neuron brains under delay-`2`.

Decision:

- Promote the `16`-neuron / `128`-edge topology as the current delayed-reward
  baseline.
- Use `gentle_ltw_scheduled/16` as the raw-fitness compact baseline.
- Use `low_ltw_pruning/16` as the sparse-efficiency compact baseline.
- Do not spend more delay-`2` budget on larger hidden-node pools unless a new
  mechanism preserves the useful direct core.

Artifact:

- `gen5/outputs/neuron_scaling_delay2_cuda_2026-06-29/analysis.md`

### 37. Sprint 15 direction: freeze AMMC and diversify tasks

Decision: before adding more plasticity or hidden capacity, evaluate the frozen
AMMC sparse substrate on diversified non-foraging tasks.

Rationale:

- Delay-`2` neuron scaling showed that larger hidden-node pools do not improve
  the current foraging benchmark.
- Compact `16`-neuron models win because the task still rewards direct
  sensor-motor loops.
- A realistic view requires separating three effects:
  1. what the frozen sparse wiring prior can already do,
  2. what simple reflex baselines can do,
  3. what genuinely requires learning, protected hidden expansion, or a new
     task transducer.

Implementation:

- Add `FrozenTaskRunner` and `FrozenTaskConfig`.
- Add `sprint15_frozen_diversified_tasks.py`.
- Add download-free synthetic tasks:
  - `direction_copy`,
  - `anti_toxin`,
  - `cue_switch`,
  - `delayed_recall`,
  - `two_pulse_sum`.
- Report frozen AMMC accuracy against random, instant-reflex, integrated-reflex,
  and oracle baselines.

Important limitation:

- Current archived evolution outputs contain population-level statistics, not a
  single champion genome snapshot for the winning `16`-neuron organism.
- Therefore Sprint 15 freezes the current sparse AMMC prior/seeded substrate,
  not a fully trained champion. Champion-level frozen evaluation should be
  added once future runs export best genomes automatically.

Decision rule:

- If frozen AMMC only wins direct/reflexive tasks, the next major work is task
  transduction or learning, not scaling.
- If frozen AMMC beats reflex baselines on temporal tasks, the current sparse
  recurrent substrate already has useful memory.
- If larger hidden models later beat `16` neurons on these tasks, hidden scaling
  has finally earned its keep.

Artifacts:

- `gen5/ammc_gen5/frozen_tasks.py`
- `gen5/examples/sprint15_frozen_diversified_tasks.py`
- `gen5/docs/FROZEN_DIVERSIFIED_TASKS.md`

### 38. Frozen diversified tasks: current substrate is reflexive, not contextual

Finding: the first Sprint 15 frozen diversified task run completed on CUDA for
`4096` samples and `8` timesteps using the `16`-neuron / `128`-edge frozen
sparse AMMC prior.

Results:

| Task | Frozen AMMC | Random | Instant reflex | Integrated reflex | Inactive output | Margin |
|---|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 25.29% | 100.00% | 100.00% | 0.00% | 2.000 |
| anti_toxin | 25.00% | 25.27% | 25.00% | 25.00% | 100.00% | 0.000 |
| cue_switch | 50.42% | 24.49% | 50.42% | 50.42% | 0.00% | 0.196 |
| delayed_recall | 100.00% | 26.05% | 25.00% | 100.00% | 0.00% | 0.319 |
| two_pulse_sum | 25.00% | 24.56% | 25.00% | 6.05% | 0.00% | -0.339 |

Interpretation:

- `direction_copy` is solved, but reflex baselines solve it too. This confirms
  the direct food-direction prior works.
- `delayed_recall` is solved by frozen AMMC and integrated reflex, but not by
  instant reflex. This is evidence accumulation across time, not yet proof of
  hidden-state memory.
- `anti_toxin` stays at chance with `100%` inactive output. The frozen toxin
  prior suppresses motor output instead of actively routing to the opposite
  direction.
- `cue_switch` sits at about `50%`, matching reflex baselines. The frozen model
  ignores the context cue.
- `two_pulse_sum` is chance-level. The frozen circuit does not perform modular
  temporal composition.

Decision:

- The current frozen AMMC substrate should be described as a reflex/evidence
  integration system, not yet a context-routing or sequence-composition system.
- The next Sprint 15 improvement should be a frozen representation probe:
  collect final membrane/spike traces, train only a small linear readout, and
  keep the recurrent sparse substrate frozen.
- If a linear probe solves `cue_switch` or `two_pulse_sum`, the information is
  present but the motor readout is wrong.
- If the linear probe fails, the substrate itself lacks the needed
  representation and will require training, plasticity, or architectural
  changes.

Artifact:

- `gen5/outputs/frozen_diversified_tasks_cuda_2026-06-29/analysis.md`

### 39. Frozen representation probe implemented

Decision: add a Sprint 15 frozen representation probe to separate readout
failure from representation failure.

Implementation:

- Add `FrozenRepresentationProbeRunner`.
- Add `FrozenProbeConfig`, `FrozenProbeResult`, and
  `FrozenProbeSummaryRecord`.
- Add `sprint15_frozen_representation_probe.py`.
- Feature vector: final membrane state plus per-neuron spike counts from the
  frozen recurrent sparse AMMC substrate.
- Trainable component: only a small linear classifier over frozen features.
- Frozen components: sparse recurrent edge topology, STW/LTW weights,
  transducer dynamics, and generated task inputs.

Metrics:

- frozen motor-readout accuracy,
- linear-probe test accuracy,
- linear-probe train accuracy,
- random accuracy,
- instant-reflex accuracy,
- integrated-reflex accuracy,
- representation gain over frozen readout,
- representation gain over best reflex baseline.

Decision rule:

- If probe accuracy beats frozen motor readout on `cue_switch`, `anti_toxin`, or
  `two_pulse_sum`, prioritize readout/transducer learning.
- If probe accuracy remains at chance, prioritize substrate learning,
  plasticity, or architecture changes.

Artifacts:

- `gen5/ammc_gen5/frozen_tasks.py`
- `gen5/examples/sprint15_frozen_representation_probe.py`
- `gen5/docs/FROZEN_DIVERSIFIED_TASKS.md`

### 40. Frozen representation probe: readout is the main near-term bottleneck

Finding: the first Sprint 15 frozen representation probe completed on CUDA.
The sparse AMMC recurrent substrate remained frozen; only a linear classifier
over final membrane plus spike-count features was trained.

Results:

| Task | Frozen motor | Linear probe | Probe train | Best reflex | Gain over frozen | Gain over best reflex |
|---|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| anti_toxin | 25.00% | 100.00% | 100.00% | 24.98% | 75.00% | 75.02% |
| cue_switch | 50.42% | 85.76% | 88.32% | 53.13% | 35.35% | 32.63% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| two_pulse_sum | 25.00% | 31.81% | 33.00% | 26.69% | 6.81% | 5.13% |

Interpretation:

- `anti_toxin` is a readout/transducer failure, not a representation failure.
  The frozen substrate contains enough information for a linear readout to
  reach `100%`, while the fixed motor readout is inactive.
- `cue_switch` is partially represented. The probe reaches `85.76%`, which is
  far above reflex baselines, but not perfect.
- `two_pulse_sum` remains near chance even with the probe, so modular temporal
  composition is not linearly recoverable from the current frozen substrate.
- `direction_copy` and `delayed_recall` remain solved, but they are still
  explainable as direct reflex / evidence integration.

Decision:

- Do not jump directly to full recurrent/plastic training.
- First implement a minimal trainable readout/transducer adapter while keeping
  the sparse recurrent AMMC substrate frozen.
- Use the linear probe as a ceiling for what readout adaptation should recover.
- After readout adaptation, revisit `two_pulse_sum`; if it remains near chance,
  that is the first clear target for substrate learning or architecture change.

Artifact:

- `gen5/outputs/frozen_representation_probe_cuda_2026-06-29/analysis.md`

### 41. Frozen readout/transducer adapter implementation

Decision: implement the trainable readout/transducer adapter as the deployable
bridge between diagnostic probing and full recurrent/plastic learning.

Implementation:

- Add `FrozenReadoutAdapterConfig`, `FrozenReadoutAdapterResult`,
  `FrozenReadoutAdapterSummaryRecord`, and `FrozenReadoutAdapterRunner`.
- Add `sprint15_frozen_readout_adapter.py`.
- Keep the sparse AMMC recurrent substrate frozen.
- Train only a small adapter head over frozen trace features.
- Support `linear` and `mlp` adapter heads.
- Support two diagnostic feature modes:
  - `full_trace`: final membrane plus spike counts for all neurons.
  - `motor_trace`: final membrane plus spike counts for motor neurons only.

Run command:

```bash
python gen5/examples/sprint15_frozen_readout_adapter.py \
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

Decision rule:

- If `linear/full_trace` reproduces the frozen representation probe, the
  readout adapter is enough for tasks like `anti_toxin`.
- If `motor_trace` fails while `full_trace` succeeds, useful information is
  distributed in hidden/non-motor state and the fixed motor pathway is the
  bottleneck.
- If `mlp/full_trace` beats `linear/full_trace`, the substrate representation
  exists but is not linearly separated.
- If all modes remain near chance on `two_pulse_sum`, the next change must be
  recurrent substrate learning or a temporal-composition mechanism, not a
  readout-only fix.

Artifacts:

- `gen5/ammc_gen5/frozen_tasks.py`
- `gen5/examples/sprint15_frozen_readout_adapter.py`
- `gen5/docs/FROZEN_DIVERSIFIED_TASKS.md`

### 42. First frozen readout-adapter result

Finding: the first Sprint 15 frozen readout/transducer adapter run completed on
CUDA. The recurrent sparse AMMC substrate remained frozen; only a linear
adapter over `full_trace` features was trained.

Configuration:

- Adapter kind: `linear`
- Feature mode: `full_trace`
- Samples: `4096`
- Train/test split: `2867 / 1229`
- Timesteps: `8`
- Neurons: `16`
- Max edges: `128`
- Feature dimension: `32`

Results:

| Task | Frozen motor | Adapter | Adapter train | Best reflex | Gain over frozen | Gain over best reflex |
|---|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| anti_toxin | 25.00% | 100.00% | 100.00% | 25.31% | 75.00% | 74.69% |
| cue_switch | 50.42% | 73.39% | 75.65% | 51.18% | 22.98% | 22.21% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| two_pulse_sum | 25.00% | 41.50% | 46.46% | 24.90% | 16.50% | 16.60% |

Interpretation:

- `anti_toxin` is confirmed as a readout/transducer failure. The frozen AMMC
  trace contains the correct information, but the fixed motor readout was
  unable to express it.
- `cue_switch` improves materially above frozen and reflex baselines, but it is
  lower than the earlier representation probe. This needs same-seed/same-split
  variant comparison before over-interpreting the gap.
- `two_pulse_sum` is still unsolved, but the adapter lifts it from chance to
  `41.50%`. This means the previous "no sequence signal" interpretation was
  too strong; the substrate contains partial sequence information.

Decision:

- Add a frozen readout-adapter sweep runner so `linear/full_trace`,
  `linear/motor_trace`, and `mlp/full_trace` can be compared in one Colab run.
- Use the sweep to decide whether the next bottleneck is fixed motor routing,
  nonlinear readout separation, or missing recurrent temporal composition.

Artifacts:

- `gen5/outputs/frozen_readout_adapter_cuda_2026-06-29/analysis.md`
- `gen5/examples/sprint15_frozen_readout_adapter_sweep.py`

### 43. Frozen readout-adapter sweep: nonlinear reservoir behavior

Finding: the first frozen readout-adapter sweep completed on CUDA. It compared
`linear/full_trace`, `linear/motor_trace`, and `mlp/full_trace` on the same
tasks, seed, and split.

Results:

| Task | Linear full trace | Linear motor trace | MLP full trace | Frozen motor |
|---|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 25.00% |
| cue_switch | 73.39% | 73.39% | 100.00% | 50.42% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% |
| two_pulse_sum | 41.50% | 34.09% | 100.00% | 25.00% |

Interpretation:

- `anti_toxin` being solved by `linear/motor_trace` means the avoidant signal
  already reaches motor-neuron state. The failure is not missing representation;
  it is the fixed hardcoded motor decision rule.
- `cue_switch` and `two_pulse_sum` are nonlinearly decodable from frozen AMMC
  traces. The previous chance-level frozen readout was not sufficient evidence
  that the substrate lacked temporal/compositional information.
- The current frozen AMMC substrate is best described as a nonlinear reservoir:
  useful temporal signals are preserved and mixed, but a trivial motor argmax
  discards most of the structure.

Decision:

- Add a held-out-seed generalization runner.
- Train each adapter on `train_seed=42`, then evaluate the same trained adapter
  on fresh seeds such as `43` through `47` without retraining.
- If `mlp/full_trace` remains high on held-out seeds, we can claim a reusable
  reservoir representation for the synthetic task family.
- If it collapses, the adapter is overfitting the current synthetic
  distribution and we must diversify task generation before making stronger
  claims.

Artifacts:

- `gen5/outputs/frozen_readout_adapter_sweep_cuda_2026-06-29/analysis.md`
- `gen5/examples/sprint15_frozen_readout_adapter_generalization.py`

### 44. Frozen readout-adapter generalization: held-out seeds pass

Finding: the first held-out-seed generalization run completed on CUDA. It
trained `mlp/full_trace` adapters on `train_seed=42` and evaluated the same
trained adapters without retraining on seeds `43` through `47`.

Results:

| Task | Train-seed split | Held-out seeds mean | Frozen held-out baseline |
|---|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 25.00% |
| cue_switch | 100.00% | 100.00% | ~50.02% |
| delayed_recall | 100.00% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 100.00% | 25.00% |

Interpretation:

- The nonlinear adapter is not merely memorizing one train/test split. It
  generalizes perfectly across fresh synthetic seeds for the current task
  family.
- This supports the nonlinear-reservoir interpretation: the frozen AMMC trace
  carries stable reusable information that can be decoded by a small MLP.
- The main near-term threat to the claim is now distribution shift, not
  seed-to-seed variation.

Decision:

- Add a robustness runner that trains on the base distribution, then evaluates
  without retraining under one-axis perturbations:
  - input-amplitude shifts,
  - sensory noise,
  - sequence-length/timestep shifts.
- If robustness remains high, move back toward embodied harder-world tasks with
  an adapter-equipped controller.
- If robustness collapses, prioritize data/task augmentation before claiming a
  robust reusable reservoir.

Artifacts:

- `gen5/outputs/frozen_readout_adapter_generalization_cuda_2026-06-29/analysis.md`
- `gen5/examples/sprint15_frozen_readout_adapter_robustness.py`

### 45. Frozen readout-adapter robustness: noise and calibration fail

Finding: the first robustness run completed on CUDA. It trained `mlp/full_trace`
on the base synthetic distribution and evaluated without retraining under
amplitude, sensory-noise, and timestep perturbations.

Aggregate adapter accuracy:

| Task | Base | Amp 0.35 | Amp 0.55 | Amp 1.0 | Noise 0.05 | Noise 0.15 | T=4 | T=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 25.00% | 100.00% | 18.84% | 23.64% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 100.00% | 27.99% | 25.55% | 100.00% | 100.00% |
| cue_switch | 100.00% | 100.00% | 25.00% | 87.50% | 20.85% | 24.54% | 100.00% | 100.00% |
| delayed_recall | 100.00% | 0.00% | 100.00% | 100.00% | 23.15% | 23.60% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 24.91% | 100.00% | 100.00% | 24.19% | 24.72% | 81.45% | 100.00% |

Interpretation:

- Clean timestep shifts are mostly robust.
- Additive sensory noise is the dominant failure mode. Even `noise_std=0.05`
  collapses nearly all tasks toward chance.
- Several clean amplitude shifts also collapse, which means the current adapter
  is not calibrated over reservoir-state scale.
- The reusable-reservoir claim remains promising, but only on the clean
  synthetic manifold. Robust deployment needs augmented adapter training or
  explicit normalization.

Decision:

- Extend the robustness runner with optional augmented training conditions:
  `--train-amplitudes`, `--train-noise-stds`, and
  `--train-timestep-values`.
- Next run should train on amplitude/noise/timestep augmentation and evaluate
  against the same robustness suite.
- If augmented readout training fixes the collapse, move toward adapter-equipped
  embodied/harder-world experiments. If not, add explicit feature normalization,
  denoising, or reservoir-level noise robustness.

Artifacts:

- `gen5/outputs/frozen_readout_adapter_robustness_cuda_2026-06-29/analysis.md`
- `gen5/examples/sprint15_frozen_readout_adapter_robustness.py`

### 46. Augmented readout training repairs robustness

Finding: the augmented robustness run completed on CUDA. The adapter was trained
across amplitude, noise, and timestep variants, then evaluated on the same
robustness suite used in Section 45.

Aggregate adapter accuracy:

| Task | Base | Amp 0.35 | Amp 0.55 | Amp 1.0 | Noise 0.05 | Noise 0.15 | T=4 | T=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.99% | 100.00% | 100.00% |
| cue_switch | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.98% | 100.00% | 100.00% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.84% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.41% | 100.00% | 100.00% |

Interpretation:

- The previous noise/amplitude failure was primarily a readout data
  coverage/calibration problem.
- Once trained on augmented reservoir traces, the small MLP adapter decodes the
  frozen AMMC substrate robustly across the tested perturbation grid.
- This strengthens the current working model: AMMC Gen-5 can be framed as a
  sparse evolved reservoir plus a lightweight trainable readout adapter, rather
  than requiring immediate full recurrent retraining for these synthetic tasks.

Decision:

- Move the next major validation back to embodied/harder worlds.
- Add or adapt an evaluation path where the frozen AMMC substrate feeds an
  adapter-equipped motor policy.
- Compare:
  - fixed frozen motor argmax,
  - frozen substrate plus unaugmented adapter,
  - frozen substrate plus augmented adapter.
- Only after embodied transfer should we move to external datasets such as
  MNIST, because the current project thesis is embodied continuous learning
  before static classification.

Artifacts:

- `gen5/outputs/frozen_readout_adapter_augmented_robustness_cuda_2026-07-28/analysis.md`
- `gen5/examples/sprint15_frozen_readout_adapter_robustness.py`

### 47. Decision: test frozen readout transfer in embodied harder worlds

Date: 2026-08-09

Finding: augmented MLP readouts decode the current synthetic frozen-AMMC task
family almost perfectly under the tested amplitude, timestep, noise, and seed
shifts. That evidence still stops short of physical closed-loop control: the
adapter has not yet altered an agent's trajectory, food acquisition, or toxin
exposure.

Decision:

- Implement Sprint 16 as a controlled three-arm embodied ablation:
  - existing fixed motor-spike decoder,
  - clean/nominal MLP readout,
  - amplitude/noise-augmented MLP readout.
- Freeze recurrent AMMC topology and every LTW/STW value in all arms.
- Train the two adapters with an explicit sensor-space oracle that combines
  food attraction with toxin repulsion.
- Evaluate identical held-out world seeds across `simple`, `moving_toxins`, and
  `gauntlet`, with sensor noise `0.0`, `0.05`, and `0.15`.
- Report mean fitness, food hits, toxin hits, non-negative-fitness rate,
  cue-conditioned action coverage, oracle agreement, and action magnitude.

Scientific boundary:

- This is a representation-to-action transfer test, not autonomous policy
  discovery. Any improvement demonstrates that the frozen AMMC state supports a
  more useful motor transducer; it does not demonstrate that evolution or local
  plasticity learned the oracle policy.
- The same finite trace reset schedule is applied to every arm to control
  spike-count scale. A later continuous-state follow-up is required if episodic
  windows succeed.

Implementation:

- `gen5/examples/sprint16_frozen_embodied_adapter.py`
- `gen5/docs/FROZEN_EMBODIED_ADAPTER.md`
- `gen5/tests/test_sprint16_embodied_adapter_contract.py`

### 48. Frozen embodied readout succeeds, but action coverage is confounded

Date: 2026-08-09

Finding: the complete Sprint 16 CUDA run evaluated three policies across three
worlds, three sensor-noise levels, and five held-out seeds (135 runs total).

Overall results:

| Policy | Mean fitness | Positive runs | Survival | Cue-action coverage | Oracle agreement |
|---|---:|---:|---:|---:|---:|
| Augmented adapter | 1.857 | 45/45 | 66.52% | 100.00% | 64.78% |
| Base adapter | 1.764 | 45/45 | 64.80% | 100.00% | 63.55% |
| Fixed motor decoder | -2.807 | 14/45 | 45.57% | 5.18% | 39.22% |

Interpretation:

- Trained readouts reliably convert the frozen trace into positive-fitness
  embodied behavior, including all `noise_std=0.15` evaluations.
- The strongest gains occur in `moving_toxins`: augmented-minus-fixed mean
  fitness is `+11.50`, `+12.92`, and `+9.57` across the three noise levels.
- Oracle agreement improves substantially, so the result is not explained only
  by movement frequency.
- Nevertheless, action coverage is a severe confound. The adapters issue a
  full-magnitude action on every cue-bearing step, whereas the fixed spiking
  decoder is active on only about one step in twenty.
- Augmentation does not consistently beat clean adapter training. Its overall
  paired fitness advantage is `+0.094`, with 28/45 wins. The strongest evidence
  for augmentation is in the gauntlet, not across all worlds.

Decision:

- Do not yet attribute the full fitness gain to AMMC reservoir computation.
- Implement Sprint 17 activity-matched controls:
  - fixed spiking decoder,
  - normalized analog fixed-AMMC decoder,
  - full-activity random cardinal policy,
  - direct sensor oracle,
  - base and augmented adapters.
- Keep identical seeds, action magnitudes, worlds, and sensor-noise conditions.
- Use the result to decide whether the next major task should be continuous
  embodied state or MNIST/event-stream classification.

Artifacts:

- `gen5/outputs/frozen_embodied_adapter_cuda_2026-08-09/`
- `gen5/outputs/frozen_embodied_adapter_cuda_2026-08-09/analysis.md`

### 49. Activity-matched controls validate the frozen AMMC representation

Date: 2026-08-09

Finding: Phase 17 evaluated six controllers across three worlds, three sensor
noise levels, and five held-out seeds (270 evaluations). Both trainable
frozen-trace adapters beat the full-activity random controller and the
normalized fixed analog AMMC decoder in every paired condition (`45/45`).

Overall mean fitness:

| Policy | Mean fitness | Positive runs | Cue-action coverage | Oracle agreement |
|---|---:|---:|---:|---:|
| Augmented adapter | 1.860 | 45/45 | 100.0% | 64.8% |
| Base adapter | 1.721 | 45/45 | 100.0% | 63.5% |
| Direct sensor oracle | 3.083 | 45/45 | 100.0% | 100.0% |
| Fixed analog cardinal | -0.139 | 19/45 | 73.3% | 26.0% |
| Fixed motor spiking | -2.799 | 14/45 | 5.1% | 39.3% |
| Random cardinal | -0.211 | 27/45 | 100.0% | 25.0% |

Interpretation:

- Full-time movement is not sufficient to explain the adapter advantage.
- Fixed analog calibration is also insufficient; useful action information is
  distributed in the frozen AMMC trace and is recoverable by a small readout.
- The direct sensor oracle remains a substantial ceiling, particularly in the
  simple world.
- Augmentation is a mild, inconsistent improvement over clean training
  (`+0.139` mean fitness; `29/45` paired wins), not a settled advantage.

Decision:

- Close the first bot-world validation cycle.
- Implement Phase 18 as a frozen event-coded MNIST benchmark.
- Compare frozen-AMMC linear and MLP readouts against raw-pixel linear and MLP
  baselines on the official MNIST test split.
- Keep the sparse recurrent topology frozen so the phase measures
  representation usefulness rather than end-to-end training or plasticity.

Scientific boundary:

- Phase 17 demonstrates closed-loop representation-to-action decoding. It does
  not demonstrate autonomous policy learning or broad task transfer.
- Phase 18 is a classification benchmark, not evidence for continuous learning.

Artifacts:

- `gen5/outputs/embodied_action_controls_cuda_2026-08-09/`
- `gen5/outputs/embodied_action_controls_cuda_2026-08-09/analysis.md`

### 50. Phase 18 freezes the sparse substrate for event-coded MNIST

Date: 2026-08-09

Decision: implement the first external-data benchmark as a controlled frozen
representation study rather than immediately enabling recurrent training or
plasticity.

Design:

- resize MNIST to `8x8` and encode each pixel as a one-spike latency event;
- use 64 sensor plus 64 hidden neurons;
- freeze a 384-edge sparse AMMC reservoir in a 512-slot pool;
- extract normalized spike counts plus final membrane state;
- train linear and MLP readouts only;
- compare against a linear head and a parameter-budget-matched MLP trained
  directly on identical raw downsampled pixels;
- evaluate three topology/readout seeds on the untouched official test split.

Rationale:

- Raw baselines distinguish representation value from ordinary readout
  capacity.
- Freezing the reservoir keeps this phase interpretable after the Phase 17
  frozen-trace result.
- Multi-seed reporting prevents a favorable random topology from being treated
  as a general result.

Decision rule: the primary representation test is
`frozen_ammc_linear > raw_pixel_linear`. The stronger architectural test is
`frozen_ammc_mlp > raw_pixel_mlp`. Beating chance alone is not meaningful on
MNIST.

Artifacts:

- `gen5/ammc_gen5/event_mnist.py`
- `gen5/examples/sprint18_event_mnist.py`
- `gen5/docs/EVENT_MNIST.md`
- `gen5/tests/test_event_mnist_contract.py`

### 51. Phase 18 shows a representation deficit on MNIST

Date: 2026-08-09

Finding: the frozen event-coded AMMC reservoir underperformed both raw-input
controls across all three seeds.

| Model | Test accuracy | Parameters |
|---|---:|---:|
| Raw pixel linear | 85.94% | 650 |
| Frozen AMMC linear | 79.31% | 2,570 |
| Raw pixel MLP | 95.14% | 34,210 |
| Frozen AMMC MLP | 86.11% | 34,186 |

Paired deficits were `-6.63` percentage points for the linear comparison and
`-9.03` points for the parameter-matched MLP comparison. Hidden neurons were
not silent, but their mean spike rate was only `2.37%`.

Interpretation:

- The current event code plus random frozen reservoir degrades MNIST
  information rather than improving it.
- Scaling neuron count now would add compute without identifying the loss.
- Readout-only throughput is not end-to-end AMMC throughput because reservoir
  feature extraction is excluded.
- Three seeds establish a consistent engineering signal but remain too few for
  a publication-grade statistical claim.

Decision:

- Do not enable plasticity or scale topology yet.
- Implement Phase 19 representation decomposition:
  - raw intensities;
  - flattened latency events;
  - sensor trace;
  - hidden trace;
  - full sensor-plus-hidden trace;
  - raw-plus-hidden residual features.
- Train linear and parameter-budget-matched MLP heads on each representation.
- Record end-to-end feature throughput and hidden spike rate.
- Use the decomposition to choose the Phase 20 coding/dynamics intervention.

Artifacts:

- `gen5/outputs/event_mnist_cuda_2026-08-09/`
- `gen5/outputs/event_mnist_cuda_2026-08-09/analysis.md`

### 52. Phase 19 isolates event-coding and recurrent information loss

Date: 2026-08-09

Decision: preserve the Phase 18 dataset, topology, and frozen-weight contract,
then decompose the representation instead of tuning multiple mechanisms at
once.

The runner compares raw intensity, flattened latency events, sensor trace,
hidden trace, full trace, and raw-plus-hidden features. Every feature receives
both a linear head and an approximately 34k-parameter MLP head.

Rationale:

- A latency deficit isolates temporal quantization as the first bottleneck.
- A sensor-summary deficit isolates final-state pooling.
- A hidden/full deficit isolates recurrent dynamics.
- A raw-plus-hidden gain demonstrates complementary reservoir information and
  motivates a residual sensor pathway.
- If raw-plus-hidden also loses, the frozen random reservoir has not earned
  further scale; Phase 20 must train/evolve or fundamentally recode it.

Scientific boundary: this is a causal decomposition by controlled feature
substitution, not a final MNIST performance run. Phase 19 reuses the same
5,000-image official-test subset whose Phase 18 result motivated the design;
it is therefore an engineering validation subset now. Reserve the unused
5,000-image complement for confirmation after the intervention is fixed.

Artifacts:

- `gen5/examples/sprint19_event_representation_decomposition.py`
- `gen5/docs/EVENT_REPRESENTATION_DECOMPOSITION.md`

### 53. Phase 19 identifies temporal pooling as the MNIST bottleneck

Date: 2026-08-09

Finding: explicit latency events are more linearly separable than raw pixels,
but final trace pooling destroys much of that advantage.

| Representation | Linear | MLP |
|---|---:|---:|
| Raw intensity | 85.94% | 95.14% |
| Flattened latency | 88.11% | 91.40% |
| Sensor summary | 72.74% | 85.97% |
| Hidden summary | 73.03% | 83.43% |
| Full summary | 79.33% | 85.97% |
| Raw plus hidden | 86.92% | 93.50% |

Interpretation:

- Latency encoding adds `+2.17` points of linear separability over raw input.
- Collapsing latency into the sensor summary loses `15.37` linear points.
- Hidden summary recurrence adds no stable standalone benefit.
- Raw-plus-hidden yields a small, consistent `+0.98` linear gain, showing that
  the hidden state contains complementary information even though it is weak.
- The parameter-matched MLP result is partly affected by feature dimension:
  higher-dimensional inputs require narrower hidden layers at a fixed budget.

Decision:

- Keep topology and weights frozen for one more controlled phase.
- Implement Phase 20 per-timestep pre-reset state traces.
- Compare sensor, hidden, full, and raw-residual temporal representations
  against raw, latency, and Phase 19 summary baselines.
- If temporal hidden state remains unhelpful, stop representation engineering
  and move to trained/evolved substrate dynamics in Phase 21.

Artifacts:

- `gen5/outputs/event_mnist_decomposition_cuda_2026-08-09/`
- `gen5/outputs/event_mnist_decomposition_cuda_2026-08-09/analysis.md`

### 54. Phase 20 preserves pre-reset temporal state

Date: 2026-08-09

Decision: test the pooling hypothesis directly without changing event coding,
topology, LTWs, neuron count, or plasticity.

Implementation records the pre-reset membrane state for every sensor and
hidden neuron at all eight timesteps. This representation retains
subthreshold voltage and threshold crossings while avoiding separate doubled
spike/membrane tensors.

Comparisons:

- raw intensity and flattened latency controls;
- Phase 19 full summary;
- sensor, hidden, and full temporal state;
- raw plus hidden temporal residual;
- linear and approximately 34k-parameter MLP heads.

Decision boundary:

- If temporal state recovers latency accuracy, make time-preserving readout a
  core AMMC interface.
- If hidden temporal state adds to sensor/raw features, tune or train sparse
  recurrence in Phase 21.
- If it remains unhelpful, stop frozen feature engineering and train/evolve
  the substrate rather than scaling random recurrence.

Artifacts:

- `gen5/ammc_gen5/temporal_mnist.py`
- `gen5/examples/sprint20_temporal_state_mnist.py`
- `gen5/docs/TEMPORAL_STATE_MNIST.md`

### 55. Phase 20 validates temporal AMMC state but not broad project claims

Date: 2026-08-09

Finding: preserving all pre-reset temporal states raises frozen AMMC linear
accuracy to `91.52%`, compared with `79.40%` for the final summary, `88.11%`
for flattened latency, and `85.94%` for raw pixels. The temporal gain is stable
across all three seeds.

The parameter-matched raw MLP remains the strongest model at `95.14%`; full
temporal AMMC reaches `92.43%`. Hidden temporal state alone does not reliably
beat sensor temporal state, though their combination clearly helps a linear
readout.

Sanity check:

- Supported: sparse temporal expansion can improve linear separability.
- Not yet tested: MNIST LTW learning, structural plasticity, continuous
  learning, retention, and astrocyte modulation.
- Not demonstrated: parameter/memory efficiency or superiority to dense ML.
- Unsupported: Transformer replacement or best-SNN claims.

Decision:

- Adopt time-preserving traces as the default sequential AMMC interface.
- Implement Phase 21 fixed-topology LTW training with surrogate spike
  gradients.
- Compare raw, frozen-temporal, and LTW-trained temporal linear/MLP groups.
- Record end-to-end cost, LTW displacement, event rate, and fixed edge count.
- Allow structural plasticity in Phase 22 only if LTW training improves the
  frozen substrate without activity collapse.

Artifacts:

- `gen5/outputs/event_mnist_temporal_cuda_2026-08-09/`
- `gen5/outputs/event_mnist_temporal_cuda_2026-08-09/analysis.md`

### 56. Phase 21 isolates LTW learning before topology learning

Date: 2026-08-09

Decision: train only the active long-term synaptic weights and task readout
while holding the sparse source/target topology fixed. STW remains zero and
inactive capacity slots remain inactive.

Rationale:

- Phase 20 established that time-preserving state is useful, so the next
  unresolved variable is the quality of the sparse dynamics.
- Adding sprouting and pruning now would confound weight learning, topology
  search, and event-rate stability.
- Hard forward spikes with a surrogate derivative provide a direct test of
  whether supervised signal reaches the fixed sparse LTWs.

Controls and diagnostics:

- matched raw, frozen-temporal, and LTW-trained temporal linear/MLP groups;
- active edge count and both physical optimizer versus effective active
  parameter counts;
- hidden event rate, mean LTW, and mean absolute LTW change;
- training time and end-to-end inference throughput.

Gate for Phase 22: introduce structural mutation only when trained LTWs beat
their paired frozen controls across seeds without hidden activity collapsing.
Otherwise diagnose learning rate, surrogate slope, recurrence, and gradient
flow first.

Artifacts:

- `gen5/ammc_gen5/trainable_temporal_mnist.py`
- `gen5/examples/sprint21_trainable_temporal_mnist.py`
- `gen5/docs/TRAINABLE_TEMPORAL_MNIST.md`
- `gen5/tests/test_trainable_temporal_mnist_contract.py`

### 57. Phase 21 proves LTW gradient flow, not useful LTW learning

Date: 2026-08-09

Finding: joint LTW/readout training changes the fixed sparse substrate but does
not reliably improve it. Linear accuracy declines by `0.31` points against its
paired frozen control; MLP accuracy gains only `0.13` points, with one of three
seeds declining.

Diagnostics:

- LTW displacement is material (`0.108` linear, `0.063` MLP), rejecting a
  disconnected-gradient explanation.
- Hidden event activity rises by `61.2%` for linear and `12.8%` for MLP.
- Training time rises by more than `70%` for both trained groups.
- The raw MLP remains `2.90` points more accurate than trained temporal MLP.

Sanity boundary:

- Supported: surrogate gradients reach active sparse LTWs.
- Not supported: the current LTW optimizer improves the representation.
- Not eligible: structural plasticity, because the Phase 21 weight-only gate
  failed.
- Still unsupported: efficiency or broad model-superiority claims.

Decision: Phase 22 remains fixed-topology and performs paired diagnostics over
readout warmup, lower LTW rates, surrogate slopes, and sensor/recurrent update
scope. Structural mutation is deferred until a weight-learning setting gains
at least `0.5` points on average, improves at least two of three seeds, keeps
event-rate ratio in `[0.5, 2.0]`, and avoids material boundary saturation.

Artifacts:

- `gen5/outputs/trainable_temporal_mnist_cuda_2026-08-09/`
- `gen5/outputs/trainable_temporal_mnist_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/ltw_diagnostics.py`
- `gen5/examples/sprint22_ltw_optimization_diagnostic.py`
- `gen5/docs/LTW_OPTIMIZATION_DIAGNOSTIC.md`
- `gen5/tests/test_ltw_diagnostics_contract.py`

### 58. Phase 22 stabilizes LTW updates but closes static-MNIST tuning

Date: 2026-08-09

Finding: no paired LTW intervention meets the practical-gain gate. The best
linear arm (`10`-epoch warmup, `3e-4` LTW rate, slope `10`) improves all three
seeds but averages only `+0.087` points. The best MLP arm is tied with frozen on
average. Every arm has zero seeds reaching the `0.5`-point practical threshold.

Mechanistic findings:

- Warmup holds hidden-event ratios near `1.0-1.04` and eliminates LTW boundary
  saturation.
- Joint `1e-3` training over-activates linear dynamics (`1.714x` event rate),
  saturates weights, and harms accuracy.
- Surrogate slopes `5` versus `10` have negligible effect at stable rates.
- Sensor-only LTW updates give a tiny linear gain; recurrent-only updates are
  neutral or negative.

Sanity boundary:

- Supported: stable, scoped sparse-LTW optimization is technically possible.
- Rejected for tested settings: useful LTW improvement on static MNIST.
- Deferred: topology mutation, because weight usefulness remains unproven.
- Not tested: continuous temporal adaptation and retention.

Decision: stop supervised LTW hyperparameter tuning on this engineering subset.
Phase 23 performs a causal recurrence ablation, disabling only hidden-to-hidden
edges while preserving sensor projections and paired readout initialization.
If recurrent state fails to add `0.5` points, move to a task with true temporal
dependence rather than using static MNIST to justify recurrent plasticity.

Artifacts:

- `gen5/outputs/ltw_optimization_diagnostic_cuda_2026-08-09/`
- `gen5/outputs/ltw_optimization_diagnostic_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/recurrence_ablation.py`
- `gen5/examples/sprint23_recurrence_ablation.py`
- `gen5/docs/RECURRENCE_ABLATION.md`
- `gen5/tests/test_recurrence_ablation_contract.py`

### 59. Phase 23 closes static MNIST as evidence for recurrence

Date: 2026-08-09

Finding: the paired causal recurrence ablation fails its practical-effect gate.
The full 384-edge recurrent state improves the full linear representation by
only `+0.107` percentage points over the matched 128-edge feedforward graph,
despite improving all three seeds. With an MLP readout, recurrence changes
accuracy by `-0.053` points and is negative in all three seeds. Both effects
are far below the pre-registered `0.5`-point threshold.

Mechanistic findings:

- Full feedforward temporal state already reaches `91.407%` linear and
  `92.613%` MLP accuracy.
- The extra 256 recurrent edges increase hidden event rate by about `11.7%`
  but provide no practical return.
- The full feedforward state adds `+1.620` linear points over sensor temporal
  state, so sparse random expansion is useful even though recurrence is not.
- Raw-pixel MLP remains the strongest tested static-MNIST model at `95.140%`.

Goal sanity boundary:

- Supported: sparse temporal/feedforward expansion can improve representation.
- Not supported on static MNIST: useful causal recurrence, useful LTW
  adaptation, structural plasticity, or superiority over conventional dense
  readouts.
- Not yet tested: whether recurrent state helps when information actually
  arrives sequentially and only final state is available.
- No claim of a Transformer alternative, best SNN, or continuous-learning
  advantage is currently justified by these image experiments.

Decision: close the static-MNIST recurrence/plasticity branch. Phase 24 streams
one image row per step and exposes only final hidden spikes and membrane
state. It compares paired 16-edge feedforward and 272-edge recurrent graphs,
plus raw, last-row, and orderless integrated-row controls. Recurrence must gain
at least `0.5` points on average and improve at least two of three seeds. A pass
unlocks fixed-topology LTW training on this sequential task; a failure triggers
time-constant and delay-buffer redesign before any plasticity claim.

Artifacts:

- `gen5/outputs/recurrence_ablation_cuda_2026-08-09/`
- `gen5/outputs/recurrence_ablation_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/sequential_mnist.py`
- `gen5/examples/sprint24_sequential_mnist.py`
- `gen5/docs/SEQUENTIAL_MNIST.md`
- `gen5/tests/test_sequential_mnist_contract.py`

### 60. Phase 24 establishes a causal recurrence benefit under memory demand

Date: 2026-08-09

Finding: recurrence passes the pre-registered sequential-memory gate by a wide
margin. With only final hidden state exposed, the 272-edge recurrent reservoir
beats the paired 16-edge feedforward projection by `+11.673` percentage points
for a linear readout and `+17.240` points for an MLP. Every seed improves, and
the smallest paired gain is `+9.50` points.

Mechanistic findings:

- Recurrent final state reaches `43.853%` linear and `55.547%` MLP accuracy.
- The orderless integrated-row controls reach only `34.787%` and `48.327%`;
  recurrence therefore adds `+9.067` and `+7.220` points beyond an intensity
  summary that discards row order.
- Hidden event rate rises only about `5.3%` over feedforward, much less than the
  accuracy effect.
- Feedforward final-state MLP is weaker than even the last-row MLP, while the
  recurrent MLP is `+11.413` points stronger than the last-row control.
- Raw-pixel MLP remains much stronger at `94.207%`, leaving a `38.66`-point
  absolute capability gap.

Goal sanity boundary:

- Supported: recurrence causally preserves useful information in a true
  sequential/final-state task.
- Supported: task design, rather than neuron count alone, determines whether
  recurrent capacity becomes useful.
- Not supported: competitive MNIST accuracy, useful LTW adaptation, structural
  plasticity, continuous learning, or catastrophic-forgetting resistance.
- Still unjustified: claims of being the best SNN or replacing Transformers.

Decision: Phase 25 trains durable weights while keeping the proven recurrent
topology fixed. It uses paired frozen, all-edge LTW, and recurrent-only LTW
arms with the stable Phase 22 schedule: ten readout-only warmup epochs followed
by five LTW epochs at `3e-4`, surrogate slope `10`. Structural plasticity
remains locked until a trained arm gains at least `0.5` points over paired
frozen, improves at least two of three seeds, holds event-rate ratio within
`[0.5, 2.0]`, and avoids material LTW saturation.

Artifacts:

- `gen5/outputs/sequential_mnist_cuda_2026-08-09/`
- `gen5/outputs/sequential_mnist_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/trainable_sequential_mnist.py`
- `gen5/examples/sprint25_trainable_sequential_mnist.py`
- `gen5/docs/TRAINABLE_SEQUENTIAL_MNIST.md`
- `gen5/tests/test_trainable_sequential_mnist_contract.py`

### 61. Phase 25 validates LTW learning and localizes the input bottleneck

Date: 2026-08-09

Finding: the paired warm-all intervention passes the durable-weight gate on
the sequential task. It improves frozen recurrent accuracy by `+2.113`
percentage points with a linear readout and `+0.893` points with an MLP. Every
seed improves; two of three MLP seeds and all three linear seeds exceed the
`0.5`-point practical threshold.

Mechanistic findings:

- Warm-all reaches `45.967%` linear and `56.427%` MLP accuracy.
- Event-rate ratios remain bounded at `1.285` linear and `1.087` MLP.
- Upper LTW saturation stays below `0.7%`; lower saturation is zero.
- Sensor LTWs move `0.10045` linear and `0.04576` MLP on average, versus only
  `0.01488` and `0.00415` for recurrent LTWs.
- Recurrent-only LTW learning gives `+0.500` mean linear points but includes a
  negative seed, and gives effectively zero MLP improvement.
- The trained recurrent MLP remains `37.78` points below the raw-pixel MLP.

Goal sanity boundary:

- Supported: stable fixed-topology LTW learning can improve the recurrent
  sequential substrate.
- Supported: sparse sensor projection quality is the immediate optimization
  bottleneck; recurrent weight adaptation alone is insufficient.
- Not supported yet: beneficial synaptogenesis, pruning, online continual
  learning, retention, or topology-driven efficiency gains.
- Competitive classification and broad architecture-superiority claims remain
  unsupported.

Decision: unlock only targeted synaptogenesis. Phase 26 preserves the original
272-edge graph and compares 16/48 additional sensor edges against 64 additional
recurrent edges and the paired fixed warm-all control. New edges appear after
the ten-epoch readout warmup and train from LTW `0.1`. Pruning remains disabled
until a growth arm beats fixed topology by at least `0.5` points, improves two
of three seeds, and preserves activity/saturation stability.

Artifacts:

- `gen5/outputs/trainable_sequential_mnist_cuda_2026-08-09/`
- `gen5/outputs/trainable_sequential_mnist_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/structural_sequential_mnist.py`
- `gen5/examples/sprint26_structural_sequential_mnist.py`
- `gen5/docs/STRUCTURAL_SEQUENTIAL_MNIST.md`
- `gen5/tests/test_structural_sequential_mnist_contract.py`

### 62. Phase 26 conditionally validates sensor growth, not general structural plasticity

Date: 2026-08-09

Finding: 48 random sensor sprouts pass the predeclared gate with the linear
readout, reaching `46.733%` versus `45.967%` for fixed warm-all, a paired mean
gain of `+0.767` percentage points. Two of three seeds achieve at least a
`0.5`-point gain, but the paired changes are heterogeneous: `-0.96`, `+1.20`,
and `+2.06` points.

Mechanistic findings:

- The 16-edge sensor arm gains `+0.493` linear points and improves all three
  seeds, but only one seed clears the practical threshold.
- The 64-edge recurrent arm gains only `+0.207` linear points.
- No structural arm improves MLP accuracy on average: sensor-16 is `-0.233`,
  sensor-48 is `-0.207`, and recurrent-64 is `-0.080` points versus fixed.
- Linear sensor sprouts move `0.0579-0.0705` from birth LTW, while recurrent
  sprouts move only `0.0127`, again localizing useful adaptation to the input
  projection.
- Event-rate ratios remain within `1.07-1.38`; mean LTW saturation remains
  below `0.6%` upper and `0.4%` lower.
- The linear sensor-48 benefit costs 48 additional active edges, expanding the
  graph from 272 to 320 edges for a small and seed-sensitive gain.

Goal sanity boundary:

- Supported: additional sensor routes can improve linear separability and can
  learn nontrivial LTWs without destabilizing activity.
- Supported: sensor growth is a better current direction than indiscriminate
  recurrent growth.
- Not supported: random synaptogenesis is not robust across readouts or seeds.
- Not supported: core pruning, continual learning, competitive MNIST, or broad
  SNN/Transformer-alternative claims.

Decision: Phase 27 tests utility-gated growth. A deterministic pool of 192
inactive sensor edges is ranked by absolute task-loss gradient after the
readout warmup; only the top 16 or 48 are retained. A paired random sensor-48
arm remains the causal control. The original 272 edges stay protected. One
conservative arm may prune at most half of the new edges, and only when their
LTW falls below 95% of birth weight after three training epochs.

Acceptance gate: gradient growth must beat paired random growth by at least
`0.5` points on average, improve at least two of three seeds, preserve an event
ratio in `[0.5, 2.0]`, and avoid material LTW saturation. Peripheral pruning
must remain within `0.25` points of the unpruned guided arm while actually
removing edges.

Artifacts:

- `gen5/outputs/structural_sequential_mnist_cuda_2026-08-09/`
- `gen5/outputs/structural_sequential_mnist_cuda_2026-08-09/analysis.md`
- `gen5/ammc_gen5/utility_gated_structural_mnist.py`
- `gen5/examples/sprint27_utility_gated_structural_mnist.py`
- `gen5/docs/UTILITY_GATED_STRUCTURAL_MNIST.md`
- `gen5/tests/test_utility_gated_structural_mnist_contract.py`

### 63. Literature inference: temporal state and delays likely precede further topology scaling

Date: 2026-08-09

Finding: a focused primary-paper review changes the interpretation of the
Phase 24-26 evidence. Strong temporal SNNs commonly combine adaptive neuron
state, eligibility-based credit assignment, explicit delays, controlled sparse
rewiring, and homeostatic regulation. AMMC currently tests only a subset of
these mechanisms in the sequential MNIST path.

Key project inferences:

- Phase 26's sensor-localized gain suggests an information-entry or temporal
  alignment bottleneck, not a general shortage of recurrent edges.
- Adaptive LIF/LSNN neurons provide slow state without increasing topology and
  should be tested before another neuron-count or recurrent-edge sweep.
- Gen-5 `delay_steps` are serialized but are not executed by the Phase 24-27
  sparse forward path; polychronization remains an architectural goal rather
  than validated computation.
- Phase 27 is a valid one-shot selector diagnostic, but dynamic sparse SNN
  papers generally use repeated prune-regrow cycles and fixed sparsity budgets.
- Continuous structural plasticity should add explicit firing-rate homeostasis
  and log silent/hyperactive neurons, churn, and edge lifetimes.
- Row-sequential MNIST should remain a causal smoke test. SHD, then SSC, should
  become the main temporal benchmarks after adaptive-neuron and delay support.
- STW/LTW and sleep replay must be evaluated through continual-learning
  retention metrics, not single-task accuracy.
- Efficiency claims must add NeuroBench-style spikes, effective synaptic
  operations, latency, memory, and backend-specific energy measurements.

Decision: run Phase 27 unchanged. Unless its result exposes an implementation
problem, the next architecture phase should be a paired LIF-versus-adLIF/LSNN
ablation at fixed topology. Delay buckets and an SHD benchmark follow before
periodic structural rewiring. This ordering tests temporal mechanism before
adding more capacity.

Artifact:

- `gen5/docs/SNN_PROJECT_INFERENCES_2026-08-09.md`

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

1. Run Phase 30 trainable delay assignment on row-sequential MNIST:
   - retain the Phase 29 distance-delay winner as paired control,
   - compare soft and straight-through per-edge delay gates,
   - test distance and flat initialization without changing topology,
   - require a paired `0.5`-point gain and two improved seeds.
2. End MNIST mechanism tuning after Phase 30:
   - carry a passing learned-delay arm or the fixed Phase 29 winner,
   - implement an SHD dataset/benchmark adapter with timing-preserving bins,
   - compare no-delay and retained-delay sparse LIF against a standard SNN
     baseline before proceeding to SSC.
3. Run the Phase 11 benchmark suite on Colab TPU/XLA:
   - `--device xla` throughput,
   - `--device xla` multi-seed convergence,
   - `--device xla` plasticity and retention ablations,
   - compare against the existing CUDA/T4 evidence.
4. Complete topology-aware hotpath throughput coverage:
   - sweep champion `--max-edges 96`, `128`, and optionally `160` using
     adjacency SHA `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`,
   - compare eager vs `--compile` for the `foraging` 8-edge prior,
   - report `tick_mode`, active-edge count, edge-pool capacity, utilization,
     `adjacency_sha256`, memory, and agent-steps/sec at 1k/10k/50k/100k.
5. Redesign gated/adult plasticity:
   - test separate gates for sprouting, pruning, LTW decay, and LTW noise,
   - add protected-core champion masks,
   - tune dopamine/fitness thresholds from retention results.
6. Run fair trained baselines:
   - BPTT-trained static LIF SNN,
   - PPO-trained MLP after installing `stable-baselines3`,
   - report fitness, active parameters, memory, and inference speed.
7. Add active-edge pressure to evolution:
   - fitness penalty per active edge,
   - lower sprout probability,
   - stronger low-LTW pruning,
   - compare fitness-per-active-synapse.
8. Add astrocyte reward/punishment coupling from `TensorEnvironment2D` into
   `DualTensorManager`.
9. Continue Gen-5 -> Gen-4 bridge calibration:
   - deterministic browser world seed,
   - Gen-5 sensor gain,
   - Gen-5 motor assist gain,
   - compare tensor-environment replay against browser replay for the same
     champion genome.
10. Use `gen5/tools/verify_phase11_outputs.py` after every future output upload
   to avoid ambiguous evidence status.

## 2026-08-09 - Phase 27 utility-gated structural MNIST result

Evidence retained at
`gen5/outputs/utility_gated_structural_mnist_cuda_2026-08-09/` from archive
SHA-256
`34CABEF370422F71E60F4ECEF1FEE5758509A6BD76D9308243DA59FA4628348C`.
The CUDA run used seeds 42-44, 20,000 train and 5,000 engineering-validation
examples, the proven 64-hidden-neuron/272-core-edge sequential graph, 15
epochs, a ten-epoch readout warmup, four candidate-scoring batches, and
peripheral pruning three epochs after growth.

Core results:

- `random_sensor_48`: `46.733%` linear and `56.193%` MLP.
- `gradient_sensor_16`: `45.260%` linear and `55.073%` MLP, deficits of
  `-1.473` and `-1.120` points versus paired random growth.
- `gradient_sensor_48`: `45.027%` linear and `54.093%` MLP, deficits of
  `-1.707` and `-2.100` points.
- `gradient_sensor_48_prune`: `45.360%` linear and `55.220%` MLP, still
  `-1.373` and `-0.973` points below random.
- Every guided-versus-random paired delta was negative across all three seeds
  and both readouts.
- Pruning removed exactly 24 of the 48 new edges, improved the unpruned guided
  arm by `+0.333` linear and `+1.127` MLP points, and never touched the core.

Mechanistic finding: this was not an activity-collapse or boundary-saturation
failure. Event-rate ratios remained about `1.06-1.39`. The linear guided edges
ended near `0.082-0.090` LTW from a `0.1` birth weight, while random edges rose
to about `0.127`; the optimizer later suppressed many candidates ranked highly
by absolute gradient at zero. MLP gradient magnitudes were much larger without
better performance. One-shot absolute sensitivity is therefore not a valid
durable utility estimate in this implementation.

Decision: reject the Phase 27 selector and schedule. Do not scale it or tune
its candidate count. Preserve conservative peripheral pruning as a validated
mechanism, but do not generalize it to the core from this experiment. This is a
negative result about one-shot absolute-gradient rewiring, not a disproof of
all structural plasticity or online gradient rewiring.

Goal sanity check: the evidence still supports the project's narrower claim
that sparse recurrent spike dynamics carry useful temporal state and can
benefit from durable LTW optimization. It does not yet support a claim that
structural growth improves conventional sequence learning. Topology quantity
is not the present bottleneck; temporal computation is the next controlled
target.

## 2026-08-09 - Phase 28 adaptive-neuron experiment generated

Decision: freeze topology and test an adaptive LIF/LSNN-style slow threshold
state before implementing delays. Phase 28 adds no trainable neuron parameters
and keeps the readout input at final spikes plus membrane, so the comparison
isolates dynamics rather than readout capacity.

Paired arms:

- `lif_frozen` and `lif_warm_all` controls;
- `alif50_frozen` to isolate fixed adaptive dynamics;
- `alif25_warm_all`, `alif50_warm_all`, and `alif100_warm_all` for a dose
  response under the successful Phase 25 LTW schedule;
- the raw linear/parameter-matched MLP ceiling.

Default state equations use adaptation decay `0.95` and threshold increment
`0.5 * adaptation`. Adaptive neuron identities are deterministically shuffled
per seed. LIF and ALIF arms share graph, LTWs, readout shape, data, and seed.

Pass gate: at least `+0.5` accuracy points over the paired LIF control, at
least two of three seeds improved, event rate within `[0.5x, 2.0x]` paired LIF,
and no material LTW saturation. A pass carries adaptive neurons into fixed
delay buckets. A failure carries ordinary LIF into the delay experiment.

## 2026-08-09 - Phase 28 adaptive-neuron result

Evidence retained at
`gen5/outputs/adaptive_sequential_mnist_cuda_2026-08-09/` from archive SHA-256
`CA3E12253AD3878C1BCE1637F49ABD7881980A746A3F776703BF2F7E2EF14761`.
The run used the registered seeds 42-44, 20,000 train and 5,000
engineering-validation samples, 64 hidden neurons, 272 active edges, 15
epochs, ten warmup epochs, adaptation decay `0.95`, and strength `0.5`.

Core results:

- Frozen LIF: `43.853%` linear and `55.567%` MLP.
- Frozen 50% ALIF: `43.533%` linear and `55.400%` MLP, paired deficits of
  `-0.320` and `-0.167` points.
- Warm LIF: `45.967%` linear and `56.373%` MLP.
- Warm 25% ALIF: `45.360%` linear and `56.067%` MLP, deficits of `-0.607`
  and `-0.307` points.
- Warm 50% ALIF: `44.587%` linear and `55.980%` MLP, deficits of `-1.380`
  and `-0.393` points.
- Warm 100% ALIF: `43.467%` linear and `55.280%` MLP, deficits of `-2.500`
  and `-1.093` points; all six paired seed/readout deltas were negative.

Mechanistic finding: the tested adaptive rule progressively suppresses useful
events. Relative to paired LIF, linear event rates fell from `0.943x` at 25%
coverage to `0.852x` at 50% and `0.729x` at 100%; MLP rates fell from `0.953x`
to `0.874x` and `0.770x`. Effective thresholds rose only to about `1.04-1.06`,
LTW changes remained stable, and saturation stayed below `0.7%`. The frozen
ALIF arm also lost, localizing the harm to dynamics rather than LTW optimizer
instability.

Decision: reject this fixed ALIF rule on the eight-step row task and do not
tune adaptation strength or coverage here. This is not evidence against ALIF
on longer speech or continual-learning sequences. It says only that suppressing
spikes after recent activity does not improve this short final-state memory
problem.

Goal sanity check: the project still has positive causal evidence for sparse
recurrence and durable LTW training, but conventional-task evidence does not
yet support adaptive thresholds or structural growth. The correct objective is
not to rescue every biological mechanism on MNIST. It is to identify which
mechanisms survive controlled tests, then move them to intrinsically temporal
benchmarks.

## 2026-08-09 - Phase 29 executable-delay experiment generated

Decision: retain ordinary LIF and make `DynamicSparseLinear.delay_steps`
executable. Each recurrent edge now reads from a fixed source-state history
bucket; delay zero reproduces the current forward path, while delays one and
two access older row states. Sensor edges remain at delay zero.

Paired arms compare no-delay frozen/warm controls with uniform recurrent delay
one, deterministic heterogeneous delays 0-2, and a reproducible hidden-index
distance proxy. Graph, initial LTWs, readout dimensions, seeds, and optimizer
budget remain fixed.

Pass gate: at least `+0.5` points over paired no-delay LIF, at least two of
three seeds improved, event rate within `[0.5x, 2.0x]`, and no material LTW
saturation. A pass permits delay-assignment optimization. A failure ends this
MNIST diagnostic track and moves the retained LIF baseline to SHD.

## 2026-08-09 - Phase 29 executable-delay result

Evidence retained at
`gen5/outputs/delayed_sequential_mnist_cuda_2026-08-09/` from archive SHA-256
`DA0276966CD7A1F8D025C244E7812EDEA31BCD0FDC1B6ED1C6B00972BDA43890`.
The run reproduces the no-delay controls and holds the 272-edge topology,
initial LTWs, readout dimensions, and seed fixed across delay interventions.

Core results:

- No-delay warm LIF: `45.967%` linear and `56.440%` MLP.
- Uniform recurrent delay one: `48.220%` linear and `54.133%` MLP, a
  `+2.253` linear gain but `-2.307` MLP loss.
- Heterogeneous hash delays 0-2: `53.967%` linear and `63.907%` MLP, gains of
  `+8.000` and `+7.467` points.
- Heterogeneous distance-proxy delays 0-2: `54.053%` linear and `64.033%` MLP,
  gains of `+8.087` and `+7.593` points.
- Every heterogeneous-delay arm improved all three seeds for both readouts.
  Distance-arm paired gains ranged from `+6.16` to `+9.76` linear and `+6.36`
  to `+8.42` MLP points.

Mechanistic finding: heterogeneity is essential. Uniformly slowing recurrence
does not transfer across readouts, whereas two distinct approximately balanced
0/1/2 delay constructions produce nearly identical large improvements. Event
rates remain about `0.98-1.02x` no-delay LIF, LTW movement is stable, and
saturation remains below `0.8%`. The gain therefore comes from temporal routing
rather than extra edges, extra neurons, global slowing, or activity inflation.

Goal sanity check: this is the strongest controlled conventional-task result
for the AMMC hypothesis. It supports sparse heterogeneous delays as a useful
temporal representation mechanism. It does not establish state of the art:
the best sparse arm reaches `64.03%` MLP versus the `94.21%` raw MLP ceiling.
Claims should remain causal and mechanistic, not competitive.

Decision: Phase 29 passes. Preserve the distance 0-2 arm as the new sequential
baseline and proceed to one final MNIST phase that optimizes delay assignment.
After that result, move the retained fixed or learned delay mechanism to SHD.

## 2026-08-09 - Phase 30 trainable-delay experiment generated

Decision: add three differentiable delay logits per recurrent edge while
holding the 272-edge topology fixed. Sensor edges are forced to delay zero and
their delay gradients are masked. LTWs and delay logits activate only after the
ten-epoch readout warmup.

The arms retain raw, no-delay, and fixed-distance controls and compare:

- soft delay mixtures initialized near the winning distance assignment;
- hard forward assignments with straight-through gradients from the same
  initialization;
- soft mixtures from flat one-third probabilities.

Delay optimization adds 768 trainable logits and reports that cost explicitly,
along with changed assignments, entropy, event rate, LTW movement, and
saturation. Pass gate: at least `+0.5` points over the paired fixed-distance
control, two improved seeds, and stable dynamics. Regardless of outcome, Phase
30 ends row-sequential MNIST tuning; Phase 31 begins SHD with the retained
winner.

## 2026-08-09 - Phase 30 trainable-delay result

Evidence retained at
`gen5/outputs/trainable_delays_mnist_cuda_2026-08-09/` from archive SHA-256
`447E5512BAB6084E0E9B3D094D07343A2ECD821CEC72485F857DF685524E3EDB`.
The run held the 272-edge graph, initial LTWs, readout dimensions, seeds, and
training schedule fixed while adding 768 delay logits to learned arms.

Core results:

- Fixed distance 0-2: `54.053%` linear and `64.073%` MLP.
- Soft distance initialization: `53.467%` linear and `64.013%` MLP, deficits
  of `-0.587` and `-0.060` points.
- Straight-through distance initialization: `54.307%` linear and `64.173%`
  MLP, gains of only `+0.253` and `+0.100` points.
- Soft flat initialization: `50.280%` linear and `57.993%` MLP, deficits of
  `-3.773` and `-6.080` points.

Mechanistic finding: the straight-through arm changed few assignments and
provided only marginal, non-practical gains. Soft distance gates slightly
degraded both readouts. Flat soft gates retained high entropy (`~1.06`),
changed many assignments, and biased selected delays toward zero instead of
recovering the balanced 0/1/2 structure. Event rates, LTW movement, and
saturation remained stable, so this is an optimization failure rather than an
activity collapse.

Decision: Phase 30 fails the registered `+0.5` point mean-improvement gate.
Reject the added delay logits and retain Phase 29's deterministic distance
0/1/2 assignment. End row-sequential MNIST tuning.

Goal sanity check: the evidence supports fixed heterogeneous temporal routing
as a causal AMMC mechanism; it does not support the claim that learned delays
are presently better or that AMMC is competitive with conventional MNIST
models. The next experiment must test transfer on genuinely event-timed data.

## 2026-08-09 - Phase 31 SHD transfer experiment generated

Decision: move the retained fixed-distance delay mechanism to the Spiking
Heidelberg Digits (SHD) benchmark. SHD provides 20 spoken-digit classes as
event times over 700 cochlear channels, with 8,156 official training samples
and 2,264 test samples. This makes temporal structure intrinsic to the data.

The registered comparison uses identical sparse topology, initial LTWs,
readout, seeds, and optimizer schedule for `sparse_no_delay_warm_all` and
`sparse_distance012_warm_all`. Event-count linear and MLP arms deliberately
discard event order and serve as timing-ablated controls. The sparse arms use
one sensor edge per input channel, four recurrent edges per hidden neuron, a
linear readout over accumulated spikes plus final membrane, five readout-only
warmup epochs, and then conservative LTW training.

Pass gate: fixed heterogeneous delays must improve paired no-delay accuracy by
at least `+1.0` percentage point on average, improve at least two of three
seeds, keep event rate within `[0.5x, 2.0x]`, and avoid material LTW
saturation. A pass establishes cross-domain transfer and motivates stronger
SHD baselines. A failure localizes the Phase 29 effect to the imposed MNIST row
encoding and sends us back to temporal representation design rather than more
delay tuning.

## 2026-08-09 - Phase 31 SHD plumbing-screen result

Screen evidence retained at
`gen5/outputs/shd_benchmark_screen_cuda_2026-08-09/` from archive SHA-256
`6318E20E86A73F912F9C20FB985EA339AE63CF1A02FC64B2ECE5474594C5DDE2`.
This was the explicitly non-registered one-seed screen: 1,000 train examples,
500 test examples, two epochs, one readout-only warmup epoch, and only the two
sparse arms. Chance accuracy is 5%.

Core observations:

- No delay: `6.0%` train and `6.0%` test accuracy.
- Fixed distance 0-2: `5.9%` train and `6.0%` test accuracy.
- The delay arm executed 348 delayed recurrent edges with mean delay `1.002`.
- Hidden event rates remained stable around `0.36-0.37`; the delayed/no-delay
  final-rate ratio was `1.015x`.
- LTW movement was `~0.00060` and both saturation rates were zero.
- Delay execution reduced measured test throughput from about 7,430 to 5,023
  examples/s in this screen.

Interpretation: the SHD data, temporal binning, CUDA path, sparse gradients,
delay routing, and serializers work. Identical near-chance accuracy under this
tiny training budget is not an ablation result. The screen omitted event-count
controls and cannot establish whether the current preprocessing is learnable.

Decision: Phase 31 remains open. Do not generate Phase 32 or tune dynamics from
screen evidence. Run the registered full matrix on all 8,156/2,264 examples,
three seeds, 15 epochs, and all four arms. A passing delay arm leads to
capacity-matched published SHD baselines; a near-chance sparse failure with
successful count controls leads to a temporal-representation diagnostic.

## 2026-08-09 - Phase 31 full SHD transfer result

Evidence retained at `gen5/outputs/shd_benchmark_cuda_2026-08-09/` from
archive SHA-256
`698E42533E42C251ECBD8F38399C0527AA05F17DE7F68A6363B9C0D64985CFAE`.
This is the registered full run: all 8,156/2,264 official examples, seeds
42-44, 15 epochs, five warmup epochs, and all four arms. Although the uploaded
request calls it Phase 32, this result scientifically closes Phase 31.

Core results:

- Event-count linear: `47.350%` test accuracy.
- Event-count MLP: `51.914%`.
- Sparse no delay: `36.204%`.
- Sparse fixed distance 0-2: `36.425%`, only `+0.221` points versus no delay.
- Per-seed delay gains were `+0.398`, `+0.309`, and `-0.044` points. Two seeds
  improved, but none reached the registered `+1.0` practical threshold.

Mechanistic finding: preprocessing is learnable because both event-count
controls perform far above 5% chance. The sparse model underfits, trailing the
count-linear control by `11.15` points and count-MLP by `15.71` points. Event
rates remain high but stable near `0.36-0.37`; the delay/no-delay rate ratio is
`1.002x`. LTW movement is `~0.056`, lower saturation is `~0.1%`, and upper
saturation is only `3.0-3.4%`. The failure is representational, not a dead
network, broken optimizer, or data-pipeline failure. Delays also add roughly
57% training time and reduce inference throughput by about one third.

Decision: Phase 31 fails the fixed-delay cross-domain gate. Preserve Phase 29
as a valid task-specific causal result, but reject any universal-delay claim.
Do not tune delay logits or delay patterns next.

Goal sanity check: AMMC remains interesting as a compact continuous-time sparse
framework, but current evidence does not support SHD competitiveness. The next
experiment must locate the 11-16 point representation gap before comparison
with published SHD systems.

## 2026-08-09 - Phase 32 SHD representation diagnostic generated

Decision: decompose the SHD gap with a registered diagnostic rather than a
leaderboard attempt. Phase 32 retains the event-count controls and tests:

- linear versus MLP decoding of the same 128-neuron no-delay representation;
- paired no-delay versus fixed-distance delays under the stronger MLP decoder;
- 128 versus 256 hidden neurons under the same MLP/delay configuration;
- a higher firing threshold (`1.5`) to test whether the observed `~36%` event
  rate is washing out temporal selectivity.

The topology remains fixed within each paired comparison and LTWs retain the
same warmup/training schedule. Diagnostic gates are `+3` points for nonlinear
decoding, `+1` point for delay transfer under MLP, `+3` points for capacity,
and `+2` points for activity control with a materially lower event rate. The
winner, if any, becomes the registered SHD baseline; otherwise the next step is
a redesigned temporal encoder rather than larger reservoirs.

## 2026-08-09 - Phase 33 SHD representation result

Evidence retained at `gen5/outputs/shd_representation_cuda_2026-08-09/` from
archive SHA-256
`A36329DB52A5526CF9A0393574EF4B2F8F31453EE9256C3BC9CAB3E2D17B916B`.
This result corresponds to the repository's Phase 32 diagnostic runner and the
project's externally numbered Phase 33 result.

Core results:

- Sparse 128 linear no delay: `36.278%`.
- Sparse 128 MLP no delay: `42.535%`, a `+6.257` point decoder gain; all three
  seeds clear the registered `+3` point gate.
- Sparse 128 MLP distance delays: `42.609%`, only `+0.074` points over paired
  MLP no delay. Delays fail again.
- Sparse 256 MLP distance delays: `54.711%`, a `+12.102` point capacity gain;
  all three seeds clear the `+3` point gate.
- Sparse 128 threshold 1.5: `44.405%`, a directional `+1.796` point gain that
  misses the `+2` point gate.
- Event-count MLP: `51.914%` with 92,308 parameters. The 256-neuron sparse arm
  exceeds it by `2.80` points with 69,968 effective parameters.

Mechanistic finding: both readout nonlinearity and hidden width are genuine
bottlenecks. Width is dominant. The 256-neuron graph also lowers event rate to
`23.59%` from `35.46%`, moves LTWs by only `~0.022`, and keeps upper saturation
below `0.4%`. The fixed-delay effect remains negligible under the MLP decoder,
so delays cannot explain the wider model's gain.

Goal sanity check: this is positive evidence for sparse recurrent temporal
representation rather than a universal delay mechanism. The model now beats a
count-based MLP control with fewer effective parameters, but `54.71%` is not a
competitive SHD result and does not support state-of-the-art claims.

Decision: retain MLP decoding and 256 hidden neurons as the new SHD baseline.
Retire delays from the next optimization phase. Validate scaling and locate the
capacity/efficiency knee before adding new biological mechanisms.

## 2026-08-09 - Phase 34 SHD capacity-scaling result

Evidence retained at `gen5/outputs/shd_capacity_cuda_2026-08-09/` from archive
SHA-256
`B6ABF47978285467E8F77108733D390AFFB7666B91D9D5F8C2AA56771D34C794`.

Core no-delay results:

- 128 neurons: `42.624%`, 36,688 effective parameters, event rate `35.07%`.
- 192 neurons: `48.837%`, a `+6.213` point gain.
- 256 neurons: `51.929%`, a `+9.305` point gain; all seeds improve and two
  clear the registered `+8` point primary gate.
- 384 neurons: `57.759%`, a `+5.830` point gain over 256.
- 512 neurons: `60.615%`, a `+8.687` point gain over 256; all seeds clear the
  `+2` point secondary gate.

Mechanistic finding: width robustly improves accuracy and lowers hidden event
rate from `35.07%` at 128 neurons to `14.00%` at 512, while LTW movement and
saturation remain stable. But accuracy per thousand effective parameters falls
monotonically from `0.01162` to `0.00444`; width is an accuracy lever, not an
efficiency improvement. The 256-neuron no-delay model matches the 51.914%
event-count MLP with 24% fewer effective parameters, while 384 and 512 exceed
it with more parameters.

Unexpected delay interaction: the 256-neuron distance-delay arm reaches
`54.682%`, `+2.753` points over no delay. All seeds improve, but paired gains
are uneven (`+1.634`, `+6.581`, `+0.044`). Delays add roughly 55% training time
and reduce inference throughput by about one third.

Goal sanity check: AMMC now shows robust sparse-capacity scaling and beats the
count control, but 60.6% remains noncompetitive and parameter efficiency
declines. More width alone is not the endpoint. The apparent width-dependent
delay gain needs factorial replication before it can revise the prior delay
conclusion.

## 2026-08-09 - Phase 35 SHD capacity-delay interaction generated

Decision: test 256 and 512 hidden neurons under four paired timing patterns:
no delay, uniform recurrent delay one, heterogeneous hash delays 0-2, and
heterogeneous distance delays 0-2. Each scale/pattern retains identical graph,
initial LTWs, MLP readout, optimizer schedule, data, and seed.

Pass gate for a heterogeneous pattern: at least `+2` mean points versus paired
no delay, at least two seeds gain `+1` point, event rate remains within
`[0.5x, 2.0x]`, and LTW saturation remains stable. Uniform delay one is the
generic-slowing control. A heterogeneous pass at both widths establishes a
capacity-delay interaction; a one-width or one-seed effect is treated as
unstable and the next phase moves to temporal encoder redesign.

## 2026-08-09 - Phase 35 SHD capacity-delay interaction result

Evidence retained at `gen5/outputs/shd_delay_interaction_cuda_2026-08-09/`
from archive SHA-256
`8452829B4AB6AF10DB51A3A49998C156914A5767756FC0ED117D799D4F1761FB`.

At 256 neurons, uniform, hash 0-2, and distance 0-2 delays improve over the
paired no-delay mean by `+2.532`, `+2.577`, and `+2.473` points respectively.
The heterogeneous arms clear the registered single-width gate, but the nearly
identical uniform-delay gain and concentration of improvement in seed 43 make
the mechanism consistent with generic temporal slowing rather than uniquely
heterogeneous polychronization.

At 512 neurons, uniform, hash, and distance delays add only `+0.103`, `+0.324`,
and `+0.427` points. No delayed arm has a seed gaining one point, so the
registered cross-capacity hypothesis fails. Activity, LTW movement, and
saturation stay stable; the negative result is not caused by firing collapse.
Delayed inference is also roughly 40% slower at 512 neurons.

The 512-neuron no-delay mean (`60.704%`) reproduces Phase 34 (`60.615%`) within
`0.089` points. Decision: retain 512/no-delay as the reliable SHD baseline,
stop delay tuning, and test whether preserving temporal order at the readout is
the next useful lever. Accuracy remains far from competitive SHD systems, so
the evidence does not support state-of-the-art claims.

## 2026-08-09 - Phase 36 SHD temporal-pyramid experiment generated

Decision: replace global spike averaging with a parameter-matched temporal
pyramid over 1, 2, 4, and 8 contiguous windows while retaining final membrane
state. A shared 32-dimensional projection keeps readout capacity close to the
paired global MLP budget. Test global versus pyramid readouts at 256 and 512
hidden neurons, plus a fixed time-shuffled pyramid at 512 neurons.

Primary gate: the ordered 512-neuron pyramid must improve by at least `+3` mean
accuracy points over the paired global readout, with at least two seeds gaining
`+2` points and event rate remaining within `[0.5x, 2.0x]`. Causal timing gate:
ordered pyramid must exceed the parameter-identical shuffled-time control by at
least `+2` mean points across two seeds. Parameter counts must remain within
`10%` of the global baseline. Passing only the first gate indicates richer
summary features; passing both supports temporal-order-sensitive computation.

## 2026-08-09 - Phase 34 SHD capacity-scaling experiment generated

Decision: run paired no-delay MLP reservoirs at 128, 192, 256, 384, and 512
hidden neurons, plus the event-count MLP reference and one 256-neuron fixed-delay
comparator. All scales retain one sensor projection per channel, four recurrent
edges per hidden neuron, the same optimizer schedule, and the same three seeds.

Primary gate: the no-delay 256-neuron arm must improve by at least `+8` points
over no-delay 128 neurons across at least two seeds, confirming that Phase 33
was not a delay interaction. Secondary gate: 384 or 512 neurons must add at
least `+2` points over 256 across two seeds to justify further width. The runner
reports accuracy per 1,000 effective parameters, activity, LTW saturation, and
throughput to identify the useful scaling knee. The 256 delay comparator keeps
the universal-delay hypothesis falsifiable without expanding delay tuning.

## 2026-08-09 - Phase 36 SHD temporal-pyramid result

Evidence retained at `gen5/outputs/shd_temporal_pyramid_cuda_2026-08-09/`
from archive SHA-256
`419F26034B7FA13995025808901C58BFCD23A049414B7D94DEE572B88B76D8C7`.

The parameter-matched ordered temporal pyramid reaches `76.193%` at 256 hidden
neurons and `80.065%` at 512. These are paired gains of `+23.910` and `+19.287`
points over global pooling. Every seed clears the registered practical gate.
At 512 neurons the ordered model exceeds the parameter-identical fixed-shuffle
control by `+6.257` points, with every seed gaining more than five points.

The fixed shuffle itself reaches `73.807%`, `+13.030` points over global. Since
the permutation is fixed across examples, it preserves position-specific
information while disrupting natural chronology and local recurrent
continuity. The combined evidence says time-resolved features provide most of
the gain, with an additional robust chronology-sensitive component.

The result is not caused by extra capacity or unstable dynamics: the 512
pyramid uses `99.38%` of global effective parameters, lowers event rate from
`14.00%` to `13.16%`, lowers LTW movement and saturation, and costs only about
9% inference throughput. This is the project's strongest SHD result, but
`80.065%` remains below strong published systems and does not justify
state-of-the-art claims.

Decision: do not tune pyramid resolution yet. First isolate whether recurrence
and the sparse reservoir add value beyond a parameter-matched temporal decoder
on raw events.

## 2026-08-09 - Phase 37 SHD temporal-control decomposition generated

Decision: compare five paired 512-neuron or matched-budget arms: event-count
MLP, raw-event temporal pyramid, global AMMC, feedforward AMMC temporal pyramid
with recurrent edges disabled, and recurrent AMMC temporal pyramid.

Recurrence gate: recurrent pyramid must beat feedforward pyramid by at least
`+3` mean points with at least two seeds gaining `+2` points. Reservoir gate:
recurrent pyramid must beat the matched raw temporal model by at least `+2`
mean points with at least two seeds gaining `+1` point. If raw temporal matches
or wins, Phase 36 is primarily a readout result and the recurrent core must be
redesigned before further scaling. Activity and LTW saturation must remain
stable.

## 2026-08-10 - Phase 37 SHD temporal-control result

Evidence retained at `gen5/outputs/shd_temporal_controls_cuda_2026-08-10/`
from archive SHA-256
`0D075AD9404F7E0769454189DDCCD6F6022014FD025C7BB2DCC621001B017E7E`.

Mean accuracies are `51.914%` for event-count MLP, `77.959%` for the matched
raw temporal pyramid, `60.998%` for global AMMC, `79.623%` for feedforward AMMC
pyramid, and `80.271%` for recurrent AMMC pyramid.

The reservoir gate passes narrowly: recurrent AMMC gains `+2.312` mean points
over raw temporal, all seeds improve, and two gain at least one point. The
recurrence gate fails: recurrent AMMC gains only `+0.648` points over the
feedforward sparse model, no seed gains two points, and one seed declines.

Recurrence increases hidden event rate from `8.34%` to `13.16%` for that small
gain. The raw temporal control is roughly 5.7 times faster than recurrent AMMC
and uses 98.0% as many effective parameters. Current evidence therefore
attributes most of Phase 36 to temporal decoding, a modest amount to sparse
feedforward LIF transformation, and no practical causal value to the tested
random recurrent graph.

Decision: freeze recurrence tuning. Establish matched conventional baselines
before redesigning the core.

## 2026-08-10 - Phase 38 SHD matched-baseline suite generated

Decision: compare event-count and raw temporal controls, a parameter-matched
standard dense recurrent LIF trained by BPTT, a parameter-matched GRU, sparse
feedforward AMMC, and sparse recurrent AMMC on the identical SHD split and
three seeds.

All trainable comparators must remain within `10%` of the recurrent AMMC
effective parameter budget. Sparse-advantage gate: recurrent AMMC must exceed
the dense LIF mean by `+2` points with at least two seeds gaining `+1` point.
The GRU is a conventional temporal reference, not a gate to be tuned against.
Report inference throughput and activity alongside accuracy. If either matched
baseline wins, the next phase must redesign AMMC dynamics or learning rather
than further tune the temporal readout.

## 2026-08-10 - Phase 38 SHD matched-baseline result

Evidence retained at `gen5/outputs/shd_matched_baselines_cuda_2026-08-10/`
from archive SHA-256
`D2AD248E20CD0EB551A4B0BB089B969CC1C47E2E0DDE0B0D345AC2D789162928`.

Sparse recurrent AMMC reaches `79.873%` versus `73.763%` for the matched dense
recurrent LIF, a paired `+6.110` point gain. Every seed improves by at least one
point, so the registered sparse-advantage gate passes. Sparse feedforward AMMC
also reaches `79.417%` (`+5.654` points versus dense LIF), while recurrence adds
only `+0.456` points over feedforward and fails its earlier causal gate again.

The raw temporal control reaches `77.577%`, `+3.813` points above dense LIF and
only `2.297` below sparse recurrent AMMC, while running about `5.18x` faster.
Sparse recurrent activity is lower than dense LIF (`13.15%` versus `20.07%`),
but the current implementation has no inference-throughput advantage. The GRU
reference reaches only `44.464%` despite high train accuracy and large seed
variance; treat it as an overfit/under-calibrated failed reference rather than
evidence of general superiority.

Decision: the defensible claim is narrow—this sparse feedforward LIF expansion
beats this matched dense LIF. Recurrence, structural plasticity, state-of-the-art
SHD performance, and hardware efficiency remain unsupported. Freeze recurrence
tuning and isolate hard spikes and LTW optimization before continuing.

## 2026-08-10 - Phase 39 sparse SHD mechanism ablation generated

Decision: cross hard LIF versus analog leaky dynamics with frozen versus
trainable LTWs while holding the 512-node feedforward sensor graph and temporal
pyramid decoder fixed. Retain the raw temporal pyramid as the paired reference.

Spiking gate: trainable-LTW LIF must exceed trainable-LTW analog by at least
`+2` mean points, with at least two seeds gaining `+1` point. LTW gate:
trainable-LTW LIF must exceed frozen-LTW LIF by at least `+1` mean point with at
least two positive seeds. Sparse-representation gate: trainable LIF must retain
at least a `+2` point mean gain over raw temporal. Frozen arms report zero LTW
movement by construction. Failure of the spiking or LTW gate requires narrowing
the AMMC mechanism claim; passing identifies what deserves the next structural
plasticity experiment.

## 2026-08-10 - Phase 39 sparse SHD mechanism result

Evidence retained at `gen5/outputs/shd_sparse_mechanisms_cuda_2026-08-10/`
from archive SHA-256
`B4A021423C00270E4B29014250D196DAC10AD4CD7539E116CA72399F7BEEF002`.

The raw temporal model reaches `77.959%`. Frozen and trainable sparse LIF reach
`79.623%` and `79.608%`, while frozen and trainable sparse analog models both
reach `80.624%` to displayed precision. Hard spiking is therefore `1.016`
points worse than its matched analog control. No seed improves and the spiking
gate fails in the opposite direction.

LTW optimization also fails. Trainable LIF changes LTWs by `0.01525` on average
but loses `0.015` points relative to frozen LIF. Trainable analog LTWs change by
`0.00798`, yet the paired mean gain is effectively zero and seed effects cancel.
The LIF sparse gain over raw temporal is only `+1.649` points, below its
registered `+2` gate. Frozen analog retains the sole passing signal at `+2.665`
points over raw temporal and runs about 23% faster than LIF.

Decision: the current evidence does not support hard spikes, trainable LTWs,
recurrence, or structural plasticity as the source of SHD performance. The
surviving candidate is a frozen sparse analog expansion with leaky temporal
state. Reclassify the current SHD result as an architecture observation, not an
SNN-mechanism result, until analog dynamics and topology are decomposed.

## 2026-08-10 - Phase 40 SHD analog/topology controls generated

Decision: compare raw temporal decoding, matched dense recurrent LIF, dense
analog feedforward and recurrent networks, and frozen sparse analog expansion
with instant versus leaky state. All arms use the same data split, temporal
pyramid family, parameter target, and seeds.

Analog gate: dense recurrent analog must exceed dense recurrent LIF by `+2`
mean points with two seeds gaining `+1`. Sparse-topology gate: sparse leaky
analog must exceed dense feedforward analog by `+2` points with two one-point
seed gains. Leak gate: sparse leaky analog must exceed sparse instant analog by
`+1` mean point with at least two positive seeds. It must also retain the
`+2`-point raw-temporal gain. These controls determine whether the remaining
effect belongs to analog dynamics, low-cost sparse width, or temporal leak.

## 2026-08-10 - Phase 40 SHD analog/topology result

Evidence retained at `gen5/outputs/shd_analog_topology_cuda_2026-08-10/`
from archive SHA-256
`EE66EB871400680B16803533061F296BDFF35C82F2B62ED703003DBB6D6AD27F`.

Sparse leaky analog reaches `81.140%`, versus `79.078%` for sparse instant
analog, `77.959%` for raw temporal, `75.501%` for dense feedforward analog,
`74.264%` for dense recurrent LIF, and `71.555%` for dense recurrent analog.

The sparse-topology gate passes strongly: sparse leaky exceeds matched dense
feedforward analog by `+5.639` points and every seed gains at least `4.9`
points. The leak gate also passes: leaky exceeds instant sparse analog by
`+2.061` points with all seeds positive. It retains a `+3.180` point gain over
raw temporal across all seeds.

The broad analog gate fails. Dense recurrent analog is `2.709` points worse
than dense recurrent LIF, while dense feedforward analog adds only `1.237`
points. Analog activation alone is not explanatory; recurrence remains harmful.

Decision: the current supported mechanism is a low-cost frozen sparse input
expansion combined with leaky temporal state and a temporal readout. It is not
an SNN or plasticity result. Test its fixed-budget width scaling and topology
occupancy before considering a new spiking formulation.

## 2026-08-10 - Phase 41 fixed-budget sparse width scaling generated

Decision: compare 128, 256, 512, and 1024 hidden-node sparse leaky analog
expansions plus the raw temporal control. Every sparse arm has exactly 700
frozen sensor edges, no recurrent edges, and the same `133,631` effective-model
parameter target; the readout bottleneck absorbs width changes.

Width gate: 512 nodes must beat 128 by `+2` mean points with two one-point seed
gains. Further-scaling gate: 1024 must beat 512 by `+1` point with two positive
seeds. The best width must retain at least `+2` points over raw temporal.
Connected hidden-node count, occupancy, fan-in, throughput, and fixed-budget
ratios are reported to locate the useful width knee and distinguish capacity
from topology coverage.

## 2026-08-10 - Phase 41 fixed-budget sparse width result

Evidence retained at `gen5/outputs/shd_sparse_width_cuda_2026-08-10/`
from archive SHA-256
`562E5813608B99ADB8EC54BB1C5A9ABDAE66B98F54C4A2DE5CE566B8B6E3A5FF`.

At a fixed effective parameter target, 128, 256, 512, and 1024 sparse nodes
reach `62.898%`, `74.823%`, `77.856%`, and `78.696%`; raw temporal reaches
`77.959%`. The 512-versus-128 width gate passes by `+14.959` points across all
seeds. The 1024-versus-512 gain is only `+0.839` points, so it misses the
registered `+1` further-scaling gate despite all seeds improving.

The best width gains only `+0.736` points over raw temporal, with just one seed
gaining a point. The absolute sparse-advantage gate therefore fails. Occupancy
falls from `99.7%` at width 128 to `50.2%` at 1024 while connected nodes rise
from `127.7` to `513.7`; mean fan-in falls from `5.48` to `1.36`. Narrow models
show high analog activity and severe information-collision loss.

The 512-node mean also fails to reproduce Phase 40 (`77.856%` versus
`81.140%`). The conceptual architecture and budget match, but Phase 41 replaces
the readout after constructing the original one, changing its RNG position.
This exposes initialization sensitivity that the original three-seed protocol
did not separate from topology variance.

Decision: width is genuinely important up to the sensor-coverage knee, but a
stable advantage over the raw temporal decoder is not established. Do not tune
width further. Factor topology seeds from readout/optimizer seeds and quantify
both variance sources before retaining or abandoning the sparse transform.

## 2026-08-10 - Phase 42 SHD initialization robustness generated

Decision: run raw temporal at three readout seeds and sparse 512/1024 models on
a `3 topology x 3 readout` seed matrix. The final readout is explicitly reseeded
after graph construction, removing graph allocator and constructor RNG order as
a confound. Optimizer batch order uses the readout seed.

Robust sparse gate: a sparse width must beat its paired raw readout by `+2`
mean points across the nine topology/readout pairs, with at least six positive
pairs. Further-scaling gate: 1024 must beat paired 512 by `+1` point with at
least six positive pairs. Report between-topology and mean within-topology
readout standard deviations. If neither sparse width passes, the temporal
decoder remains the principal SHD contribution and sparse expansion is demoted
to an initialization-sensitive auxiliary transform.

## 2026-08-10 - Phase 42 SHD initialization robustness result

Evidence retained at
`gen5/outputs/shd_initialization_robustness_cuda_2026-08-10/` from archive
SHA-256 `70AF5E4E662B195862063DFD68FC98893D975376BAF59C9230DAB6C5817A0394`.

Across independent readout/optimizer seeds, raw temporal reaches `78.357%`,
sparse 512 reaches `78.058%`, and sparse 1024 reaches `77.380%`. Sparse 512 has
a paired `-0.299` point mean gain versus raw and wins only `3/9` pairs; sparse
1024 has a `-0.977` point gain and also wins `3/9`. Both robust sparse gates
fail. The 1024 model is `0.677` points worse than paired 512 and wins `4/9`, so
the further-scaling gate fails too.

Readout/optimizer initialization is the larger variance source. At width 512,
mean within-topology readout standard deviation is `1.863` points versus
`1.025` points between topology means. At 1024 these are `1.268` and `0.854`
points. The best individual sparse result (`81.449%`) is therefore a selection
outlier, not evidence of a stable architectural gain.

Decision: demote the frozen sparse analog expansion to an
initialization-sensitive auxiliary transform. The reproducible SHD contribution
is the temporal pyramid decoder. Run one validation-selected checkpoint audit
to test whether conventional overfitting explains the instability; if sparse
still fails, freeze this branch and move to calibrated temporal baselines or a
fundamentally new spiking formulation.

## 2026-08-10 - Phase 43 SHD validation-selected checkpoint audit generated

Decision: compare raw temporal and sparse-512 models using a fixed stratified
10% validation split. Run three raw readout seeds and a `3 topology x 3 readout`
sparse matrix. Train all models for 15 epochs, then report both final-epoch and
best-validation checkpoints without consulting test accuracy.

Final sparse gate: the validation-selected sparse checkpoint must beat its
paired raw checkpoint by `+2` mean points with at least six positive pairs.
Checkpointing should reduce sparse test standard deviation by at least 25%
without reducing its mean by more than `0.5` points. Failure closes the current
SHD sparse-expansion branch; passing would identify overfitting rather than the
transform itself as the primary weakness.

## 2026-08-10 - Phase 43 SHD validation-selected checkpoint result

Evidence retained at `gen5/outputs/shd_validation_checkpoint_cuda_2026-08-10/`
from archive SHA-256
`663F7AC8BFD01A549981FDA669CA70FFCB4122DF2794A3875F46C65AC5192877`.

Raw temporal improves from `78.092%` at the final epoch to `80.374%` using the
best validation checkpoint. Sparse 512 changes from `78.195%` to `78.023%`.
The validation-selected sparse model trails its paired raw reference by
`2.351` points, wins only `2/9` pairs, and exceeds raw by two points in only one
pair. The final sparse gate fails decisively.

Checkpoint selection does not stabilize sparse performance. Sparse standard
deviation rises from `1.301` to `1.577` points and mean accuracy falls by
`0.172` points. Raw checkpointing raises accuracy by `2.282` points, although
its three-seed deviation also rises. Sparse best validation accuracy is only
`83.606%`, versus `91.830%` for raw, showing a representation/optimization
deficit rather than simple late-epoch overfitting.

Decision: close the current SHD sparse-expansion branch. Its isolated high
scores do not survive independent initialization or validation selection.
Retain the temporal pyramid as the reproducible result, freeze claims about
spiking, recurrence, LTW/STW plasticity, and sparse superiority, and establish
calibrated conventional temporal baselines before designing a new spiking core.

## 2026-08-10 - Phase 44 calibrated SHD temporal baselines generated

Decision: compare four validation-selected models on the identical stratified
split and seeds at an approximately `133,631` trainable-parameter budget: raw
temporal pyramid, a one-layer temporal Conv1D with multi-scale pooling, a GRU,
and dense recurrent LIF. This repairs the under-calibrated GRU reference from
Phase 38 and adds a strong local-temporal ANN control.

Calibration gate: each model is selected only by validation accuracy and must
remain within 5% of the target parameter budget. The raw temporal decoder is
considered competitive only if it remains within `2` mean test points of the
best matched ANN. Any matched baseline exceeding raw by `+2` points across at
least two seeds becomes the minimum target for a future spiking redesign. This
phase establishes the honest SHD ceiling; it does not test an AMMC mechanism.

## 2026-08-10 - Phase 44 result: local temporal convolution is the new target

Evidence retained at
`gen5/outputs/shd_calibrated_baselines_cuda_2026-08-10/` from archive SHA-256
`558F9DAA53050B9A8F2EA6FE43B85B7FA5AC427615820DB96C6862CC4517FBAD`.

The validation-selected temporal Conv1D reaches `82.847% +/- 0.930` points,
versus `80.374% +/- 1.838` for the raw temporal pyramid and
`75.103% +/- 2.355` for dense recurrent LIF. Conv1D improves over raw by
`+2.473` mean points and wins all three seeds, but only one seed clears the
strict `+2`-point threshold. Therefore raw fails the within-two-points
competitive gate, while Conv1D shows directional rather than fully replicated
strict dominance.

Conv1D is also the most useful practical reference: about `51,371` test
examples/s versus `37,997` for raw and `11,748` for dense LIF, with a lower
three-seed accuracy deviation than raw. The calibrated GRU reaches only
`46.363%`; its best validation accuracy is also low (`58.047%`), so this
specific small-GRU formulation is unsuitable. This is not evidence that GRUs
in general are weak.

Sanity decision: do not claim SNN, sparse, or AMMC superiority from the current
SHD work. The reproducible scientific result is that learned local temporal
filters provide the strongest matched representation found so far. A future
spiking core must reach at least `80.85%` (within two points of Conv1D) and
materially exceed the `75.10%` dense-LIF reference before mechanism ablations
are warranted.

## 2026-08-10 - Phase 45 learned spiking temporal convolution generated

Decision: transplant the successful trainable temporal Conv1D front end into
two explicitly stateful variants: a leaky analog core and a surrogate-gradient
LIF core. Compare them against the Phase 44 Conv1D, raw temporal pyramid, and
dense recurrent LIF on identical splits, seeds, validation selection, and an
approximately `133,631`-parameter budget.

Primary viability gate: convolutional LIF must finish within `2` mean test
points of both Conv1D and leaky analog controls, with at least two of three
seeds within that margin and a non-degenerate spike rate between `1%` and
`30%`. Architectural-improvement gate: it must beat dense recurrent LIF by
`3` mean points and by at least `3` points on two seeds. Failure ends this
redesign branch after one diagnostic phase; success permits mechanism and
energy ablations.

## 2026-08-10 - Phase 45 result: state placement fails before spike thresholding

Evidence retained at
`gen5/outputs/shd_spiking_temporal_conv_cuda_2026-08-10/` from archive SHA-256
`66D5D92F64F290F7EACE46FCED03EF378F93E8E0B039A889B01D82CE457B30DF`.

The replicated Conv1D reference reaches `82.921% +/- 0.998` points. Leaky
analog state-only processing reaches `76.472% +/- 1.180`, a `-6.449`-point
loss, and leaky LIF reaches `74.308% +/- 1.307`, a `-8.613`-point loss versus
Conv1D and `-2.164` points versus analog. LIF is also `-0.795` points below the
dense recurrent LIF reference and clears none of the required `+3`-point
paired comparisons.

The LIF spike rate is `28.963%` on average, with individual seeds from
`24.800%` to `32.758%`. The model is therefore active rather than silent; dead
neurons or an excessively high initial threshold cannot explain the primary
accuracy loss. The analog predecessor already loses most of the performance,
showing that replacing direct local Conv1D features with accumulated temporal
state is the dominant failure. Thresholding adds a smaller secondary loss.

Sanity decision: Phase 45 fails both the primary viability gate and the
architectural-improvement gate. Do not tune thresholds or claim an energy
advantage: LIF also processes only about `19,954` test examples/s versus
`49,966` for Conv1D. Permit one branch-closing state-placement diagnostic.

## 2026-08-10 - Phase 46 state-placement diagnostic generated

Decision: compare state-only analog/LIF arms with matched residual variants
that preserve pooled direct Conv1D features beside the accumulated state. This
tests whether state replacement destroys a useful representation, without
mislabeling a direct ANN bypass as a successful spiking core.

Recovery gate: each residual arm must improve by at least `4` mean points over
its matching state-only arm, with two seeds clearing `+4`, and finish within
`2` mean points of Conv1D. If residual LIF recovers only because of the direct
bypass, a subsequent component ablation must show that removing its spike
branch causes a measurable loss before it can count as a spiking contribution.
If the recovery gate fails, close the current SHD stateful redesign immediately.

## 2026-08-10 - Phase 46 result: residual feature preservation recovers accuracy

Evidence retained at
`gen5/outputs/shd_state_placement_diagnostic_cuda_2026-08-10/` from archive
SHA-256
`F90F7D11AE2040E0AFD567F0053C7709C3144DBEDB79E8C19A7D655C0C78D312`.

The matched Conv1D reference reaches `82.862% +/- 0.862` points. Residual
analog reaches `83.142% +/- 1.007`, recovering `+5.933` mean points over its
state-only arm; two seeds clear the `+4`-point gate and all three finish within
two points of Conv1D. Residual LIF reaches `83.804% +/- 1.520`, recovering
`+8.525` points over state-only; all three seeds clear `+4` and remain within
two points of Conv1D. Both recovery gates pass.

Residual LIF exceeds Conv1D by `+0.942` mean points, but the paired differences
are `-1.413`, `+4.196`, and `+0.044` points. The mean gain is therefore driven
mostly by seed `143` and is not a robust superiority result. It also processes
about `19,587` test examples/s versus `51,781` for Conv1D. Healthy `25.736%`
spike activity establishes viability, not efficiency or causal usefulness.

Sanity decision: Phase 46 proves that the Phase 45 failure came from replacing
direct local features with state. It establishes a viable hybrid architecture,
not a successful standalone SNN. The residual model can still solve the task
entirely through its direct Conv1D path. Require a post-training component
ablation before making any claim about the state or spike branch.

## 2026-08-10 - Phase 47 residual-state contribution ablation generated

Decision: train matched residual analog and residual LIF models, select by the
same validation protocol, then evaluate each fixed checkpoint in four modes:
full, direct-only, state-only, and batch-shuffled state. No ablation mode is
retrained, so the test measures dependence of the learned solution.

Contribution gate: full accuracy must exceed direct-only by at least `1` mean
point with two seeds clearing `+1`. Specificity gate: full must exceed
shuffled-state accuracy by at least `1` mean point with two seeds clearing
`+1`. Both gates and the existing within-two-points Conv1D viability condition
must pass to justify cross-dataset replication. Otherwise close the residual
state claim and retain Conv1D as the honest SHD result.

## 2026-08-10 - Phase 47 result: residual state is causally used on SHD

Evidence retained at
`gen5/outputs/shd_residual_state_contribution_cuda_2026-08-10/` from archive
SHA-256
`3971FF33F009DC1242BC51F395CB1DBBF417C3B8E6C476315E3D136350BAF1E5`.

Residual LIF reaches `83.908% +/- 0.435` points versus a paired Conv1D mean of
`82.656%`, a `+1.251`-point mean gain. Removing its state features reduces
accuracy to `77.488%`, a `6.419`-point loss. Shuffling state between samples
reduces accuracy to `79.741%`, a `4.167`-point loss. All three seeds clear both
pre-registered `+1`-point causal gates. State alone reaches only `22.144%`, so
the result is cooperative rather than a standalone spiking solution.

Residual analog shows even stronger co-dependence: full `83.746%`, direct-only
`52.680%`, state-only `18.802%`, and shuffled state `59.364%`. Because removing
features from a jointly trained classifier creates distribution shift, the
direct-only drop alone is insufficient evidence. The shuffled-state test is
more informative: it preserves the state distribution while breaking sample
identity, and its replicated loss supports sample-specific state use.

Sanity decision: this is the first causal evidence that the residual spike
state contributes to classification beyond the direct path. It does not prove
generalization beyond SHD, standalone SNN superiority, or efficiency; residual
LIF throughput is only about `18,503` test examples/s. Proceed to one official
cross-dataset replication before any broader claim.

## 2026-08-10 - Phase 48 SSC residual-LIF replication generated

Decision: use the official 35-class Spiking Speech Commands dataset with its
provided `75,466/9,981/20,382` train/validation/test splits. Compare a matched
Conv1D reference with residual LIF, then evaluate the best-validation residual
checkpoint in full, direct-only, state-only, and shuffled-state modes. The
default run uses all official samples and does not derive a validation split
from training data.

Replication gates: residual LIF must stay within `2` mean test points of Conv1D
with all three seeds within that margin. Full must exceed direct-only and
shuffled-state accuracy by at least `1` mean point, with two seeds clearing each
`+1` threshold. Passing establishes cross-dataset causal replication; failing
restricts the contribution claim to SHD. Absolute SSC accuracy is reported
descriptively and is not compared to state of the art without stronger matched
baselines.

## 2026-08-10 - Phase 48 result: causal state contribution replicates on SSC

Evidence retained at
`gen5/outputs/ssc_residual_lif_replication_cuda_2026-08-10/` from archive
SHA-256
`2575EA0A0098C7E0CDF38AA97B69DD86D30AB68246D58A9A71C00FCE624C7477`.

On all official `75,466/9,981/20,382` train/validation/test samples, residual
LIF reaches `56.498% +/- 0.360` points versus `49.248%` for matched Conv1D, a
`+7.250`-point gain. All three paired gains are positive (`+5.662`, `+5.279`,
and `+10.809` points), so the predictive viability gate passes decisively.

Removing state reduces accuracy to `45.226%`, an `11.271`-point mean loss.
Shuffling state identity reduces accuracy to `53.518%`, a `2.980`-point loss.
All three seeds clear both `+1`-point causal thresholds. State alone reaches
only `6.216%`; the replicated computation remains cooperative. Mean spike rate
is `4.813%`, showing that the state effect persists in a substantially sparser
activity regime than SHD.

Sanity decision: the sample-specific residual-state contribution now
replicates across SHD and SSC. This supports a narrow architectural claim:
pooled direct temporal features and LIF state provide complementary information
in these two event-audio tasks. It does not establish standalone SNN or
state-of-the-art performance. Residual LIF throughput is `15,361` examples/s,
versus `48,003` for Conv1D, so there is no software efficiency advantage.

## 2026-08-10 - Phase 49 SSC matched baseline and efficiency audit generated

Decision: compare validation-selected Conv1D, a stronger two-layer dilated TCN,
and residual LIF on the identical full SSC splits and approximately `133,631`
trainable parameters. Report T4 throughput and explicit operation proxies:
dense multiply-accumulates, state updates, and estimated spike events.

Predictive gate: residual LIF must remain within `2` mean points of the best
matched temporal baseline and no matched baseline may beat it by `2` points on
two seeds. Efficiency claims require measured accelerator throughput or direct
hardware energy; a lower operation proxy alone is insufficient because the
current temporal convolution and Python LIF loop execute densely. This is the
last empirical phase before the final evidence synthesis unless it exposes a
specific correctness defect.

## 2026-08-10 - Phase 49 result: causal mechanism survives, competitiveness does not

Evidence retained at
`gen5/outputs/ssc_efficiency_baselines_cuda_2026-08-10/` from archive SHA-256
`55D3D7F0F68D525628715AC44129D89893C46640C98FE1FC978A7529C6FF1DC5`.

The matched dilated TCN reaches `59.225% +/- 0.541` points, residual LIF reaches
`55.973% +/- 0.018`, and Conv1D reaches `48.948% +/- 1.695`. TCN exceeds
residual LIF by `+3.253` mean points, with all three seeds over the
pre-registered `+2` threshold. The final predictive competitiveness gate
therefore fails. Residual LIF is unusually stable across these three seeds,
but stability does not compensate for the accuracy gap.

Residual LIF uses a `6.527` million dense-MAC proxy per sample versus `7.381`
million for TCN, an `11.569%` reduction, plus `1,856` state updates and about
`95.5` spike events per sample. The implementation nevertheless achieves only
`16,682` examples/s versus `53,080` for TCN: TCN is `3.182x` faster. Dense MAC
counts do not capture Python-loop overhead and are not energy measurements.

Sanity decision: retain the cross-dataset causal mechanism result, reject
matched-baseline superiority and current software-efficiency claims, and stop
architecture tuning for this milestone. The appropriate next step is a final
evidence ledger followed by a new hardware-oriented workstream, not Phase 51
on the same benchmark.

## 2026-08-10 - Phase 50 evidence synthesis generated

Decision: generate a reproducible report directly from committed Phase 44-49
JSON files. The report must enumerate supported, rejected, proxy-only, and
untested claims; preserve exact metrics and source paths; and define the next
generation around compiled event-driven kernels, stronger accuracy scaling,
non-audio replication, and subsequent continual-plasticity reintegration.

This closes the current SHD/SSC empirical milestone. Future work should use a
new roadmap and fresh preregistered gates rather than extending the phase count
for incremental tuning of this architecture.

The synthesis was executed locally and retained at
`gen5/outputs/gen5_evidence_synthesis_2026-08-10/`. Its machine-readable ledger
marks cross-dataset residual-state contribution as supported; standalone LIF,
matched-baseline parity, and current T4 throughput advantage as rejected; lower
dense-MAC arithmetic as proxy-only; and hardware energy efficiency as untested.

## 2026-08-10 - Numbered phases replaced by three decision milestones

Decision: stop extending the project with one numbered phase per diagnostic.
The evidence ledger already identifies three distinct remaining uncertainties,
so work is consolidated into Milestone A (accuracy/architecture), Milestone B
(compiled event-driven hardware efficiency), and Milestone C (non-audio
generalization plus continual learning). Each milestone must contain its own
low-cost screen, validation-only promotion, multi-seed confirmation, causal
controls, and terminal `pass` or `stop` decision. This reduces result-transfer
round trips while preserving evidence discipline.

Milestone A is implemented as a single SSC runner. It screens matched Conv1D,
dilated TCN, residual LIF, hierarchical residual analog, and hierarchical
residual LIF arms with one seed and reduced official subsets. Only the best
conventional arm and causal candidates within `2` validation points, within
`95–105%` of the parameter budget, and with non-degenerate LIF activity are
promoted to full-split, three-seed confirmation. Test labels do not influence
promotion.

The confirmatory gate is deliberately branch-closing. A causal LIF architecture
must stay within `2` mean test points of the best conventional model, lose at
least `1` mean point in both direct-only and batch-shuffled-state ablations,
replicate each loss on at least two of three seeds, and maintain `1–30%` spike
activity. Passing freezes the architecture and opens Milestone B. Failure
closes this architecture branch rather than triggering another tuning phase.
The combined run writes an atomic progress checkpoint after every arm/seed
pair; an identical restart skips completed records, while a configuration
mismatch fails explicitly instead of mixing evidence from different protocols.

## 2026-08-10 - Submitted "Phase 51" archive is duplicate Phase 49 evidence

The uploaded archive
`ssc_efficiency_baselines_cuda-20260810T034930Z-1-001.zip` has outer SHA-256
`98441EBF84DCCEC6DD9D5D3B125C2280331040033B446FCDA5E1E1ACBA97770B`.
Its four enclosed result files are byte-for-byte identical to the committed
Phase 49 archive despite the different outer ZIP hash:

- JSON: `0161CE07EA722E01012257D607C89AD89CDEAB64197A1A78064C2812DEB5EFCD`
- records CSV: `F2D20F099DF0FA25D4A7CB2323F7EF8EE5CEF18E9DFA5E526EE404316648AFAE`
- summary CSV: `927DE7F88AEBC9AA0D368267428777A203FD48DE824E39B9D0476680F85A97C8`
- plot: `A9806422FB57142131D6367A0F1BE3CE17DD0C15B4AB86080F3BCD31CC8589E4`

Decision: do not relabel or count this as Phase 51 and do not generate another
experiment from duplicate evidence. The sanity conclusion remains unchanged:
the residual-LIF state contribution is supported across SHD and SSC, while the
matched TCN retains a `3.253`-point accuracy advantage and `3.182x` throughput
advantage on the dense T4 implementation. The next valid experiment is the
already preregistered Milestone A architecture run, whose output schema begins
with `milestone_a_architecture.json` and `milestone_a_progress.json`.

## 2026-08-10 - Milestone A stops the current Gen-5 architecture branch

Evidence retained at
`gen5/outputs/milestone_a_architecture_cuda_2026-08-10/` from archive SHA-256
`FDF13E7CC2CF3A600389041D0B044E9564BD26E7C4DC736A777A779D304591C3`.

In the preregistered one-seed SSC screen, dilated TCN validation accuracy was
`50.533%`. Residual LIF reached `44.333%` (`-6.200` points), hierarchical
residual analog reached `41.967%` (`-8.567`), and hierarchical residual LIF
reached `37.067%` (`-13.467`). Both LIF arms were within the parameter gate and
had non-degenerate spike rates (`10.547%` and `8.812%`), so their rejection is
an accuracy result rather than dead activity or budget mismatch. Only TCN was
promoted.

Full official SSC confirmation reproduces TCN at `59.170% +/- 0.230` points
over seeds 142–144 and `56,392` examples/s mean T4 throughput. This is close to
the independent Phase 49 result (`59.225% +/- 0.541`) and supports pipeline
consistency. Because no causal arm passed screening, Milestone A correctly
contains no new causal ablation records and returns `status=stop`, no qualified
arms, and `next_milestone=close_architecture_branch`.

Sanity decision: preserve the narrower cross-dataset causal-state mechanism
finding, but reject the current residual and hierarchical variants as a
competitive Gen-5 architecture under the registered protocol. Do not open the
hardware milestone, run a rescue sweep, or claim that this screen disproves
all possible SNNs. The next action is an architecture-branch closeout and an
updated final claim ledger; any successor must be a separately preregistered
generation with a genuinely new hypothesis.

## 2026-08-10 - Final evidence ledger updated with Milestone A

The reproducible synthesis now requires Milestone A alongside Phase 44–49 and
is retained at `gen5/outputs/gen5_architecture_closeout_2026-08-10/`. Two final
claims were added: hierarchical residual scaling closes the SSC gap
(`rejected`) and the current Gen-5 architecture qualifies for hardware
optimization (`rejected`).

Decision: Milestone B and continual-plasticity reintegration are deferred for
this architecture. The completed project claim is narrower but defensible:
sample-specific residual LIF state contributes on two event-audio datasets,
yet the tested implementations are neither the strongest matched predictors
nor efficient in dense T4 execution. This closeout is the gate-selected next
action; generating another numbered rescue phase would contradict the
preregistered stop rule.

## 2026-08-10 - Gen-6 weight-shared residual successor preregistered

Decision: begin a separately named Gen-6 successor rather than reopen Gen-5.
The new hypothesis directly addresses the identified failure mode. The Gen-5
hierarchical models replaced or rebalanced the strong TCN representation and
lost validation accuracy. Gen-6 preserves the complete TCN direct computation
and adds state only as a zero-initialized residual correction to class logits.

The state branch pools the same hidden current, projects it through the
existing classifier weight matrix, and learns only leak, optional threshold,
and 35 per-class gate values. At the default 32-channel width, the LIF
successor adds 99 parameters while retaining the exact TCN width and beginning
with identical logits. An analog arm controls for whether any benefit requires
spiking dynamics.

The protocol is frozen in
`gen5/docs/GEN6_SUCCESSOR_PREREGISTRATION.md`. One-seed reduced-split screening
promotes a residual arm only within `1` validation point of TCN, within the
parameter budget, and with healthy LIF activity. Full confirmation uses all
official SSC splits and seeds 142–144. The LIF successor passes only within `1`
mean test point of TCN, with at least `0.5`-point direct-removal and
state-shuffling losses replicated on two seeds, `1–30%` spike activity, and a
mean absolute correction gate of at least `0.01`.

This is a single terminal experiment with automatic promotion, causal
ablations, and per-arm checkpoint/resume. Failure closes Gen-6 without a rescue
sweep; passing is the only condition that can reopen hardware-efficiency work.

## 2026-08-10 - Gen-6 preserves accuracy but fails causal specificity

Evidence retained at
`gen5/outputs/gen6_successor_cuda_2026-08-10/` from archive SHA-256
`F644395B407CCA4A33B820EE34C62C729CFE1B4EBF24949E5524C6FA74AF83CD`.

The reduced-data screen promoted TCN, shared residual analog, and shared
residual LIF. On complete SSC confirmation over seeds 142–144, TCN reaches
`59.082% ± 0.208` and shared residual LIF reaches `59.016% ± 0.159`, a gap of
only `-0.065` point. The LIF model learns a `0.1898` mean absolute gate and
maintains a healthy `6.020%` spike rate. The zero-initialized, weight-shared
design therefore succeeds at preserving the conventional predictor.

The causal hypothesis fails. Removing LIF state costs `0.386` mean point,
below the registered `0.5` threshold, and only one of three seeds passes.
Shuffling state identity improves accuracy by `0.657` mean point, with zero of
three seeds passing the specificity threshold. The analog arm has an even
larger removal effect (`21.651` points) but an adverse `-4.345`-point
specificity result, supporting the interpretation that the correction is
non-specific rather than inactive. Dense LIF throughput is `15,162` examples/s
versus TCN's `54,966`, or `0.276x`.

Decision: accept predictive parity as supported, reject beneficial
sample-specific correction, and accept the stored `status=stop` decision with
zero qualified arms. Do not run a rescue sweep, reopen hardware optimization,
or present Gen-6 as a competitive SNN result. The gate-selected next direction
is a final reproducible claim ledger and publication-oriented closeout. Any
future generation requires a genuinely new hypothesis and a separate
preregistration approved before training.

## 2026-08-10 - Final Gen-5/Gen-6 evidence closeout generated

The machine-readable claim ledger, claims CSV, and human-readable report were
regenerated from eight retained evidence files and stored in
`gen5/outputs/gen6_research_closeout_2026-08-10/`. The final ledger separates
three conclusions that must not be conflated:

1. sample-specific residual LIF state contributed causally in the earlier SHD
   and SSC experiments;
2. the tested Gen-5 architectures did not match the stronger SSC TCN and did
   not qualify for hardware work;
3. Gen-6 recovered TCN-level accuracy but its correction failed the new
   sample-specific causal gate and likewise did not qualify.

Decision: this completes the registered research branch. The repository keeps
all implementation and evidence as a reproducible negative/qualified result.
There is no active next empirical phase. A future program begins only after a
new mechanism-level hypothesis, dataset, baseline, causal test, and terminal
gate are written and approved before training.

## 2026-08-10 - Gen-7 predictive-state hypothesis approved and implemented

After the Gen-6 terminal closeout, the user explicitly approved a new
mechanism-level direction. This does not reopen or retune the rejected Gen-6
shared residual. Gen-7 tests whether state becomes sample-specific when it is
given an explicit temporal prediction objective and may affect output only
through a sample-conditioned direct/state interaction.

The direct two-layer dilated TCN and classifier remain intact. Every residual
arm begins with an exactly zero correction. The state branch uses heterogeneous
initial leak constants (`0.50`, `0.75`, `0.90`, `0.97`) across channels. Early
state predicts later encoder activity four bins ahead with a symmetric in-batch
contrastive objective at temperature `0.10`; the fixed auxiliary loss weight
is `0.20`. The decisive comparison holds architecture and optimizer constant
while changing future target identity: paired versus one-step batch-shuffled.

Registered arms are TCN, LIF without prediction, paired predictive analog,
shuffled-target predictive LIF, and paired-target predictive LIF. If the paired
candidate reaches confirmation, the no-predictive and shuffled-target controls
are forced into confirmation even if their screen accuracies trail. This avoids
declaring a mechanism without its necessary controls.

The terminal candidate must preserve TCN accuracy, pass direct-removal,
state-shuffling, and time-reversal losses of at least `0.5` point on two of
three seeds, achieve paired-minus-shuffled future cosine of at least `0.02`,
beat shuffled-target training by at least `0.01` alignment, maintain `1–30%`
spikes, and learn at least `0.01` mean absolute sample-conditioned gate
activity. The complete protocol is frozen in
`gen5/docs/GEN7_PREDICTIVE_STATE_PREREGISTRATION.md`. A failure closes this
hypothesis without a loss-weight, horizon, threshold, or architecture sweep.

## 2026-08-10 - Gen-7 learns predictive state but fails identity-specific use

Evidence retained at
`gen5/outputs/gen7_predictive_state_cuda_2026-08-10/` from archive SHA-256
`5AF3B42A569EADEB5CA56E7E33005334E73D93820B298BA77E4850A62DFB67F0`.

All registered arms passed screening. On complete SSC confirmation over seeds
142–144, paired predictive LIF reaches `58.807% ± 1.093`, exceeding the matched
TCN at `58.390% ± 1.848` by `+0.417` point. Removing its state costs `1.009`
mean points with two seeds clearing the gate. Correctly paired future training
produces a `0.2928` alignment margin on all seeds versus `-0.0017` for
shuffled-target training, a decisive `+0.2945` difference. Spike activity
(`7.204%`) and the sample-conditioned gate (`0.2350`) are healthy.

The terminal causal claim nevertheless fails. Batch-shuffling state improves
candidate accuracy by `1.022` mean points and no seed passes specificity.
Time reversal costs only `0.165` point and likewise passes on no seed. This
separates representation from use: the auxiliary objective encodes paired
future identity, but the pooled additive correction does not use that identity
beneficially or depend strongly on temporal order. Dense throughput is
`16,004` examples/s versus TCN's `53,556` (`0.299x`).

Decision: accept predictive alignment and modest mean accuracy improvement as
supported; reject beneficial identity-specific and order-specific state use;
accept the stored `status=stop` with zero qualified arms. Do not tune the
registered loss weight, horizon, thresholds, or decoder. The gate-selected
next direction is a final evidence-ledger update and research closeout. Any
temporal-binding successor is a new generation requiring explicit approval
and a preregistration before implementation.

## 2026-08-10 - Gen-7 final evidence ledger generated

The reproducible synthesis now requires nine retained evidence files through
Gen-7 and is stored in `gen5/outputs/gen7_research_closeout_2026-08-10/`.
Its machine-readable ledger records paired future alignment and TCN-level
accuracy as supported, identity/order-specific state use as rejected, and
hardware qualification as rejected.

Sanity decision: the project has evidence that spiking state can be causally
useful in earlier residual experiments and that an explicit predictive
objective can align state strongly. It still lacks a competitive architecture
whose output benefit depends on the correct sample's temporally ordered state.
This is now the central unresolved research question. No active empirical
phase follows automatically; a time-local binding/fusion successor requires a
new preregistration and explicit user approval.

## 2026-08-10 - Gen-8 time-local predictive binding approved and implemented

The user explicitly approved the next experiment after the Gen-7 terminal
closeout. This authorizes one new mechanism-level test, not a retuning sweep of
Gen-7. The decision addresses the exact unresolved observation: Gen-7 state
contains paired future information, but pooling direct and state traces before
their interaction allows the output correction to ignore sample identity and
temporal order.

Gen-8 moves both operations to aligned time. Its candidate contrasts
`state[t]` with the same sample's encoder activity at `t + 4` independently at
each timestep using the adjacent samples in both batch directions as fixed
negative identities. This keeps the objective linear in batch size rather than
materializing a quadratic time×batch×batch similarity tensor. It produces the residual class trace from
`direct[t] * state[t]` before temporal averaging. The binding projection is
zero-initialized, preserving the matched TCN logits exactly at initialization.
Its trainable parameter count is deliberately identical to the corresponding
Gen-7 residual arm.

Registered controls are the matched dilated TCN, the pooled Gen-7 predictive
LIF, an analog time-local binder, a time-local LIF trained on batch-shuffled
future targets, and the paired time-local LIF candidate. Screening uses seed
145 and reduced official SSC splits; confirmation uses complete splits and
seeds 145–147. Candidate promotion automatically forces all mechanistic
controls into confirmation.

The frozen terminal gate requires accuracy within one TCN point; replicated
0.5-point losses under state removal, sample shuffling, and time reversal;
local future alignment and paired-over-shuffled alignment; healthy spikes and
binding activity; and at least 0.5-point identity and order improvements over
the pooled Gen-7 reference. Passing opens only a separate runtime/external
replication preregistration. Failure closes temporal binding without a horizon,
temperature, loss-weight, threshold, or gate sweep. The complete protocol is
recorded in `gen5/docs/GEN8_TEMPORAL_BINDING_PREREGISTRATION.md`.

## 2026-08-10 - Gen-8 local LIF fails screening; analog binding adds order but not identity

Evidence is retained at
`gen5/outputs/gen8_temporal_binding_cuda_2026-08-10/` from archive SHA-256
`5D3087148BF735FE921E6895B06318B979F3371454095B57170F2B6286493E79`.

The paired time-local LIF candidate screened at `7.267%` validation accuracy,
`30.833` points below the matched TCN, with a `50.656%` spike rate. Its
shuffled-target control screened at `10.133%` with `41.091%` spikes. Both fail
the registered accuracy and maximum-activity gates by wide margins, so neither
was promoted. Their parameter ratios were matched and their binding
corrections were active, making this a destructive-dynamics result rather than
an inactive-path or capacity artifact.

Three arms entered full confirmation. Pooled predictive LIF reaches
`60.684% ± 0.283`, `+1.536` points over TCN, but state removal costs only
`0.221` point, shuffling improves accuracy by `2.069` points, and reversal
costs `0.203` point. This independently repeats the Gen-7 separation between
predictive alignment (`0.3094`) and beneficial causal use.

The analog time-local binder reaches `58.692% ± 2.176`, only `0.456` point
below TCN. Reversal costs `0.561` point and passes on 2/3 seeds, supporting a
limited conclusion that pre-pooling fusion introduces temporal-order
sensitivity. Shuffling state costs only `0.118` point and passes on no seed;
the correct sample identity still is not causally useful. Its state-removal
cost (`0.502` point) replicates on only one seed. TCN throughput is `53,167`
examples/s versus `21,207` for analog binding and `14,071` for pooled LIF.

Decision: accept partial analog temporal-order sensitivity; reject stable
time-local LIF, identity-specific binding, architecture qualification, and
hardware-efficiency claims. Accept the stored `status=stop` with zero
qualified arms. Do not retune the horizon, temperature, loss weight, spike
threshold, or gate. The evidence-selected next phase is the final ten-source
claim ledger and publication closeout, not Gen-9.

## 2026-08-10 - Gen-8 final evidence ledger generated

The reproducible synthesis now requires ten retained evidence files through
Gen-8 and is stored in `gen5/outputs/gen8_research_closeout_2026-08-10/`. It
records predictive representation and partial analog order sensitivity as
supported; stable identity-specific spiking binding and hardware qualification
as rejected.

Sanity decision: the architecture-development branch has exhausted three
separately preregistered successors without producing a parameter-matched,
stable spiking model whose output benefit requires the correct sample's
ordered state. The defensible contribution is a set of causal residual-state,
predictive-representation, and partial temporal-binding findings with explicit
negative selection results. A later program must begin from a new task-level
hypothesis and independent preregistration rather than another decoder rescue.

## 2026-08-10 - Gen-9 continual-adaptation program approved and implemented

The user approved a new program aligned with the original AMMC goal: adaptation
during an organism's lifetime. This does not reopen Gen-6–Gen-8 or modify their
terminal conclusions. Gen-9 begins with a representation-level gate before
adding STW/LTW, neuromodulation, sleep replay, or structural plasticity.

The fixed task shift permanently zeros 35% of SSC's 700 input channels using
mask seed 909 while preserving labels and timing. Official training examples
train the undamaged source model; official validation examples form the
damaged adaptation stream; official test examples are evaluated in undamaged
and damaged forms only. This separation prevents test-driven checkpoint or
budget selection.

Registered source models are the matched dilated TCN and Gen-7 pooled
predictive LIF. Screening uses seed 148 and reduced official splits. The LIF
source promotes only within one TCN validation point, 95–105% of the parameter
budget, and 1–30% spikes. Confirmation uses full splits and seeds 148–150.

Registered strategies are static TCN, TCN readout adaptation, full TCN
fine-tuning, static predictive LIF, and predictive-LIF readout adaptation.
Adaptation observes new, non-replayed blocks at cumulative budgets 0, 64, 256,
1,024, and 4,096, using three epochs per new block at learning rate 0.001.
Readout arms update identically shaped classifiers; the predictive state and
gate remain frozen, isolating representation adaptability.

The terminal LIF gate requires a non-trivial five-point shift, source parity,
a replicated two-point self-improvement, a replicated one-point adaptation-AUC
advantage over TCN readout, final damaged accuracy within one TCN point,
forgetting no more than 0.5 point worse, and healthy spikes. Passing opens only
a separately preregistered STW/LTW memory milestone. Failure closes this
continual-adaptation representation and blocks automatic memory, replay,
plasticity, or shift-severity rescue experiments. The complete frozen protocol
is `gen5/docs/GEN9_CONTINUAL_ADAPTATION_PREREGISTRATION.md`.

## 2026-08-10 - Gen-9 checkpoint schema hotfix

The first Colab execution completed its initial screen arm but failed before
writing the checkpoint because Gen-9 passed `promoted_source_arms` and
`adaptation_records` into the older Milestone-A writer, whose fixed schema
accepts only `promoted_arms` and `confirmation_records`. This is an
orchestration defect, not an experimental result; no terminal metric or gate
was observed.

Decision: Gen-9 now owns an atomic progress writer with its explicit source and
adaptation fields while retaining the shared signature-validated loader. A
round-trip regression test verifies the exact schema. The scientific protocol,
seeds, masks, budgets, models, and gates are unchanged, so rerunning the same
Colab command is valid.

## 2026-08-10 - Gen-9 continual adaptation stops at source competence

The retained archive
`gen9_continual_adaptation_cuda-20260810T110831Z-1-001.zip` has SHA-256
`920e697a70885fd91a04ac18c73ec1348d57f504b17e9080a2e7d034de5e4caf`.
Its extracted evidence is stored in
`gen5/outputs/gen9_continual_adaptation_cuda_2026-08-10/`.

The source screen promoted only `dilated_tcn`. Predictive LIF reached 25.100%
validation accuracy versus 31.567% for TCN, missing the one-point parity gate
by 6.467 points. Its 8.964% spike rate and 100.436% parameter-budget ratio were
healthy, so this is a source-representation failure rather than silent neurons
or a parameter mismatch. Predictive-LIF adaptation was correctly not run.

The confirmed TCN controls validate the task. Across seeds 148–150, sensor
damage reduced static accuracy by 9.364 points. Frozen-readout adaptation
recovered 5.601 points on all three seeds, finishing at 53.927% damaged
accuracy with 2.831 points of source forgetting. Full fine-tuning recovered
8.462 points, finished at 56.787%, and forgot only 1.138 points. It exceeded
readout adaptation by 2.860 final points and 2.270 AUC points while forgetting
1.693 fewer points.

Decision: accept the sensor-damage benchmark and conventional continual-
adaptation baselines. Reject the claim that this predictive-LIF representation
is source-competent or that AMMC continual adaptation has been demonstrated.
Accept the stored `status=stop` with zero qualified arms. STW/LTW,
neuromodulation, sleep replay, structural plasticity, and an automatic damage-
severity sweep remain closed. The next phase is the eleven-source evidence
closeout; any later continual-learning program requires a separately
preregistered representation that first passes source competence.

## 2026-08-10 - Gen-10 masked-sensor representation reset approved

The user authorized the next step after the Gen-9 closeout. The new hypothesis
is not a pooled-predictive-LIF rescue: residual state receives a parameter-free
objective that aligns its masked-sensor representation to the detached clean
direct representation. This directly targets the failure mode exposed by
Gen-9 while preserving a conventional direct path for source competence.

The frozen matrix contains ordinary TCN, sensor-dropout TCN, masked residual
analog, and masked residual LIF. Training masks 20% of sensors independently;
evaluation reuses Gen-9's unseen fixed 35% mask at seed 909. The dropout TCN
separates augmentation benefit from spiking-state benefit, and the analog arm
separates temporal state from spiking dynamics.

Screening requires clean validation within one point and damaged validation
within two points of the best conventional arm, matched parameter budget, and
healthy LIF activity. Confirmation uses seeds 151–153 and complete SSC splits.
The LIF terminal gate requires clean and damaged parity, no more than 0.5
point damage loss, replicated 0.5-point state-removal and state-shuffling
costs, a damage drop no more than 0.5 point worse than the best conventional
control, and 1–30% spikes.

Decision: Gen-10 can open only a separately preregistered Gen-11 adaptation
comparison. STW/LTW, replay, neuromodulation, and structural plasticity remain
closed even if Gen-10 passes. A stop forbids automatic mask-rate, alignment-
weight, threshold, leak, or gate sweeps. The protocol is
`gen5/docs/GEN10_ROBUST_REPRESENTATION_PREREGISTRATION.md` and the runner is
`gen5/examples/gen10_robust_representation.py`.

## 2026-08-10 - Gen-10 residual-state representation fails; sensor dropout succeeds

The archive `gen10_robust_representation_cuda-20260810T114444Z-1-001.zip`
has SHA-256
`174da16e3dc9d685dc49c560a28ef197fa91cfc718bad376e76e33183b5d099a`.
Its extracted evidence is retained in
`gen5/outputs/gen10_robust_representation_cuda_2026-08-10/`.

Only ordinary and sensor-dropout TCN promoted. On the screen, dropout TCN
reached 47.400% clean and 41.033% damaged validation. Residual analog trailed
by 5.500 and 3.200 points; residual LIF trailed by 9.200 and 6.733 points.
Residual LIF was parameter-matched at 98.832% of target and spiked at 11.538%,
so the result rejects source competence rather than activity or capacity.

Confirmation establishes sensor dropout as a strong control: it improved
clean TCN accuracy by 2.684 points and damaged accuracy by 8.763 points, while
reducing the fixed-mask damage drop from 9.930 to 3.851 points. No state arm
confirmed, so state-removal and identity-shuffling claims are not available.

Decision: accept sensor dropout robustness; reject Gen-10 residual analog/LIF
source competence and accept the stored `status=stop`. Do not sweep masking,
alignment weight, leak, threshold, or gate. The user authorized a genuinely
new functional-separation hypothesis: freeze the proven dropout-TCN sensory
backbone and test a bounded downstream spiking adapter during damage
adaptation. This preserves source competence by construction while asking
whether a plastic spiking subsystem adds causal adaptation value.

## 2026-08-10 - Gen-11 functional-separation adapter approved and implemented

Gen-11 freezes the evidence-supported 20%-sensor-dropout TCN backbone and adds
a downstream classifier-sized state adapter. The adapter receives the
backbone's time-resolved hidden trace and produces bounded correction logits.
Its zero-initialized correction gate guarantees identical source behavior at
zero adaptation samples. Analog and LIF dynamics share the same interface and
parameter scale.

Controls are static dropout TCN, classifier-only readout adaptation, and full
backbone fine-tuning. Adaptation uses the same official validation stream,
fixed 35% damage mask, cumulative non-replayed budgets, and official-test-only
measurement as Gen-9. Seeds 154–156 are new. Final adapter evaluation removes
the correction and batch-shuffles state identity without retraining.

Decision: a pass requires a valid shift, replicated two-point LIF adaptation,
readout-level AUC/final accuracy, bounded forgetting, replicated 0.5-point
state-removal and state-shuffling costs, and healthy spikes. Only that result
can open a separately preregistered STW/LTW milestone. Failure closes the
adapter without width, gate, leak, threshold, learning-rate, or mask sweeps.
The frozen protocol is `gen5/docs/GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md`.

## 2026-08-10 - Gen-11 adapters fail adaptation and sample-specificity gates

The archive `gen11_plastic_adapter_cuda-20260810T122441Z-1-001.zip` has
SHA-256 `150d6d350ed14c55a3eed44e4980264e5d3f460b6404ad35653017870a488a24`.
Its extracted evidence is retained in
`gen5/outputs/gen11_plastic_adapter_cuda_2026-08-10/`.

The frozen sensor-dropout TCN lost 3.910 points under the fixed sensor failure.
Full fine-tuning recovered 3.295 points and readout adaptation recovered 2.330.
Analog and LIF adapters recovered only 1.353 and 0.783 points. Neither reached
the registered two-point gate or matched readout adaptation.

The LIF adapter remained healthy at 16.782% spikes and had effectively no
source forgetting. Removing its state erased the complete 0.783-point gain on
all seeds, but shuffling sample identity cost only 0.011 point with 0/3 seeds
passing. Analog state likewise showed a 1.353-point removal cost but only a
0.018-point shuffle cost. The correction is causal in aggregate but not
sample-specific; it behaves like a learned class bias.

Decision: accept the conventional adaptation and source-retention controls.
Reject beneficial sample-specific analog/LIF adapter state and accept the
stored `status=stop`. Do not sweep the Gen-11 adapter. Synaptic STW/LTW,
replay, neuromodulation, structural plasticity, and hardware claims remain
closed. The next experiment must introduce an explicitly sample-indexed
mechanism under a separate preregistration.

## 2026-08-10 - Gen-12 associative fast-memory hypothesis approved and implemented

Gen-12 tests stable-sensory/fast-memory functional separation rather than a
parametric adapter rescue. The source-competent sensor-dropout TCN remains
frozen. A dense prototype control and a sparse rank-order spiking prototype
memory accumulate labeled damaged-stream associations by sums and counts,
without backpropagating into the backbone.

The registered controls are static TCN, readout adaptation, full fine-tuning,
dense prototype memory, and spiking prototype memory. Seeds 157–159, source
training, fixed mask, splits, and cumulative budgets match Gen-11. Memory is
explicitly gated by the known damaged context; this prevents source forgetting
but does not demonstrate autonomous context inference.

Decision: the spiking memory must match readout AUC/final accuracy, gain two
points on 2/3 seeds, preserve source performance, maintain 5–35% code density,
and lose at least 0.5 point under both memory removal and class-association
shuffling on 2/3 seeds. A pass opens only context-free memory and consolidation
testing. A stop closes prototype memory without hyperparameter or damage
sweeps. Synaptic STW/LTW remains closed until a later, separately registered
gate. The frozen protocol is
`gen5/docs/GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md`.

## 2026-08-10 - Gen-12 prototype memory fails fast-adaptation gate

The archive `gen12_associative_memory_cuda-20260810T133927Z-1-001.zip` has
SHA-256 `7121cbd458b4f1a10736d6c8ac3e1b0a2d02f2b97c25d85897c32eda2f109717`.
Its extracted evidence is retained in
`gen5/outputs/gen12_associative_memory_cuda_2026-08-10/`.

The fixed damage shift was 5.158 points. Readout adaptation recovered 3.564
points and full fine-tuning recovered 4.767, with every seed improving. Dense
and spiking class prototypes recovered only 0.250 and 0.278 point; neither
passed on any seed.

The spiking memory held exactly 20% event density, used 16,800 active cells,
and preserved source accuracy through the registered context gate. Removing it
cost only 0.278 point and shuffling class associations cost 0.417, with 0/3
seeds crossing either 0.5-point causal threshold. Dense memory behaved
similarly. The result is therefore not explained by silent codes, missing
capacity, or catastrophic forgetting. Class-average retrieval loses the
task-specific structure that output-weight adaptation exploits.

Decision: accept the source, shift, conventional adaptation, context-gating,
and activity controls. Reject useful associative-prototype adaptation and
accept the stored `status=stop`. Do not sweep memory mix, temperature, spike
density, number of prototypes, damage, or budgets. Synaptic STW/LTW, replay,
neuromodulation, structural plasticity, and hardware claims remain closed.

## 2026-08-10 - Gen-13 local three-factor plasticity approved and implemented

Gen-13 consolidates the next decision into one mechanism-selection experiment
rather than separate rescue phases. The robust sensor-dropout TCN remains
frozen. Manual analog and sparse-spiking readout updates are compared with
static, autograd-readout, and full-fine-tuning controls.

The local rule is an explicit outer product of normalized presynaptic trace
and postsynaptic multiclass error. It uses no autograd and cannot modify the
sensory backbone. It remains supervised and is not presented as reward-only
or fully biologically plausible. Seeds 160–162, source protocol, fixed damage,
budgets, and three passes over each new block are frozen. Local learning rate
is 0.50 to compensate for normalized, minibatch-averaged outer products;
decay is 0.0001 and the spiking trace has 20% density.

Decision: the spiking local rule must gain two points on 2/3 seeds, match
autograd readout AUC and final accuracy within one point, preserve source
accuracy, and lose 0.5 point on 2/3 seeds under both fast-weight removal and
output-class shuffling. A pass opens only a separately preregistered STW/LTW
consolidation experiment. A stop closes local readout plasticity without
learning-rate, density, epoch, normalization, damage, or budget sweeps. The
frozen protocol is `gen5/docs/GEN13_LOCAL_PLASTICITY_PREREGISTRATION.md`.
