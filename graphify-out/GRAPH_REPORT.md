# Graph Report - A-SNN  (2026-10-13)

## Corpus Check
- 409 files · ~325,946 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4130 nodes · 9713 edges · 256 communities (245 shown, 11 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 293 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `68de89af`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- resolve_device
- evaluation.py
- EvolutionTelemetryLogger
- TensorEvolver
- trainable_sequential_mnist.py
- trainable_temporal_mnist.py
- FrozenProbeResult
- sprint15_frozen_readout_adapter_robustness.py
- sync
- shd_calibrated_baselines.py
- AMMC Research Log
- EventMNISTConfig
- shd_temporal_pyramid.py
- ammc_gen5/gen16_local_score_credit.py
- ammc_gen5/gen6_successor.py
- ammc_gen5/gen9_continual_adaptation.py
- ammc_gen5/nmnist_accuracy_benchmark.py
- ammc_gen5/gen30_dendritic_predictive_credit.py
- Core findings
- TensorEnvironment2D
- ammc_gen5/gen17_sparse_spiking_credit.py
- ammc_gen5/gen25_event_driven_sparse_audit.py
- shd_capacity.py
- tensor_environment.py
- trainable_delays_mnist.py
- ammc_gen5/gen15_reward_baseline.py
- synthesize_gen5_evidence
- ammc_gen5/gen28_triton_event_kernel.py
- ammc_gen5/gen10_robust_representation.py
- sprint16_frozen_embodied_adapter.py
- run_milestone_a
- ammc_gen5/gen18_local_credit_replication.py
- ammc_gen5/gen13_local_plasticity.py
- ammc_gen5/gen22_dual_memory_replication.py
- shd_analog_topology.py
- ChampionExporter
- ammc_gen5/gen11_plastic_adapter.py
- ammc_gen5/gen23_boundary_consolidation.py
- benchmark_throughput.py
- ammc_gen5/gen14_reward_eligibility.py
- ammc_gen5/gen19_nmnist_state_replication.py
- ammc_gen5/gen24_compiled_residual_state.py
- README.md
- shd_representation.py
- shd_matched_baselines.py
- ammc_gen5/gen27_trained_threshold_robustness.py
- PlasticityAblationRunner
- ammc_gen5/gen12_associative_memory.py
- __init__.py
- DynamicSparseLinear
- ammc_gen5/gen7_predictive_state.py
- Gen20Config
- SHDConfig
- shd_sparse_mechanisms.py
- shd_temporal_controls.py
- shd_sparse_width.py
- ammc_gen5/gen20_spiking_spatiotemporal.py
- device_kind
- shd_initialization_robustness.py
- DenseLIFPolicy
- ammc_gen5/gen21_matched_causal_mechanisms.py
- A-SNN / AMMC
- PlasticStateAdapter
- utility_gated_structural_mnist.py
- ConvPLIFClassifier
- 4. Development milestones
- DualTensorManager
- Phase 11 uploaded outputs review
- ammc_gen5/gen29_program_closure.py
- ammc_gen5/gen8_temporal_binding.py
- verify_phase11_outputs.py
- LTWSTWMemory
- verify_graph
- AMMC Gen-5
- MultiTimescaleResidualPLIF
- Gen21MechanismReadout
- PredictiveStateTCNClassifier
- SharedResidualStateTCNClassifier
- top_fraction_spike_code
- Gen-6 successor preregistration
- FrozenEventReservoir
- FrozenReadoutAdapterRunner
- Sprint16EmbodiedAdapterContractTest
- Sprint17ActionControlsContractTest
- Frozen diversified task benchmarks
- Project decisions
- FrozenTaskRunner
- NmnistAccuracyEvidenceContractTest
- shd_spiking_temporal_conv.py
- sprint15_frozen_readout_adapter_generalization.py
- shd_validation_checkpoint.py
- Gen-7 predictive-state preregistration
- Gen-8 time-local predictive-binding preregistration
- RetentionAblationRunner
- available_frozen_tasks
- FrozenReadoutAdapterConfig
- Harder bot-world benchmarks
- Phase 26: Targeted Synaptogenesis on Sequential MNIST
- Phase 25: Fixed-Topology LTW Training on Sequential MNIST
- TemporalDilatedTCNClassifier
- Gen-20 spiking spatial-temporal translation preregistration
- Phase 24: Streaming Row-Sequential MNIST
- LinearRewardPolicy
- Phase 23: Causal Recurrence Ablation
- Gen-5 TPU/XLA migration note
- load_ssc_tensors
- Gen-9 continual-adaptation preregistration
- Approved sequence
- Full-resolution N-MNIST accuracy benchmark preregistration
- Primary evidence track roadmap
- Browser seeded plasticity replay monitor - champion-001
- CUDA saturated-topology compiled hotpath throughput
- Graphify knowledge workflow
- event_mnist.py
- Phase 11 Colab Runbook
- AMMC Gen-5 Colab Run Analysis — 500 Generations
- Browser Champion Bridge Replay Monitor — 2026-06-25
- Phase 20 temporal-state MNIST: analysis and goal sanity check
- CUDA saturated-topology throughput with `torch.compile`
- Phase 21 fixed-topology LTW training: analysis and goal sanity check
- Contributing notes
- .step
- StreamingMNISTReservoir
- ResidualTemporalConvStateClassifier
- Frozen embodied readout benchmark
- Gen-10 masked-sensor representation reset preregistration
- Gen-13 three-factor local-plasticity preregistration
- Gen-16 local score-function credit preregistration
- Gen-17 sparse-spiking local-credit preregistration
- Gen-19 N-MNIST residual-state replication preregistration
- Gen-21 matched causal mechanism benchmark preregistration
- Gen-30 dendritic predictive-credit analysis
- Gen-30 dendritic predictive-credit preregistration
- Program sanity check after Gen-20
- Phase 21: Fixed-Topology LTW Training
- Browser Champion Replay Monitor — 2026-06-25
- Browser seeded replay monitor — champion-001
- Phase 19 event representation decomposition: analysis
- Sprint 16 frozen embodied adapter analysis
- AMMC Gen-5 through Gen-10 evidence report
- AMMC Gen-5 through Gen-11 evidence report
- AMMC Gen-5 through Gen-12 evidence report
- AMMC Gen-5 through Gen-13 evidence report
- AMMC Gen-5 through Gen-14 evidence report
- AMMC Gen-5 through Gen-15 evidence report
- AMMC Gen-5 through Gen-16 evidence report
- AMMC Gen-5 through Gen-17 evidence report
- AMMC Gen-5 through Gen-18 evidence report
- AMMC Gen-5 through Gen-20 evidence report
- Gen-20 spiking spatial-temporal translation analysis
- AMMC Gen-5 evidence report
- AMMC Gen-5 through Gen-19 evidence report
- AMMC Gen-5/Gen-6 evidence report
- AMMC Gen-5/Gen-6/Gen-7 evidence report
- AMMC Gen-5/Gen-6/Gen-7/Gen-8 evidence report
- AMMC Gen-5 through Gen-9 evidence report
- Phase 22 LTW optimization diagnostic: analysis and goal sanity check
- CUDA champion-topology compiled hotpath throughput
- CUDA champion-topology compiled hotpath throughput, fingerprinted
- CUDA champion-topology eager hotpath throughput
- CUDA champion-topology eager hotpath throughput, fingerprinted
- CUDA saturated-topology throughput analysis
- CUDA saturated-topology eager hotpath throughput
- Repository structure
- AdaptiveSequentialClassifier
- Phase 18: frozen event-coded MNIST
- Phase 19: event representation decomposition
- Gen-11 frozen sensory backbone plus plastic state adapter preregistration
- Gen-12 frozen-backbone associative-memory preregistration
- Gen-14 reward-modulated embodied eligibility preregistration
- Gen-18 held-out local-credit replication preregistration
- Gen-19 N-MNIST residual-state replication analysis
- Gen-22 dual-memory sequential-shift replication preregistration
- Phase 22: LTW Optimization Diagnostic
- N-MNIST accuracy benchmark analysis
- Phase 38: parameter-matched SHD baselines
- Phase 37: SHD temporal-control decomposition
- Phase 36: parameter-matched SHD temporal pyramid
- Phase 20: temporal-state preservation on MNIST
- AMMC-SNN Behavior Observation
- AMMC-SNN Behavior Observation — Corrected Parser
- CUDA delayed-reward delay-1 screen
- CUDA delayed-reward delay-2 full run, console-only archive
- CUDA delayed-reward delay-2 screen
- Phase 17 embodied action controls: analysis
- Phase 18 frozen event-coded MNIST: analysis
- Frozen Readout Adapter Augmented Robustness Result
- Frozen Readout Adapter Held-Out-Seed Generalization Result
- Frozen Readout Adapter Robustness Result
- Frozen Readout Adapter Sweep CUDA Result
- Gen-10 robust-representation analysis
- Gen-29 causal evidence closure
- Gen-9 continual-adaptation analysis
- Milestone A: architecture screen and confirmation
- CUDA neuron-scaling evaluation
- CUDA delay-2 neuron scaling, full artifact archive
- CUDA delay-2 neuron scaling, console-only archive
- Gen-5 outputs
- Phase 31 screen analysis: SHD temporal transfer plumbing
- Phase 34 analysis: SHD hidden-capacity scaling
- Phase 35 analysis: SHD capacity-delay interaction
- Phase 33 analysis: SHD representation diagnostic
- Phase 37 analysis: SHD temporal-control decomposition
- Phase 36 analysis: parameter-matched SHD temporal pyramid
- CUDA sparse-efficiency component screen
- CUDA sparse-efficiency screen
- Phase 28: Adaptive Neurons on Sequential MNIST
- Phase 29: Executable Axonal Delays on Sequential MNIST
- Embodied action controls
- Gen-11 plastic state-adapter analysis
- Gen-12 associative-memory analysis
- Gen-15 matched reward-baseline analysis
- Gen-15 matched reward-baseline preregistration
- Gen-16 local score-credit analysis
- Gen-17 sparse-spiking local-credit analysis
- Gen-18 held-out local-credit replication analysis
- Gen-21 matched causal mechanism benchmark analysis
- Program sanity check after Gen-28
- Phase 30: Trainable Delay Assignment
- Phase 27: Utility-Gated Structural Plasticity
- Phase 28 analysis: adaptive neurons on sequential MNIST
- CUDA delayed-reward delay-3 screen
- Phase 29 analysis: executable axonal delays
- CUDA frozen diversified tasks, summary archive
- Frozen Readout Adapter CUDA Result
- CUDA frozen representation probe, summary archive
- CUDA harder-world benchmark
- Phase 40 analysis: SHD analog dynamics and topology
- Phase 31 analysis: full SHD temporal transfer
- Phase 42 analysis: SHD initialization robustness
- Phase 38 analysis: matched SHD baselines
- Phase 47: residual-state causal contribution on SHD
- Phase 39 analysis: sparse SHD mechanisms
- Phase 41 analysis: fixed-budget sparse width scaling
- Phase 43 analysis: validation-selected SHD checkpointing
- CUDA sparse-efficiency finalist run
- CUDA sparse-efficiency finalist screen
- Phase 49: SSC matched baseline and efficiency audit
- Phase 48: residual-LIF replication on SSC
- Phase 30 analysis: trainable recurrent delays
- Phase 27 analysis: utility-gated structural MNIST
- AGENTS.md
- Gen-13 local-plasticity analysis
- AMMC research evidence freeze through Gen-20
- Phase 44: calibrated SHD temporal baselines
- Phase 45: learned spiking temporal convolution
- Phase 46: SHD temporal-state placement
- Assets
- Current implementation state
- PROGRAM_SANITY_CHECK_AFTER_GEN22.md
- PROGRAM_SANITY_CHECK_AFTER_GEN24.md
- PROGRAM_SANITY_CHECK_AFTER_GEN25.md
- gen5_architecture_closeout_2026-08-10/analysis.md
- legacy_first_run_upload_2026-06-25/README.md
- run_harness
- sprint26_structural_sequential_mnist.py

## God Nodes (most connected - your core abstractions)
1. `SHDConfig` - 217 edges
2. `resolve_device()` - 145 edges
3. `seed_everything()` - 134 edges
4. `EventMNISTConfig` - 125 edges
5. `AMMC Research Log` - 125 edges
6. `sync()` - 117 edges
7. `device_kind()` - 116 edges
8. `mark_step()` - 86 edges
9. `TensorEvolver` - 64 edges
10. `Core findings` - 64 edges

## Surprising Connections (you probably didn't know these)
- `AdaptiveSequentialClassifier` --uses--> `EventMNISTConfig`  [INFERRED]
  gen5/ammc_gen5/adaptive_sequential_mnist.py → gen5/ammc_gen5/event_mnist.py
- `AdaptiveSequentialClassifier` --uses--> `SurrogateSpike`  [INFERRED]
  gen5/ammc_gen5/adaptive_sequential_mnist.py → gen5/ammc_gen5/trainable_temporal_mnist.py
- `AdaptiveSequentialMNISTResult` --uses--> `EventMNISTConfig`  [INFERRED]
  gen5/ammc_gen5/adaptive_sequential_mnist.py → gen5/ammc_gen5/event_mnist.py
- `run_adaptive_sequential_mnist()` --uses--> `EventMNISTConfig`  [INFERRED]
  gen5/ammc_gen5/adaptive_sequential_mnist.py → gen5/ammc_gen5/event_mnist.py
- `DelayedSequentialClassifier` --uses--> `SurrogateSpike`  [INFERRED]
  gen5/ammc_gen5/delayed_sequential_mnist.py → gen5/ammc_gen5/trainable_temporal_mnist.py

## Import Cycles
- None detected.

## Communities (256 total, 11 thin omitted)

### Community 0 - "resolve_device"
Cohesion: 0.06
Nodes (54): _run_stage(), _build_model(), MilestoneAArm, MilestoneAResult, Unified Milestone A screening and confirmation for SSC architectures., _run_stage(), Any, Seed global RNGs for CPU/CUDA/XLA best-effort reproducibility. (+46 more)

### Community 1 - "evaluation.py"
Cohesion: 0.04
Nodes (77): AblationGenerationRecord, AblationGroupConfig, AblationSummaryRecord, aggregate_trial_records(), AggregateGenerationRecord, default_ablation_groups(), default_neuron_scale_points(), _float() (+69 more)

### Community 2 - "EvolutionTelemetryLogger"
Cohesion: 0.12
Nodes (12): EvolutionTelemetryLogger, EvolutionTelemetryRecord, Any, Path, Telemetry and plotting utilities for headless AMMC evolution., One epoch-level telemetry point., Record and plot evolution curves for a headless Gen-5 run. The logger is…, Append one epoch record from an `EvolvingHeadlessAMMCLoop` report. (+4 more)

### Community 3 - "TensorEvolver"
Cohesion: 0.06
Nodes (31): EdgeRecord, _MissingModule, Dynamic sparse electrical graph prototype. This module is the Sprint 1 Python…, Serializable description of one directed sparse edge., _MissingModule, Tensorized genetic evolution for AMMC Gen-5. Sprint 6/7 target: perform…, Apply each organism's sparse genome to its own neural state. Args:…, Broadcast one seed genome to every organism. (+23 more)

### Community 4 - "trainable_sequential_mnist.py"
Cohesion: 0.08
Nodes (54): AdaptiveSequentialArm, AdaptiveSequentialMNISTResult, _attach_lif_comparisons(), _measure_adaptive(), plot_adaptive_sequential_mnist(), Path, Phase 28 adaptive-neuron ablation on row-sequential MNIST. Phase 27 showed that…, run_adaptive_sequential_mnist() (+46 more)

### Community 5 - "trainable_temporal_mnist.py"
Cohesion: 0.06
Nodes (47): _attach_frozen_comparisons(), available_ltw_diagnostic_arms(), ltw_scope_mask(), LTWDiagnosticArm, LTWDiagnosticResult, _mean_absolute_change(), plot_ltw_diagnostic(), Path (+39 more)

### Community 6 - "FrozenProbeResult"
Cohesion: 0.15
Nodes (14): FrozenProbeResult, FrozenReadoutAdapterResult, FrozenTaskResult, plot_frozen_probe_result(), plot_frozen_readout_adapter_result(), plot_frozen_task_result(), Path, Complete frozen representation probe result. (+6 more)

### Community 7 - "sprint15_frozen_readout_adapter_robustness.py"
Cohesion: 0.19
Nodes (20): make_generator(), Create a per-device generator when the backend supports one. PyTorch/XLA random…, _accuracy(), _apply_noise(), _collect_training_examples(), _conditions(), _evaluate(), _instant_reflex_predictions() (+12 more)

### Community 8 - "sync"
Cohesion: 0.07
Nodes (51): BaseException, _extract_reservoir_features(), _fit_and_measure(), _measure_accuracy(), _adapt_block(), _measure_mode(), accelerator_memory(), AcceleratorMemory (+43 more)

### Community 9 - "shd_calibrated_baselines.py"
Cohesion: 0.08
Nodes (32): load_shd_tensors(), Download, verify, bin, and cache the official SHD train/test splits., _attach_raw_comparisons(), available_shd_calibrated_baseline_arms(), matched_temporal_conv_channels(), plot_shd_calibrated_baselines(), Path, Phase 44 validation-calibrated, parameter-matched SHD temporal baselines. (+24 more)

### Community 10 - "AMMC Research Log"
Cohesion: 0.02
Nodes (122): 2026-08-09 - Phase 27 utility-gated structural MNIST result, 2026-08-09 - Phase 28 adaptive-neuron experiment generated, 2026-08-09 - Phase 28 adaptive-neuron result, 2026-08-09 - Phase 29 executable-delay experiment generated, 2026-08-09 - Phase 29 executable-delay result, 2026-08-09 - Phase 30 trainable-delay experiment generated, 2026-08-09 - Phase 30 trainable-delay result, 2026-08-09 - Phase 31 full SHD transfer result (+114 more)

### Community 11 - "EventMNISTConfig"
Cohesion: 0.06
Nodes (37): available_adaptive_sequential_arms(), assign_fixed_delays(), available_delayed_sequential_arms(), DelayedSequentialClassifier, Assign delays only to recurrent edges; sensor timing remains unchanged., Final-state LIF classifier with executable fixed delay buckets., EventMNISTConfig, Run all raw-pixel and frozen-AMMC MNIST comparisons. (+29 more)

### Community 12 - "shd_temporal_pyramid.py"
Cohesion: 0.08
Nodes (25): delayed_sparse_current(), Route each edge from the source-state bucket selected by delay metadata., Sparse recurrent LIF classifier with optional fixed delay buckets., SHDSparseClassifier, _attach_temporal_comparisons(), default_shd_temporal_pyramid_arms(), _next_power_of_two(), plot_shd_temporal_pyramid() (+17 more)

### Community 13 - "ammc_gen5/gen16_local_score_credit.py"
Cohesion: 0.13
Nodes (30): _action_vectors(), available_gen16_strategies(), decide_gen16(), _environment(), _evaluate(), _final_by_seed(), _gain_by_seed(), Gen16Config (+22 more)

### Community 14 - "ammc_gen5/gen6_successor.py"
Cohesion: 0.14
Nodes (23): available_gen6_successor_arms(), _build_model(), _completed_result(), decide_gen6_successor(), Gen6SuccessorArm, Gen6SuccessorResult, matched_shared_residual_channels(), plot_gen6_successor() (+15 more)

### Community 15 - "ammc_gen5/gen9_continual_adaptation.py"
Cohesion: 0.10
Nodes (34): _adapt_block(), _adaptation_curve(), apply_sensor_damage(), available_gen9_adaptation_strategies(), available_gen9_source_arms(), _completed_result(), _curve_auc(), decide_gen9_continual_adaptation() (+26 more)

### Community 16 - "ammc_gen5/nmnist_accuracy_benchmark.py"
Cohesion: 0.11
Nodes (34): _augment_events(), available_nmnist_accuracy_arms(), build_nmnist_accuracy_model(), bundle_nmnist_accuracy_artifacts(), decide_nmnist_accuracy(), encode_nmnist_full_resolution(), estimate_nmnist_dense_macs(), load_nmnist_accuracy_tensors() (+26 more)

### Community 17 - "ammc_gen5/gen30_dendritic_predictive_credit.py"
Cohesion: 0.09
Nodes (29): available_gen30_arms(), _build_datasets(), bundle_gen30_artifacts(), decide_gen30(), DendriticCreditNetwork, _evaluate(), Gen30Config, Gen30Result (+21 more)

### Community 18 - "Core findings"
Cohesion: 0.03
Nodes (64): 10. Deterministic browser replay is required for champion comparison, 11. Browser monitoring requires a visible/foreground tab, 12. Seeded champion replay now transfers visible food-seeking behavior, 13. Plasticity replay closes the learning loop but destabilizes the champion, 14. Sprint 11 raw outputs verify convergence and plasticity benefit, 15. Phase 11 evidence bundle is complete; retention and throughput are now quantified, 16. Gen-5 must transition to TPU/XLA-first execution, 17. Throughput benchmarks must report topology load (+56 more)

### Community 19 - "TensorEnvironment2D"
Cohesion: 0.05
Nodes (53): _make_generator(), Transducer that can invert food/toxin channels for adaptation tests., _resolve_device(), SensorPerturbationTransducer, EvolvingHeadlessAMMCLoop, EvolvingLoopConfig, _MissingModule, Central evolutionary runtime loop for AMMC Gen-5. Sprint 8 binds: Environment… (+45 more)

### Community 20 - "ammc_gen5/gen17_sparse_spiking_credit.py"
Cohesion: 0.11
Nodes (33): _action_vectors(), available_gen17_strategies(), bernoulli_spike_code(), decide_gen17(), _environment(), _evaluate(), _final_by_seed(), _gain_by_seed() (+25 more)

### Community 21 - "ammc_gen5/gen25_event_driven_sparse_audit.py"
Cohesion: 0.10
Nodes (29): bundle_gen25_artifacts(), _compile_module(), decide_gen25(), DenseResidualPipeline, _failed_metrics(), _fixed_density_events(), Gen25Config, Gen25Result (+21 more)

### Community 22 - "shd_capacity.py"
Cohesion: 0.16
Nodes (17): _accuracy_delta(), _attach_capacity_comparisons(), default_shd_capacity_arms(), _next_power_of_two(), plot_shd_capacity(), Path, Phase 34 no-delay hidden-capacity scaling on SHD. The preceding representation…, run_shd_capacity_scaling() (+9 more)

### Community 23 - "tensor_environment.py"
Cohesion: 0.09
Nodes (32): default_sparse_efficiency_groups(), Return sparse-efficiency groups that isolate each proposed optimization., Configuration for sparse-efficiency ablations across neuron scale points., _sparse_efficiency_environment_config(), SparseEfficiencyConfig, available_world_presets(), _MissingModule, Vectorized 2D embodiment environment for AMMC Gen-5. Sprint 4 target: simulate… (+24 more)

### Community 24 - "trainable_delays_mnist.py"
Cohesion: 0.08
Nodes (26): _attach_fixed_delay_comparisons(), available_trainable_delay_arms(), plot_trainable_delays_mnist(), Path, Phase 30 trainable delay assignment on row-sequential MNIST. Phase 29…, Mix source states across delay buckets with per-edge differentiable gates., Sparse LIF classifier with differentiable delay-bucket gates., run_trainable_delays_mnist() (+18 more)

### Community 25 - "ammc_gen5/gen15_reward_baseline.py"
Cohesion: 0.11
Nodes (30): _action_vectors(), available_gen15_strategies(), decide_gen15(), discounted_returns(), _environment(), _evaluate(), Gen15Config, Gen15RewardBaselineResult (+22 more)

### Community 26 - "synthesize_gen5_evidence"
Cohesion: 0.10
Nodes (16): _claim(), Gen5EvidenceSynthesisResult, plot_gen5_evidence_synthesis(), Path, Reproducible synthesis and architecture closeout of Gen-5 evidence., _render_report(), _screen_ops_proxy(), _strategy_row() (+8 more)

### Community 27 - "ammc_gen5/gen28_triton_event_kernel.py"
Cohesion: 0.11
Nodes (25): build_event_list(), bundle_gen28_artifacts(), decide_gen28(), EndToEndTritonPipeline, EventNativePipeline, _fixed_density_events(), Gen28Config, Gen28Result (+17 more)

### Community 28 - "ammc_gen5/gen10_robust_representation.py"
Cohesion: 0.13
Nodes (25): available_gen10_representation_arms(), _build_model(), _completed_result(), decide_gen10_robust_representation(), Gen10RepresentationArm, Gen10RobustRepresentationResult, plot_gen10_robust_representation(), Path (+17 more)

### Community 29 - "sprint16_frozen_embodied_adapter.py"
Cohesion: 0.13
Nodes (28): default_foraging_seed_edges(), Return the hand-coded prior used by the early 10k-agent Colab runs., _action_directions(), _adapter_training_data(), _config_dict(), _directions_to_actions(), _evaluate_policy(), main() (+20 more)

### Community 30 - "run_milestone_a"
Cohesion: 0.14
Nodes (16): available_milestone_a_arms(), decide_milestone_a(), hierarchical_residual_parameter_count(), matched_hierarchical_residual_channels(), plot_milestone_a(), Path, run_milestone_a(), _save_progress() (+8 more)

### Community 31 - "ammc_gen5/gen18_local_credit_replication.py"
Cohesion: 0.15
Nodes (26): available_gen18_strategies(), confidence_lower_bound(), decide_gen18(), _final_by_seed(), _gain_by_seed(), Gen18Config, Gen18LocalCreditReplicationResult, _max_optional() (+18 more)

### Community 32 - "ammc_gen5/gen13_local_plasticity.py"
Cohesion: 0.12
Nodes (24): available_gen13_plasticity_strategies(), _completed_result(), decide_gen13_local_plasticity(), Gen13LocalPlasticityResult, plot_gen13_local_plasticity(), Path, Gen-13 frozen sensory backbone with manual three-factor local plasticity., Apply a local pre × postsynaptic-error × modulator update in place. (+16 more)

### Community 33 - "ammc_gen5/gen22_dual_memory_replication.py"
Cohesion: 0.16
Nodes (22): _adapt_model(), _attach_paired_comparisons(), available_gen22_arms(), bundle_gen22_artifacts(), decide_gen22(), _gen21_config(), Gen22Config, Gen22Result (+14 more)

### Community 34 - "shd_analog_topology.py"
Cohesion: 0.11
Nodes (21): _attach_comparisons(), available_shd_analog_topology_arms(), _next_power_of_two(), plot_shd_analog_topology(), Path, Phase 40 analog/topology controls following the Phase 39 falsification., run_shd_analog_topology(), SHDAnalogTopologyResult (+13 more)

### Community 35 - "ChampionExporter"
Cohesion: 0.14
Nodes (11): ChampionExportConfig, ChampionExporter, ChampionExportResult, Any, Path, Champion genome export for the Gen-5 -> Gen-4 visual bridge., Browser-compatible export settings., Paths and summary for an exported champion bundle. (+3 more)

### Community 36 - "ammc_gen5/gen11_plastic_adapter.py"
Cohesion: 0.14
Nodes (22): _adaptation_curve(), available_gen11_adaptation_strategies(), _completed_result(), _curve_auc(), decide_gen11_plastic_adapter(), Gen11PlasticAdapterResult, plot_gen11_plastic_adapter(), Path (+14 more)

### Community 37 - "ammc_gen5/gen23_boundary_consolidation.py"
Cohesion: 0.14
Nodes (24): disjoint_sensor_damage_indices(), _adapt_bounded(), _attach_comparisons(), available_gen23_arms(), bundle_gen23_artifacts(), _consolidate_boundary(), decide_gen23(), _gen22_config() (+16 more)

### Community 38 - "benchmark_throughput.py"
Cohesion: 0.27
Nodes (17): _adjacency_path_candidates(), build_seed_edges(), _discover_adjacency_files(), _edge_record(), _expand_adjacency_request(), _file_sha256(), load_adjacency_edges(), main() (+9 more)

### Community 39 - "ammc_gen5/gen14_reward_eligibility.py"
Cohesion: 0.15
Nodes (23): available_gen14_strategies(), decide_gen14(), Gen14Config, Gen14RewardEligibilityResult, _mean(), plot_gen14_reward_eligibility(), Path, Gen-14 reward-modulated embodied eligibility-trace experiment. This program is… (+15 more)

### Community 40 - "ammc_gen5/gen19_nmnist_state_replication.py"
Cohesion: 0.12
Nodes (28): available_gen19_arms(), bundle_gen19_artifacts(), decide_gen19(), encode_nmnist_events(), Gen19Config, Gen19NMNISTStateReplicationResult, _load_nmnist_split(), load_nmnist_tensors() (+20 more)

### Community 41 - "ammc_gen5/gen24_compiled_residual_state.py"
Cohesion: 0.13
Nodes (25): available_gen24_models(), _benchmark_callable(), _build_model(), bundle_gen24_artifacts(), _compile_and_benchmark(), _cuda_peak_memory_mb(), decide_gen24(), Gen24Config (+17 more)

### Community 42 - "README.md"
Cohesion: 0.04
Nodes (26): Documentation index, Start here, Where to add new docs, Continual-adaptation program closeout, Next major-program choices, Sanity check against the project goal, Decision, Gen-14 reward-eligibility analysis (+18 more)

### Community 43 - "shd_representation.py"
Cohesion: 0.14
Nodes (19): _accuracy_delta(), _attach_diagnostic_comparisons(), available_shd_representation_arms(), _next_power_of_two(), plot_shd_representation(), _practical_threshold(), Path, Phase 32 diagnostic for the SHD sparse-representation bottleneck. Phase 31… (+11 more)

### Community 44 - "shd_matched_baselines.py"
Cohesion: 0.12
Nodes (21): _attach_baseline_comparisons(), available_shd_matched_baseline_arms(), gru_parameter_count(), GRUTemporalClassifier, matched_gru_hidden_units(), _next_power_of_two(), plot_shd_matched_baselines(), Path (+13 more)

### Community 45 - "ammc_gen5/gen27_trained_threshold_robustness.py"
Cohesion: 0.11
Nodes (24): Phase-49 residual-LIF computation after the temporal input operator., ResidualLIFStateHead, bundle_gen27_artifacts(), decide_gen27(), _evaluate_trained_model(), Gen27Config, Gen27Result, head_with_trace() (+16 more)

### Community 46 - "PlasticityAblationRunner"
Cohesion: 0.27
Nodes (8): PlasticityAblationConfig, PlasticityAblationRunner, Configuration for static/full/gated plasticity A/B trials., Run the static/full/gated plasticity ablation study., main(), parse_args(), Namespace, Sprint 11: seeded plasticity ablation study. Runs three groups under a shifted…

### Community 47 - "ammc_gen5/gen12_associative_memory.py"
Cohesion: 0.13
Nodes (23): _train_dropout_backbone(), available_gen12_memory_strategies(), decide_gen12_associative_memory(), Gen12AssociativeMemoryResult, plot_gen12_associative_memory(), Path, Gen-12 frozen robust backbone with context-gated associative memory., _result() (+15 more)

### Community 48 - "__init__.py"
Cohesion: 0.10
Nodes (30): bundle_gen26_artifacts(), decide_gen26(), _dense_currents(), Gen26Config, Gen26Result, _load_test_data(), _model_config(), plot_gen26() (+22 more)

### Community 49 - "DynamicSparseLinear"
Cohesion: 0.11
Nodes (12): DynamicSparseLinear, DynamicSparseLinearFunction, Function, Sparse, structurally plastic linear operator. The public API models true…, Replace active edges with the provided edge records., Allocate a new edge in the first free slot and return its slot id., Deactivate edges whose LTW is at or below the threshold., Transfer a fraction of STW into LTW for active edges. (+4 more)

### Community 50 - "ammc_gen5/gen7_predictive_state.py"
Cohesion: 0.12
Nodes (26): available_gen7_predictive_state_arms(), _build_model(), _completed_result(), decide_gen7_predictive_state(), _defined_values(), Gen7PredictiveStateArm, Gen7PredictiveStateResult, _mean_or_zero() (+18 more)

### Community 51 - "Gen20Config"
Cohesion: 0.17
Nodes (16): available_gen20_arms(), bundle_gen20_artifacts(), decide_gen20(), gen20_plot_series(), Gen20Config, Gen20Result, plot_gen20(), Path (+8 more)

### Community 52 - "SHDConfig"
Cohesion: 0.08
Nodes (25): HierarchicalResidualStateClassifier, Dilated temporal hierarchy with preserved direct and state traces., _attach_delay_comparisons(), available_shd_arms(), _load_or_build_split(), Phase 31 transfer benchmark on Spiking Heidelberg Digits (SHD). This phase…, Registered Phase 31 dataset, graph, and optimizer configuration., Timing-ablated classifier over per-channel event counts. (+17 more)

### Community 53 - "shd_sparse_mechanisms.py"
Cohesion: 0.16
Nodes (15): _attach_mechanism_comparisons(), available_shd_sparse_mechanism_arms(), _next_power_of_two(), plot_shd_sparse_mechanisms(), Path, Phase 39 causal ablation of the sparse feedforward SHD advantage. Phase 38…, run_shd_sparse_mechanisms(), SHDSparseMechanismsResult (+7 more)

### Community 54 - "shd_temporal_controls.py"
Cohesion: 0.14
Nodes (17): _attach_control_comparisons(), available_shd_temporal_control_arms(), budget_matched_bottleneck(), _next_power_of_two(), plot_shd_temporal_controls(), Path, Phase 37 causal controls for the Phase 36 SHD temporal-pyramid result. This…, Return bottleneck width and actual parameters under a fixed budget. (+9 more)

### Community 55 - "shd_sparse_width.py"
Cohesion: 0.16
Nodes (16): _attach_width_comparisons(), default_shd_sparse_width_arms(), _next_power_of_two(), plot_shd_sparse_width(), Path, Phase 41 fixed-budget scaling of the supported sparse leaky analog model., run_shd_sparse_width(), SHDSparseWidthArm (+8 more)

### Community 56 - "ammc_gen5/gen20_spiking_spatiotemporal.py"
Cohesion: 0.17
Nodes (20): build_gen20_model(), _checkpoint_path(), estimate_gen20_operations(), Gen20SpatiotemporalTeacher, _has_record(), _load_progress(), _load_teacher(), _measure() (+12 more)

### Community 57 - "device_kind"
Cohesion: 0.16
Nodes (17): device_kind(), _attach_no_delay_comparisons(), default_shd_delay_interaction_arms(), _next_power_of_two(), plot_shd_delay_interaction(), Path, Phase 35 factorial SHD capacity-by-delay interaction. Phase 34 found a variable…, run_shd_delay_interaction() (+9 more)

### Community 58 - "shd_initialization_robustness.py"
Cohesion: 0.14
Nodes (18): _train_model(), _attach_comparisons(), available_shd_initialization_arms(), _next_power_of_two(), plot_shd_initialization_robustness(), Path, Phase 42 topology/readout seed decomposition for sparse SHD models., run_shd_initialization_robustness() (+10 more)

### Community 59 - "DenseLIFPolicy"
Cohesion: 0.25
Nodes (3): DenseLIFPolicy, DenseMLPPolicy, Small dense LIF-style baseline with fixed topology. This internal…

### Community 60 - "ammc_gen5/gen21_matched_causal_mechanisms.py"
Cohesion: 0.14
Nodes (28): available_gen21_arms(), _build_backbone(), bundle_gen21_artifacts(), _causal_accuracy(), decide_gen21(), _evaluate_backbone(), _evaluate_model(), Gen21Config (+20 more)

### Community 61 - "A-SNN / AMMC"
Cohesion: 0.05
Nodes (37): A-SNN / AMMC, Browser sandbox quick start, Common Gen-5 commands, Consolidated milestone workflow, Current research baseline, Evidence discipline, Full-resolution N-MNIST accuracy benchmark (preregistered), Gen-10 masked-sensor representation reset (completed: `stop`) (+29 more)

### Community 62 - "PlasticStateAdapter"
Cohesion: 0.24
Nodes (3): PlasticStateAdapter, Frozen TCN predictor plus a bounded analog or LIF correction pathway., _multiscale_features()

### Community 63 - "utility_gated_structural_mnist.py"
Cohesion: 0.07
Nodes (34): Sprout unique targeted edges while preserving every existing edge., sprout_targeted_edges(), Final-state row-sequential classifier with a fixed sparse topology., TrainableSequentialClassifier, _attach_comparisons(), available_utility_gated_arms(), build_sensor_candidate_pool(), _deactivate_slots() (+26 more)

### Community 64 - "ConvPLIFClassifier"
Cohesion: 0.16
Nodes (10): ConvPLIFClassifier, FrameCNN, LearnableLIF2d, _ModuleBase, Strong static-frame ceiling using all full-resolution event bins., Dense 3-D convolutional reference retaining spatial and temporal axes., Channel-wise parametric LIF cell with surrogate gradients., Full-resolution convolutional SNN with learnable membrane dynamics. (+2 more)

### Community 65 - "4. Development milestones"
Cohesion: 0.09
Nodes (23): 1. Paradigm shift, 2. Lessons carried forward from Gen-4, 3. Core modules, 4. Development milestones, 5. Validation gates, AMMC Gen-5 Architecture, Astrocytes are low-frequency spatial loss, Champion export (+15 more)

### Community 66 - "DualTensorManager"
Cohesion: 0.13
Nodes (11): AstrocyteGridConfig, DualTensorManager, _MissingModule, Dual-frequency electrical/chemical tensor manager., Nearest-neighbor sample chemical state at normalized xyz positions.…, Low-frequency chemical grid configuration., Synchronizes fast sparse electrical activity with slow astrocyte fields., Accumulate high-frequency spike/activity density into the slow grid. (+3 more)

### Community 67 - "Phase 11 uploaded outputs review"
Cohesion: 0.09
Nodes (22): Artifact inventory, Current evidence status, Image-level single-run telemetry read, Phase 11 uploaded outputs review, Raw files available in workspace, Raw Sprint 11 files still not found, Recommended next actions, Uploaded images available as attachments (+14 more)

### Community 68 - "ammc_gen5/gen29_program_closure.py"
Cohesion: 0.19
Nodes (10): bundle_gen29_artifacts(), Gen29Result, Path, Deterministic Gen-29 causal evidence closure., _render_report(), run_gen29(), _write_csv(), main() (+2 more)

### Community 69 - "ammc_gen5/gen8_temporal_binding.py"
Cohesion: 0.09
Nodes (29): available_gen8_temporal_binding_arms(), _completed_result(), decide_gen8_temporal_binding(), Gen8TemporalBindingArm, Gen8TemporalBindingResult, matched_temporal_binding_channels(), _mean(), plot_gen8_temporal_binding() (+21 more)

### Community 70 - "verify_phase11_outputs.py"
Cohesion: 0.36
Nodes (15): audit_groups(), build_file_index(), build_summaries(), int_float(), main(), parse_args(), Any, Namespace (+7 more)

### Community 71 - "LTWSTWMemory"
Cohesion: 0.16
Nodes (8): LTWSTWConfig, LTWSTWMemory, _MissingModule, Short-term / long-term synaptic memory utilities., Memory dynamics constants., Applies AMMC's two-tier memory rules to tensor weights., Return optimizer groups that can treat STW and LTW differently., _require_torch()

### Community 72 - "verify_graph"
Cohesion: 0.25
Nodes (9): GraphifyContractTest, graph_strings(), main(), missing_anchors(), _nodes(), Any, Path, _strings() (+1 more)

### Community 73 - "AMMC Gen-5"
Cohesion: 0.09
Nodes (21): Accelerator runtime stance, AMMC Gen-5, Baseline comparison scaffold, Champion export, Design stance, Latest successor experiment, Multi-seed convergence trials, Neuron / decision-node scaling (+13 more)

### Community 74 - "MultiTimescaleResidualPLIF"
Cohesion: 0.23
Nodes (6): MultiTimescaleResidualPLIF, _ModuleBase, Parametric LIF cell initialized to a specified membrane timescale., Spatial stem plus fast/medium/slow LIF banks and spiking fusion state., _require_torch(), TimescaleLIF2d

### Community 75 - "Gen21MechanismReadout"
Cohesion: 0.26
Nodes (3): Gen21MechanismReadout, no_grad, Budget-matched residual readout with mechanism-specific adaptation.

### Community 77 - "SharedResidualStateTCNClassifier"
Cohesion: 0.18
Nodes (5): MaskedSensorResidualClassifier, Residual state model with a parameter-free clean-target alignment loss., TCN direct predictor plus zero-initialized weight-shared state logits., SharedResidualStateTCNClassifier, skipIf

### Community 78 - "top_fraction_spike_code"
Cohesion: 0.36
Nodes (8): _backbone_features(), _measure_memory(), _prototype_curve(), Encode each feature vector with a fixed sparse rank-order spike code., top_fraction_spike_code(), _local_curve(), _measure_local(), skipIf

### Community 79 - "Gen-6 successor preregistration"
Cohesion: 0.10
Nodes (18): Arms, Gen-6 successor preregistration, Hypothesis, Protocol, Terminal pass gate, Why this is a new generation, Gen-20 terminal update (2026-08-20), Gen-5 milestone roadmap (+10 more)

### Community 80 - "FrozenEventReservoir"
Cohesion: 0.08
Nodes (25): FrozenEventReservoir, Frozen sparse LIF reservoir used as the AMMC representation substrate., Return sensor, hidden, and full frozen traces without changing state., Return flattened pre-reset state at every simulated timestep., _attach_causal_deltas(), disable_recurrent_edges(), _hidden_event_rate(), plot_recurrence_ablation() (+17 more)

### Community 81 - "FrozenReadoutAdapterRunner"
Cohesion: 0.13
Nodes (12): _accuracy(), FrozenProbeSummaryRecord, FrozenReadoutAdapterRunner, FrozenReadoutAdapterSummaryRecord, FrozenTaskSummaryRecord, _instant_reflex_predictions(), _integrated_reflex_predictions(), One row summarizing a linear probe for one frozen task. (+4 more)

### Community 84 - "Frozen diversified task benchmarks"
Cohesion: 0.12
Nodes (16): Augmented readout-adapter robustness result, Baselines reported, Colab command, Expected outputs, First CUDA result, First readout-adapter generalization result, First readout-adapter result, First readout-adapter robustness result (+8 more)

### Community 85 - "Project decisions"
Cohesion: 0.12
Nodes (17): Decision: Add explicit Gen-5 browser transducer bridge mode, Decision: Add Phase 11 evidence verifier, Decision: Add retention/forgetting ablation protocol, Decision: Advance to controlled evaluation and champion-stability phase, Decision: Bind environment, transducer, and evolver into one epoch loop, Decision: Build vectorized embodiment before CUDA allocation, Decision: Close Phase 11 evidence gate and move to tuning plus fair baselines, Decision: Evolve per-agent sparse genomes as batched tensors (+9 more)

### Community 86 - "FrozenTaskRunner"
Cohesion: 0.40
Nodes (5): FrozenTaskRunner, _opposite_direction(), Evaluate a frozen sparse AMMC brain on small diversified tasks., _TaskBatch, _write_direction()

### Community 88 - "shd_spiking_temporal_conv.py"
Cohesion: 0.13
Nodes (18): _attach_comparisons(), available_shd_spiking_temporal_conv_arms(), matched_temporal_conv_state_channels(), plot_shd_spiking_temporal_conv(), Path, Phase 45 learned temporal convolution with analog and spiking state., Trainable temporal filters followed by leaky analog or LIF state., run_shd_spiking_temporal_conv() (+10 more)

### Community 89 - "sprint15_frozen_readout_adapter_generalization.py"
Cohesion: 0.25
Nodes (13): _accuracy(), _evaluate_batch(), _instant_reflex_predictions(), _integrated_reflex_predictions(), main(), parse_args(), _plot_generalization(), Namespace (+5 more)

### Community 90 - "shd_validation_checkpoint.py"
Cohesion: 0.09
Nodes (25): DenseAnalogTemporalClassifier, Dense leaky analog control with a matched temporal-pyramid readout., FixedBudgetSparseAnalogClassifier, Sparse analog classifier whose total effective parameter budget is fixed., BudgetMatchedTemporalReadout, disable_shd_recurrent_edges(), Time-aware non-spiking control over the original binned SHD events., Deactivate hidden-source edges while preserving sensor projections. (+17 more)

### Community 91 - "Gen-7 predictive-state preregistration"
Cohesion: 0.15
Nodes (11): Gen-7 predictive-state preregistration, Hypothesis, Motivation, Protocol, Registered arms, Terminal pass gate, Confirmation, Gate decision and sanity check (+3 more)

### Community 92 - "Gen-8 time-local predictive-binding preregistration"
Cohesion: 0.15
Nodes (11): Gen-8 time-local predictive-binding preregistration, Hypothesis, Motivation, Protocol, Registered arms, Terminal pass gate, Confirmation, Gen-8 time-local predictive-binding analysis (+3 more)

### Community 93 - "RetentionAblationRunner"
Cohesion: 0.24
Nodes (8): Configuration for original -> perturbed -> original retention trials., Run original -> perturbed -> original trials to measure forgetting., RetentionAblationConfig, RetentionAblationRunner, main(), parse_args(), Namespace, Sprint 11: retention / catastrophic-forgetting ablation. This run extends the…

### Community 94 - "available_frozen_tasks"
Cohesion: 0.13
Nodes (20): available_frozen_tasks(), FrozenProbeConfig, FrozenRepresentationProbeRunner, FrozenTaskConfig, Configuration for linear probes over frozen AMMC traces., Configuration for frozen diversified task evaluation., Train tiny linear readouts on frozen AMMC trace features. The sparse AMMC…, Return available synthetic frozen task names. (+12 more)

### Community 95 - "FrozenReadoutAdapterConfig"
Cohesion: 0.26
Nodes (10): FrozenReadoutAdapterConfig, Configuration for a deployable readout/transducer adapter. The recurrent sparse…, main(), parse_args(), _parse_variant(), _plot_sweep(), Namespace, Path (+2 more)

### Community 96 - "Harder bot-world benchmarks"
Cohesion: 0.17
Nodes (12): Baselines to carry forward, Delay-1 screen note, Delay-2 full-run console note, Delay-2 neuron-scaling console note, Delay-2 screen note, Delay-3 screen note, First CUDA result, Harder bot-world benchmarks (+4 more)

### Community 97 - "Phase 26: Targeted Synaptogenesis on Sequential MNIST"
Cohesion: 0.17
Nodes (10): Colab CUDA run, Decision rule, First CUDA result (2026-08-09), Paired structural interventions, Phase 26: Targeted Synaptogenesis on Sequential MNIST, Decision, Main result, Phase 26 targeted-synaptogenesis analysis (+2 more)

### Community 98 - "Phase 25: Fixed-Topology LTW Training on Sequential MNIST"
Cohesion: 0.17
Nodes (10): Colab CUDA run, Decision rule, First CUDA result (2026-08-09), Paired interventions, Phase 25: Fixed-Topology LTW Training on Sequential MNIST, Decision, Goal sanity check, Phase 25 CUDA sequential LTW-training analysis (+2 more)

### Community 99 - "TemporalDilatedTCNClassifier"
Cohesion: 0.15
Nodes (14): matched_predictive_state_channels(), predictive_state_parameter_count(), Count TCN plus state dynamics, predictor, and conditional gate., Preserve the exact width selected for the matched TCN., _build_model(), available_ssc_efficiency_baseline_arms(), matched_temporal_tcn_channels(), _multiscale_features() (+6 more)

### Community 100 - "Gen-20 spiking spatial-temporal translation preregistration"
Cohesion: 0.18
Nodes (10): Arms, Bounded screen, Confirmation, Frozen representation and split protocol, Gen-20 spiking spatial-temporal translation preregistration, Implementation, Motivation, Pass gates (+2 more)

### Community 101 - "Phase 24: Streaming Row-Sequential MNIST"
Cohesion: 0.18
Nodes (9): Causal comparison, Colab CUDA run, Decision rule, First CUDA result (2026-08-09), Phase 24: Streaming Row-Sequential MNIST, Decision, Goal sanity check, Phase 24 CUDA row-sequential MNIST analysis (+1 more)

### Community 102 - "LinearRewardPolicy"
Cohesion: 0.24
Nodes (7): LinearRewardPolicy, manual_score_gradients(), Measure the maximum error between autograd and the manual local rule., Eight sensory channels mapped directly to four action logits., Return local ascent directions for ``return * grad(log policy)``. The action-…, score_gradient_parity(), skipIf

### Community 103 - "Phase 23: Causal Recurrence Ablation"
Cohesion: 0.20
Nodes (8): Causal comparison, Colab CUDA run, Decision rule, First CUDA result (2026-08-09), Phase 23: Causal Recurrence Ablation, Decision, Phase 23 CUDA recurrence-ablation analysis, Results

### Community 104 - "Gen-5 TPU/XLA migration note"
Cohesion: 0.20
Nodes (9): Colab TPU commands, Colab TPU setup, Current code changes, Decision, Gen-5 TPU/XLA migration note, Migration caveats, Next validation target, Runtime contract (+1 more)

### Community 105 - "load_ssc_tensors"
Cohesion: 0.16
Nodes (19): bin_shd_events(), _download_bytes(), _download_to(), ensure_shd_files(), _md5(), plot_shd_benchmark(), Path, Bin one variable-length SHD event stream into a dense uint8 tensor. (+11 more)

### Community 106 - "Gen-9 continual-adaptation preregistration"
Cohesion: 0.22
Nodes (8): Gen-9 continual-adaptation preregistration, Measurements, Program boundary, Registered confirmation strategies, Registered source models, Result, Shift and data separation, Terminal pass gate

### Community 107 - "Approved sequence"
Cohesion: 0.22
Nodes (9): Approved sequence, D1-D2 - Optional evolutionary diagnostics, Decision, E1-E3 - Primary audio evidence, Interpretation boundary, LTH-informed itinerary review, R1 - Context-specific dendritic supermasks, Review matrix (+1 more)

### Community 108 - "Full-resolution N-MNIST accuracy benchmark preregistration"
Cohesion: 0.22
Nodes (8): Colab command, Confirmation, Frozen candidates, Full-resolution N-MNIST accuracy benchmark preregistration, Input protocol, Scope, Screen and promotion, Terminal rule

### Community 109 - "Primary evidence track roadmap"
Cohesion: 0.22
Nodes (9): Claim boundary, Deferred research track, LTH-informed itinerary refinements, Primary evidence track roadmap, Primary question, Purpose, Stage E1 - Canonical protocol and runner, Stage E2 - Paired confirmation (+1 more)

### Community 110 - "Browser seeded plasticity replay monitor - champion-001"
Cohesion: 0.22
Nodes (8): Behavioral findings, Browser seeded plasticity replay monitor - champion-001, Comparison against no-plasticity seeded replay, Interpretation, Key sample observations, Memory and topology notes, Quantitative notes, Setup observed

### Community 111 - "CUDA saturated-topology compiled hotpath throughput"
Cohesion: 0.22
Nodes (8): Caveats, Comparison against prior full-step saturated CUDA compile, Comparison against saturated eager full-step CUDA, CUDA saturated-topology compiled hotpath throughput, Follow-up implementation, Interpretation, Next benchmark actions, Results

### Community 112 - "Graphify knowledge workflow"
Cohesion: 0.25
Nodes (7): Acceptance contract, Building the graph, Canonical corpus, Graphify knowledge workflow, Installation, Recall across sessions, Updating

### Community 113 - "event_mnist.py"
Cohesion: 0.07
Nodes (37): build_event_reservoir_edges(), _deterministic_subset(), EventMNISTDecompositionResult, EventMNISTResult, _extract_latency_features(), _extract_reservoir_components(), _extract_sensor_trace(), latency_encode() (+29 more)

### Community 114 - "Phase 11 Colab Runbook"
Cohesion: 0.25
Nodes (8): 1. Retention / forgetting ablation, 2. Throughput scaling, 3. Baseline comparison, 4. Neuron / decision-node scaling, 5. Sparse-efficiency ablation, 6. Zip and download all remaining outputs, 7. Verify package completeness, Phase 11 Colab Runbook

### Community 115 - "AMMC Gen-5 Colab Run Analysis — 500 Generations"
Cohesion: 0.25
Nodes (7): AMMC Gen-5 Colab Run Analysis — 500 Generations, Bundle validation, Champion topology, Immediate implications, Phase behavior, Summary, Telemetry findings

### Community 116 - "Browser Champion Bridge Replay Monitor — 2026-06-25"
Cohesion: 0.25
Nodes (7): Browser Champion Bridge Replay Monitor — 2026-06-25, Comparison to previous browser replay, Observation window, Recommended next step, Sampled telemetry, Setup, Summary findings

### Community 117 - "Phase 20 temporal-state MNIST: analysis and goal sanity check"
Cohesion: 0.25
Nodes (7): Aggregate result, Decision, Interpretation, Paired findings, Phase 20 temporal-state MNIST: analysis and goal sanity check, Project-goal sanity check, Validation boundary

### Community 118 - "CUDA saturated-topology throughput with `torch.compile`"
Cohesion: 0.25
Nodes (7): Action taken, Comparison against saturated eager CUDA run, CUDA saturated-topology throughput with `torch.compile`, Hotpath rerun, Interpretation, Next benchmark actions, Results

### Community 119 - "Phase 21 fixed-topology LTW training: analysis and goal sanity check"
Cohesion: 0.25
Nodes (7): Aggregate result, Decision, Learning and activity diagnostics, Paired findings, Phase 21 fixed-topology LTW training: analysis and goal sanity check, Project-goal sanity check, Validation boundary

### Community 120 - "Contributing notes"
Cohesion: 0.29
Nodes (6): Before changing code, Contributing notes, Current baseline discipline, Knowledge graph updates, When adding experiment results, When changing Gen-5 code

### Community 121 - ".step"
Cohesion: 0.29
Nodes (3): Run one vectorized embodied neural tick., Evolve the population from environment fitness and reset positions., Run multiple steps and return the final telemetry.

### Community 122 - "StreamingMNISTReservoir"
Cohesion: 0.19
Nodes (5): Sparse hidden reservoir receiving one downsampled image row per step., StreamingMNISTReservoir, summarize_sequential_mnist(), skipIf, SequentialMNISTContractTest

### Community 123 - "ResidualTemporalConvStateClassifier"
Cohesion: 0.22
Nodes (5): Return direct/state traces plus final state and mean activity. Gen-21 uses this…, Return the exact feature vector consumed by the frozen classifier., Preserve direct Conv1D features beside an analog or spiking state., ResidualTemporalConvStateClassifier, skipIf

### Community 124 - "Frozen embodied readout benchmark"
Cohesion: 0.29
Nodes (6): Colab CUDA command, Controlled comparison, First CUDA result, Frozen embodied readout benchmark, Outputs, What remains frozen

### Community 125 - "Gen-10 masked-sensor representation reset preregistration"
Cohesion: 0.29
Nodes (6): Frozen arms, Gen-10 masked-sensor representation reset preregistration, Hypothesis and boundary, Result, Screen and confirmation, Terminal gate

### Community 126 - "Gen-13 three-factor local-plasticity preregistration"
Cohesion: 0.29
Nodes (6): Causal controls, Frozen comparison, Gen-13 three-factor local-plasticity preregistration, Hypothesis, Observed terminal decision, Terminal gate

### Community 127 - "Gen-16 local score-function credit preregistration"
Cohesion: 0.29
Nodes (6): Arms and controls, Decision rule, Frozen budget, Frozen mechanism, Gen-16 local score-function credit preregistration, Question

### Community 128 - "Gen-17 sparse-spiking local-credit preregistration"
Cohesion: 0.29
Nodes (6): Arms, Decision rule, Frozen budget, Frozen translation, Gen-17 sparse-spiking local-credit preregistration, Question

### Community 129 - "Gen-19 N-MNIST residual-state replication preregistration"
Cohesion: 0.29
Nodes (6): Colab command, Decision, Frozen protocol, Gen-19 N-MNIST residual-state replication preregistration, Pass gates, Program boundary

### Community 130 - "Gen-21 matched causal mechanism benchmark preregistration"
Cohesion: 0.29
Nodes (6): Arms and confound controls, Frozen gates, Gen-21 matched causal mechanism benchmark preregistration, Interpretation limits, Question, Supported boundary

### Community 131 - "Gen-30 dendritic predictive-credit analysis"
Cohesion: 0.29
Nodes (6): Claim boundary, Decision, Gate outcome, Gen-30 dendritic predictive-credit analysis, Interpretation, Next design requirement

### Community 132 - "Gen-30 dendritic predictive-credit preregistration"
Cohesion: 0.29
Nodes (6): Fixed arms, Frozen implementation, Gates, Gen-30 dendritic predictive-credit preregistration, Microtask, Question

### Community 133 - "Program sanity check after Gen-20"
Cohesion: 0.29
Nodes (6): Goal, Next phase, Program sanity check after Gen-20, Sanity decision, What remains rejected or unproven, What the evidence supports

### Community 134 - "Phase 21: Fixed-Topology LTW Training"
Cohesion: 0.29
Nodes (6): Colab CUDA run, Comparison groups, Decision rule, Experimental boundary, First CUDA result, Phase 21: Fixed-Topology LTW Training

### Community 135 - "Browser Champion Replay Monitor — 2026-06-25"
Cohesion: 0.29
Nodes (6): Browser Champion Replay Monitor — 2026-06-25, Observation window, Recommended next test, Sampled telemetry, Setup, Summary findings

### Community 136 - "Browser seeded replay monitor — champion-001"
Cohesion: 0.29
Nodes (6): Behavioral findings, Browser seeded replay monitor — champion-001, Interpretation, Key sample observations, Quantitative notes, Setup observed

### Community 137 - "Phase 19 event representation decomposition: analysis"
Cohesion: 0.29
Nodes (6): Aggregate result, Decision, Interpretation, Paired findings, Phase 19 event representation decomposition: analysis, Validation boundary

### Community 138 - "Sprint 16 frozen embodied adapter analysis"
Cohesion: 0.29
Nodes (6): Decision, Evidence integrity, Main results, Sprint 16 frozen embodied adapter analysis, What it does not establish, What the result establishes

### Community 139 - "AMMC Gen-5 through Gen-10 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-10 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 140 - "AMMC Gen-5 through Gen-11 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-11 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 141 - "AMMC Gen-5 through Gen-12 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-12 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 142 - "AMMC Gen-5 through Gen-13 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-13 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 143 - "AMMC Gen-5 through Gen-14 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-14 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 144 - "AMMC Gen-5 through Gen-15 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-15 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 145 - "AMMC Gen-5 through Gen-16 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-16 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 146 - "AMMC Gen-5 through Gen-17 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-17 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 147 - "AMMC Gen-5 through Gen-18 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-18 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 148 - "AMMC Gen-5 through Gen-20 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-20 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 149 - "Gen-20 spiking spatial-temporal translation analysis"
Cohesion: 0.29
Nodes (6): Artifact note, Gen-20 spiking spatial-temporal translation analysis, Interpretation and limits, Program decision, Provenance and integrity, Screen results

### Community 150 - "AMMC Gen-5 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 151 - "AMMC Gen-5 through Gen-19 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-19 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 152 - "AMMC Gen-5/Gen-6 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5/Gen-6 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 153 - "AMMC Gen-5/Gen-6/Gen-7 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5/Gen-6/Gen-7 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 154 - "AMMC Gen-5/Gen-6/Gen-7/Gen-8 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5/Gen-6/Gen-7/Gen-8 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 155 - "AMMC Gen-5 through Gen-9 evidence report"
Cohesion: 0.29
Nodes (6): AMMC Gen-5 through Gen-9 evidence report, Claim ledger, Defensible contribution, Evidence sources, Executive conclusion, Next-generation roadmap

### Community 156 - "Phase 22 LTW optimization diagnostic: analysis and goal sanity check"
Cohesion: 0.29
Nodes (6): Decision, Phase 22 LTW optimization diagnostic: analysis and goal sanity check, Pre-registered gate result, Project-goal sanity check, Validation boundary, What the sweep establishes

### Community 157 - "CUDA champion-topology compiled hotpath throughput"
Cohesion: 0.29
Nodes (6): Comparison against saturated compiled hotpath, CUDA champion-topology compiled hotpath throughput, Edge-work perspective, Interpretation, Next benchmark actions, Results

### Community 158 - "CUDA champion-topology compiled hotpath throughput, fingerprinted"
Cohesion: 0.29
Nodes (6): Caveats, Comparison against saturated compiled hotpath, CUDA champion-topology compiled hotpath throughput, fingerprinted, Interpretation, Next benchmark actions, Results

### Community 159 - "CUDA champion-topology eager hotpath throughput"
Cohesion: 0.29
Nodes (6): Comparison against saturated eager hotpath, CUDA champion-topology eager hotpath throughput, Interpretation, Next benchmark actions, Non-comparable compiled champion context, Results

### Community 160 - "CUDA champion-topology eager hotpath throughput, fingerprinted"
Cohesion: 0.29
Nodes (6): Clean champion compiler comparison, Comparison against saturated eager hotpath, CUDA champion-topology eager hotpath throughput, fingerprinted, Interpretation, Next benchmark actions, Results

### Community 161 - "CUDA saturated-topology throughput analysis"
Cohesion: 0.29
Nodes (6): Comparison against prior 8-edge CUDA benchmark, Compile follow-up, CUDA saturated-topology throughput analysis, Interpretation, Next benchmark actions, Results

### Community 162 - "CUDA saturated-topology eager hotpath throughput"
Cohesion: 0.29
Nodes (6): Clean hotpath compiler comparison, Comparison against previous full-step eager benchmark, CUDA saturated-topology eager hotpath throughput, Interpretation, Next benchmark actions, Results

### Community 163 - "Repository structure"
Cohesion: 0.33
Nodes (6): Current working baseline, Gen-5 layout, Output bundle convention, Repository structure, Top-level directories, Top-level files

### Community 164 - "AdaptiveSequentialClassifier"
Cohesion: 0.22
Nodes (4): AdaptiveSequentialClassifier, Fixed sparse sequential SNN with optional adaptive thresholds. Adaptation is a…, AdaptiveSequentialMNISTContractTest, skipIf

### Community 165 - "Phase 18: frozen event-coded MNIST"
Cohesion: 0.33
Nodes (5): Decision rules, Experimental design, First CUDA result, Google Colab cells, Phase 18: frozen event-coded MNIST

### Community 166 - "Phase 19: event representation decomposition"
Cohesion: 0.33
Nodes (5): Colab command, Decision tree, First CUDA result, Phase 19: event representation decomposition, Representations

### Community 167 - "Gen-11 frozen sensory backbone plus plastic state adapter preregistration"
Cohesion: 0.33
Nodes (5): Frozen strategies, Gen-11 frozen sensory backbone plus plastic state adapter preregistration, New hypothesis, Protocol, Terminal LIF gate

### Community 168 - "Gen-12 frozen-backbone associative-memory preregistration"
Cohesion: 0.33
Nodes (5): Causal controls, Frozen comparison, Gen-12 frozen-backbone associative-memory preregistration, Hypothesis, Terminal gate

### Community 169 - "Gen-14 reward-modulated embodied eligibility preregistration"
Cohesion: 0.33
Nodes (5): Frozen screen, Gen-14 reward-modulated embodied eligibility preregistration, New hypothesis, Observed terminal decision, Terminal screen gate

### Community 170 - "Gen-18 held-out local-credit replication preregistration"
Cohesion: 0.33
Nodes (5): Decision, Frozen protocol, Gen-18 held-out local-credit replication preregistration, Pass gates, Question

### Community 171 - "Gen-19 N-MNIST residual-state replication analysis"
Cohesion: 0.33
Nodes (5): Artifact provenance, Frozen decision, Gen-19 N-MNIST residual-state replication analysis, Next decision, Result

### Community 172 - "Gen-22 dual-memory sequential-shift replication preregistration"
Cohesion: 0.33
Nodes (5): Claim boundary, Frozen primary endpoints, Gen-22 dual-memory sequential-shift replication preregistration, Hypothesis, Protocol

### Community 173 - "Phase 22: LTW Optimization Diagnostic"
Cohesion: 0.33
Nodes (5): Colab CUDA run, Decision rule, First CUDA result, Paired interventions, Phase 22: LTW Optimization Diagnostic

### Community 174 - "N-MNIST accuracy benchmark analysis"
Cohesion: 0.33
Nodes (5): Artifact limitation, Gen-20 handoff, N-MNIST accuracy benchmark analysis, Result, What this proves—and what it does not

### Community 175 - "Phase 38: parameter-matched SHD baselines"
Cohesion: 0.33
Nodes (5): Colab command, Models, Phase 38: parameter-matched SHD baselines, Purpose, Registered interpretation

### Community 176 - "Phase 37: SHD temporal-control decomposition"
Cohesion: 0.33
Nodes (5): Arms, Colab command, Phase 37: SHD temporal-control decomposition, Question, Registered gates

### Community 177 - "Phase 36: parameter-matched SHD temporal pyramid"
Cohesion: 0.33
Nodes (5): Colab command, Gates, Phase 36: parameter-matched SHD temporal pyramid, Registered arms, Why this phase exists

### Community 178 - "Phase 20: temporal-state preservation on MNIST"
Cohesion: 0.33
Nodes (5): Colab command, Decision rules, First CUDA result, Phase 20: temporal-state preservation on MNIST, Representations

### Community 179 - "AMMC-SNN Behavior Observation"
Cohesion: 0.33
Nodes (5): AMMC-SNN Behavior Observation, Distinct visible events, Final visible state, Key findings, Metrics

### Community 180 - "AMMC-SNN Behavior Observation — Corrected Parser"
Cohesion: 0.33
Nodes (5): AMMC-SNN Behavior Observation — Corrected Parser, Distinct visible events, Final visible state, Key findings, Metrics

### Community 181 - "CUDA delayed-reward delay-1 screen"
Cohesion: 0.33
Nodes (5): Comparison with earlier delay screens, CUDA delayed-reward delay-1 screen, Decision, Interpretation, Summary

### Community 182 - "CUDA delayed-reward delay-2 full run, console-only archive"
Cohesion: 0.33
Nodes (5): CUDA delayed-reward delay-2 full run, console-only archive, Decision, Interpretation, Run context, Summary

### Community 183 - "CUDA delayed-reward delay-2 screen"
Cohesion: 0.33
Nodes (5): Comparison against nearby delay screens, CUDA delayed-reward delay-2 screen, Decision, Interpretation, Summary

### Community 184 - "Phase 17 embodied action controls: analysis"
Cohesion: 0.33
Nodes (5): Decision, Interpretation, Main result, Phase 17 embodied action controls: analysis, Scientific boundary

### Community 185 - "Phase 18 frozen event-coded MNIST: analysis"
Cohesion: 0.33
Nodes (5): Decision, Interpretation, Phase 18 frozen event-coded MNIST: analysis, Result, Statistical boundary

### Community 186 - "Frozen Readout Adapter Augmented Robustness Result"
Cohesion: 0.33
Nodes (5): Aggregate adapter accuracy by condition, Decision, Findings, Frozen Readout Adapter Augmented Robustness Result, Interpretation

### Community 187 - "Frozen Readout Adapter Held-Out-Seed Generalization Result"
Cohesion: 0.33
Nodes (5): Decision, Finding, Frozen Readout Adapter Held-Out-Seed Generalization Result, Interpretation, Summary

### Community 188 - "Frozen Readout Adapter Robustness Result"
Cohesion: 0.33
Nodes (5): Aggregate adapter accuracy by condition, Decision, Findings, Frozen Readout Adapter Robustness Result, Interpretation

### Community 189 - "Frozen Readout Adapter Sweep CUDA Result"
Cohesion: 0.33
Nodes (5): Decision, Findings, Frozen Readout Adapter Sweep CUDA Result, Summary, Updated interpretation

### Community 190 - "Gen-10 robust-representation analysis"
Cohesion: 0.33
Nodes (5): Confirmed conventional controls, Decision, Gen-10 robust-representation analysis, Goal sanity check, Screen

### Community 191 - "Gen-29 causal evidence closure"
Cohesion: 0.33
Nodes (5): Claim ledger, Decision, Executive conclusion, Gen-29 causal evidence closure, Hashed evidence

### Community 192 - "Gen-9 continual-adaptation analysis"
Cohesion: 0.33
Nodes (5): Confirmed TCN adaptation controls, Decision, Gen-9 continual-adaptation analysis, Interpretation and boundary, Source screen

### Community 193 - "Milestone A: architecture screen and confirmation"
Cohesion: 0.33
Nodes (5): Confirmation, Decision, Milestone A: architecture screen and confirmation, Sanity check, Screen

### Community 194 - "CUDA neuron-scaling evaluation"
Cohesion: 0.33
Nodes (5): CUDA neuron-scaling evaluation, Decision, Interpretation, Recommended next experiment, Results

### Community 195 - "CUDA delay-2 neuron scaling, full artifact archive"
Cohesion: 0.33
Nodes (5): CUDA delay-2 neuron scaling, full artifact archive, Decision, Interpretation, Run context, Summary

### Community 196 - "CUDA delay-2 neuron scaling, console-only archive"
Cohesion: 0.33
Nodes (5): CUDA delay-2 neuron scaling, console-only archive, Decision, Interpretation, Run context, Summary

### Community 197 - "Gen-5 outputs"
Cohesion: 0.33
Nodes (5): Bundle convention, Gen-5 outputs, Latest terminal experiment, Legacy uploads, What each major bundle should include

### Community 198 - "Phase 31 screen analysis: SHD temporal transfer plumbing"
Cohesion: 0.33
Nodes (5): Goal sanity check and decision, Phase 31 screen analysis: SHD temporal transfer plumbing, Result, Scope, Systems and dynamics checks

### Community 199 - "Phase 34 analysis: SHD hidden-capacity scaling"
Cohesion: 0.33
Nodes (5): Dynamics and delay interaction, Goal sanity check and decision, Phase 34 analysis: SHD hidden-capacity scaling, Registered gates, Result

### Community 200 - "Phase 35 analysis: SHD capacity-delay interaction"
Cohesion: 0.33
Nodes (5): Decision, Goal sanity check, Mechanistic and systems interpretation, Phase 35 analysis: SHD capacity-delay interaction, Result

### Community 201 - "Phase 33 analysis: SHD representation diagnostic"
Cohesion: 0.33
Nodes (5): Goal sanity check and decision, Mechanistic diagnosis, Phase 33 analysis: SHD representation diagnostic, Registered gates, Result

### Community 202 - "Phase 37 analysis: SHD temporal-control decomposition"
Cohesion: 0.33
Nodes (5): Decision, Goal sanity check, Interpretation, Phase 37 analysis: SHD temporal-control decomposition, Result

### Community 203 - "Phase 36 analysis: parameter-matched SHD temporal pyramid"
Cohesion: 0.33
Nodes (5): Decision, Goal sanity check, Interpretation, Phase 36 analysis: parameter-matched SHD temporal pyramid, Registered-gate result

### Community 204 - "CUDA sparse-efficiency component screen"
Cohesion: 0.33
Nodes (5): CUDA sparse-efficiency component screen, Decision, Interpretation, Recommended next run, Results

### Community 205 - "CUDA sparse-efficiency screen"
Cohesion: 0.33
Nodes (5): CUDA sparse-efficiency screen, Decision, Interpretation, Recommended next run, Results

### Community 206 - "Phase 28: Adaptive Neurons on Sequential MNIST"
Cohesion: 0.40
Nodes (4): Colab CUDA run, Decision gate, Paired design, Phase 28: Adaptive Neurons on Sequential MNIST

### Community 207 - "Phase 29: Executable Axonal Delays on Sequential MNIST"
Cohesion: 0.40
Nodes (4): Colab CUDA run, Decision gate, Paired interventions, Phase 29: Executable Axonal Delays on Sequential MNIST

### Community 208 - "Embodied action controls"
Cohesion: 0.40
Nodes (4): Colab command, Decision rules, Embodied action controls, First CUDA result

### Community 209 - "Gen-11 plastic state-adapter analysis"
Cohesion: 0.40
Nodes (4): Evidence, Gen-11 plastic state-adapter analysis, Result, Sanity check against the project goal

### Community 210 - "Gen-12 associative-memory analysis"
Cohesion: 0.40
Nodes (4): Evidence, Gen-12 associative-memory analysis, Goal sanity check, Result

### Community 211 - "Gen-15 matched reward-baseline analysis"
Cohesion: 0.40
Nodes (4): Archived evidence, Gen-15 matched reward-baseline analysis, Interpretation, Registered gates

### Community 212 - "Gen-15 matched reward-baseline preregistration"
Cohesion: 0.40
Nodes (4): Frozen protocol, Gen-15 matched reward-baseline preregistration, Question, Terminal gate

### Community 213 - "Gen-16 local score-credit analysis"
Cohesion: 0.40
Nodes (4): Archived evidence, Gen-16 local score-credit analysis, Interpretation, Result

### Community 214 - "Gen-17 sparse-spiking local-credit analysis"
Cohesion: 0.40
Nodes (4): Gen-17 sparse-spiking local-credit analysis, Interpretation, Provenance, Result

### Community 215 - "Gen-18 held-out local-credit replication analysis"
Cohesion: 0.40
Nodes (4): Gen-18 held-out local-credit replication analysis, Provenance, Result, Sanity check against the project goal

### Community 216 - "Gen-21 matched causal mechanism benchmark analysis"
Cohesion: 0.40
Nodes (4): Decision, Gen-21 matched causal mechanism benchmark analysis, Interpretation, Result

### Community 217 - "Program sanity check after Gen-28"
Cohesion: 0.40
Nodes (4): Evidence-based answer, Next action, Original question, Program sanity check after Gen-28

### Community 218 - "Phase 30: Trainable Delay Assignment"
Cohesion: 0.40
Nodes (4): Arms, Colab CUDA run, Decision gate, Phase 30: Trainable Delay Assignment

### Community 219 - "Phase 27: Utility-Gated Structural Plasticity"
Cohesion: 0.40
Nodes (4): Colab run, Decision gate, Experimental arms, Phase 27: Utility-Gated Structural Plasticity

### Community 220 - "Phase 28 analysis: adaptive neurons on sequential MNIST"
Cohesion: 0.40
Nodes (4): Goal sanity check and decision, Mechanistic diagnosis, Phase 28 analysis: adaptive neurons on sequential MNIST, Result

### Community 221 - "CUDA delayed-reward delay-3 screen"
Cohesion: 0.40
Nodes (4): CUDA delayed-reward delay-3 screen, Decision, Interpretation, Summary

### Community 222 - "Phase 29 analysis: executable axonal delays"
Cohesion: 0.40
Nodes (4): Goal sanity check and decision, Mechanistic diagnosis, Phase 29 analysis: executable axonal delays, Result

### Community 223 - "CUDA frozen diversified tasks, summary archive"
Cohesion: 0.40
Nodes (4): CUDA frozen diversified tasks, summary archive, Decision, Interpretation, Summary

### Community 224 - "Frozen Readout Adapter CUDA Result"
Cohesion: 0.40
Nodes (4): Decision, Findings, Frozen Readout Adapter CUDA Result, Summary

### Community 225 - "CUDA frozen representation probe, summary archive"
Cohesion: 0.40
Nodes (4): CUDA frozen representation probe, summary archive, Decision, Interpretation, Summary

### Community 226 - "CUDA harder-world benchmark"
Cohesion: 0.40
Nodes (4): CUDA harder-world benchmark, Decision, Interpretation, Summary

### Community 227 - "Phase 40 analysis: SHD analog dynamics and topology"
Cohesion: 0.40
Nodes (4): Goal sanity check, Interpretation, Phase 40 analysis: SHD analog dynamics and topology, Registered gates

### Community 228 - "Phase 31 analysis: full SHD temporal transfer"
Cohesion: 0.40
Nodes (4): Goal sanity check and decision, Mechanistic diagnosis, Phase 31 analysis: full SHD temporal transfer, Result

### Community 229 - "Phase 42 analysis: SHD initialization robustness"
Cohesion: 0.40
Nodes (4): Goal sanity check, Interpretation, Phase 42 analysis: SHD initialization robustness, Registered gates

### Community 230 - "Phase 38 analysis: matched SHD baselines"
Cohesion: 0.40
Nodes (4): Decision, Goal sanity check, Phase 38 analysis: matched SHD baselines, Result

### Community 231 - "Phase 47: residual-state causal contribution on SHD"
Cohesion: 0.40
Nodes (4): Decision, Interpretation, Phase 47: residual-state causal contribution on SHD, Result

### Community 232 - "Phase 39 analysis: sparse SHD mechanisms"
Cohesion: 0.40
Nodes (4): Interpretation, Next decision, Phase 39 analysis: sparse SHD mechanisms, Registered gates

### Community 233 - "Phase 41 analysis: fixed-budget sparse width scaling"
Cohesion: 0.40
Nodes (4): Goal sanity check, Interpretation, Phase 41 analysis: fixed-budget sparse width scaling, Registered gates

### Community 234 - "Phase 43 analysis: validation-selected SHD checkpointing"
Cohesion: 0.40
Nodes (4): Goal sanity check, Interpretation, Phase 43 analysis: validation-selected SHD checkpointing, Registered gates

### Community 235 - "CUDA sparse-efficiency finalist run"
Cohesion: 0.40
Nodes (4): CUDA sparse-efficiency finalist run, Decision, Interpretation, Summary

### Community 236 - "CUDA sparse-efficiency finalist screen"
Cohesion: 0.40
Nodes (4): CUDA sparse-efficiency finalist screen, Decision, Interpretation, Summary

### Community 237 - "Phase 49: SSC matched baseline and efficiency audit"
Cohesion: 0.40
Nodes (4): Decision, Final sanity check, Phase 49: SSC matched baseline and efficiency audit, Result

### Community 238 - "Phase 48: residual-LIF replication on SSC"
Cohesion: 0.40
Nodes (4): Decision, Phase 48: residual-LIF replication on SSC, Result, Sanity check

### Community 239 - "Phase 30 analysis: trainable recurrent delays"
Cohesion: 0.40
Nodes (4): Goal sanity check and decision, Mechanistic diagnosis, Phase 30 analysis: trainable recurrent delays, Result

### Community 240 - "Phase 27 analysis: utility-gated structural MNIST"
Cohesion: 0.40
Nodes (4): Goal sanity check and decision, Mechanistic diagnosis, Phase 27 analysis: utility-gated structural MNIST, Result

### Community 242 - "Gen-13 local-plasticity analysis"
Cohesion: 0.50
Nodes (3): Decision, Gen-13 local-plasticity analysis, Result

### Community 243 - "AMMC research evidence freeze through Gen-20"
Cohesion: 0.50
Nodes (3): AMMC research evidence freeze through Gen-20, Next work package, Sanity check

### Community 244 - "Phase 44: calibrated SHD temporal baselines"
Cohesion: 0.50
Nodes (3): Decision, Phase 44: calibrated SHD temporal baselines, Result

### Community 245 - "Phase 45: learned spiking temporal convolution"
Cohesion: 0.50
Nodes (3): Decision, Phase 45: learned spiking temporal convolution, Result

### Community 246 - "Phase 46: SHD temporal-state placement"
Cohesion: 0.50
Nodes (3): Decision, Phase 46: SHD temporal-state placement, Result

### Community 248 - "Current implementation state"
Cohesion: 0.67
Nodes (3): Current implementation state, Gen-4 browser sandbox, Gen-5 backend scaffold

### Community 254 - "run_harness"
Cohesion: 0.47
Nodes (5): main(), Namespace, Path, query_graph(), run_harness()

### Community 255 - "sprint26_structural_sequential_mnist.py"
Cohesion: 0.47
Nodes (5): available_structural_sequential_arms(), main(), parse_args(), Namespace, Sprint 26 CLI: targeted synaptogenesis on row-sequential MNIST.

## Knowledge Gaps
- **1006 isolated node(s):** `_MissingModule`, `_MissingModule`, `_MissingModule`, `_MissingModule`, `_MissingModule` (+1001 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1709 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SHDConfig` connect `SHDConfig` to `resolve_device`, `shd_calibrated_baselines.py`, `shd_temporal_pyramid.py`, `ammc_gen5/gen6_successor.py`, `ammc_gen5/gen9_continual_adaptation.py`, `ammc_gen5/gen25_event_driven_sparse_audit.py`, `shd_capacity.py`, `ammc_gen5/gen28_triton_event_kernel.py`, `ammc_gen5/gen10_robust_representation.py`, `run_milestone_a`, `ammc_gen5/gen13_local_plasticity.py`, `shd_analog_topology.py`, `ammc_gen5/gen11_plastic_adapter.py`, `ammc_gen5/gen23_boundary_consolidation.py`, `ammc_gen5/gen19_nmnist_state_replication.py`, `ammc_gen5/gen24_compiled_residual_state.py`, `shd_representation.py`, `shd_matched_baselines.py`, `ammc_gen5/gen27_trained_threshold_robustness.py`, `ammc_gen5/gen12_associative_memory.py`, `__init__.py`, `ammc_gen5/gen7_predictive_state.py`, `shd_sparse_mechanisms.py`, `shd_temporal_controls.py`, `shd_sparse_width.py`, `device_kind`, `shd_initialization_robustness.py`, `ammc_gen5/gen21_matched_causal_mechanisms.py`, `ammc_gen5/gen8_temporal_binding.py`, `PredictiveStateTCNClassifier`, `SharedResidualStateTCNClassifier`, `shd_spiking_temporal_conv.py`, `shd_validation_checkpoint.py`, `TemporalDilatedTCNClassifier`, `load_ssc_tensors`, `ResidualTemporalConvStateClassifier`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `EventMNISTConfig` connect `EventMNISTConfig` to `sprint26_structural_sequential_mnist.py`, `AdaptiveSequentialClassifier`, `trainable_sequential_mnist.py`, `trainable_temporal_mnist.py`, `sync`, `__init__.py`, `event_mnist.py`, `FrozenEventReservoir`, `trainable_delays_mnist.py`, `StreamingMNISTReservoir`, `utility_gated_structural_mnist.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `sync()` connect `sync` to `resolve_device`, `TensorEvolver`, `trainable_sequential_mnist.py`, `trainable_temporal_mnist.py`, `sprint15_frozen_readout_adapter_robustness.py`, `EventMNISTConfig`, `ammc_gen5/gen16_local_score_credit.py`, `ammc_gen5/gen9_continual_adaptation.py`, `ammc_gen5/nmnist_accuracy_benchmark.py`, `TensorEnvironment2D`, `ammc_gen5/gen17_sparse_spiking_credit.py`, `ammc_gen5/gen25_event_driven_sparse_audit.py`, `tensor_environment.py`, `trainable_delays_mnist.py`, `ammc_gen5/gen15_reward_baseline.py`, `ammc_gen5/gen28_triton_event_kernel.py`, `ammc_gen5/gen10_robust_representation.py`, `sprint16_frozen_embodied_adapter.py`, `ammc_gen5/gen13_local_plasticity.py`, `ammc_gen5/gen11_plastic_adapter.py`, `benchmark_throughput.py`, `ammc_gen5/gen14_reward_eligibility.py`, `ammc_gen5/gen24_compiled_residual_state.py`, `ammc_gen5/gen12_associative_memory.py`, `__init__.py`, `ammc_gen5/gen7_predictive_state.py`, `SHDConfig`, `ammc_gen5/gen20_spiking_spatiotemporal.py`, `shd_initialization_robustness.py`, `ammc_gen5/gen21_matched_causal_mechanisms.py`, `utility_gated_structural_mnist.py`, `ammc_gen5/gen8_temporal_binding.py`, `top_fraction_spike_code`, `FrozenEventReservoir`, `FrozenReadoutAdapterRunner`, `sprint15_frozen_readout_adapter_generalization.py`, `shd_validation_checkpoint.py`, `event_mnist.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Are the 79 inferred relationships involving `SHDConfig` (e.g. with `Gen10RobustRepresentationResult` and `run_gen10_robust_representation()`) actually correct?**
  _`SHDConfig` has 79 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `EventMNISTConfig` (e.g. with `AdaptiveSequentialClassifier` and `AdaptiveSequentialMNISTResult`) actually correct?**
  _`EventMNISTConfig` has 44 INFERRED edges - model-reasoned connections that need verification._
- **What connects `_MissingModule`, `_MissingModule`, `_MissingModule` to the rest of the system?**
  _1006 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `resolve_device` be split into smaller, more focused modules?**
  _Cohesion score 0.061569416498993966 - nodes in this community are weakly interconnected._