# AMMC Gen-5 through Gen-15 evidence report

## Executive conclusion

Across SHD and SSC, a residual LIF state carries sample-specific information that complements direct temporal convolution features. The mechanism is causal under feature-removal and shuffled-state tests, but it is neither standalone nor computationally competitive in the current dense PyTorch implementation.

The strongest matched SSC baseline, a dilated TCN, exceeds residual LIF by 3.253 accuracy points and runs at 3.18x its throughput. The residual model uses a lower dense-MAC proxy, but no hardware-energy claim is justified.

Milestone A then tested whether residual and hierarchical state models could clear a preregistered validation screen. Only the dilated TCN was promoted. Gen-6 subsequently preserved that predictor and added a zero-initialized, weight-shared LIF correction. It matched TCN accuracy within 0.065 points and learned a non-zero gate with healthy spiking, but removing state cost only 0.386 points and shuffled state improved accuracy by 0.657 points. Gen-6 therefore returned `stop` with no qualified causal arms.

The accuracy-preservation hypothesis is supported; the beneficial sample-specific-correction hypothesis is rejected. Hardware optimization remains closed under the preregistered rule.

Gen-7 then assigned state a paired future-prediction objective and a sample-conditioned gate. Paired LIF leads TCN by +0.417 points and its future-alignment margin reaches 0.2928, but shuffled state improves accuracy by 1.022 points and time reversal costs only 0.165 points. Representation learning succeeded; beneficial identity/order-specific use did not. Gen-7 returned `stop`.

Gen-8 moved prediction and fusion to aligned timesteps. Its paired local LIF candidate screened at 7.267% with a 50.656% spike rate and was not confirmed. The analog local binder remained within 0.456 TCN points and reversal cost 0.561 points, but state shuffling cost only 0.118 points. Local fusion introduces partial order sensitivity without beneficial identity-specific spiking use. Gen-8 returned `stop`.

Gen-9 then tested adaptation after a fixed 35% sensor-bank failure. The confirmed TCN shift was 9.364 points. A frozen TCN readout recovered 5.601 points, while full fine-tuning recovered 8.462 points and retained the source task better. However, predictive LIF trailed the TCN screen by 6.467 points and was not promoted. Gen-9 therefore returned `stop`; STW/LTW, replay, modulation, and structural plasticity remain closed.

Gen-10 tested masked-sensor residual state. Sensor dropout improved conventional clean and damaged accuracy by 2.684 and 8.763 points. Residual analog missed the dropout-TCN clean/damaged screen by 5.500/3.200 points; residual LIF missed by 9.200/6.733 points despite healthy spiking. Gen-10 returned `stop`.

Gen-11 froze that robust dropout-TCN backbone and adapted bounded downstream state. Full fine-tuning, readout adaptation, analog state, and LIF state recovered 3.295, 2.330, 1.353, and 0.783 points. Removing LIF state erased 0.783 points, but shuffling sample identity cost only 0.011 points. Gen-11 returned `stop`; synaptic STW/LTW remains closed.

Gen-12 replaced the adapter with context-gated associative prototypes. Full fine-tuning and readout adaptation recovered 4.767 and 3.564 points, while dense and spiking memories recovered only 0.250 and 0.278. Removing spiking memory cost 0.278 points and shuffling its class associations cost 0.417, despite the registered 20.0% event density. Gen-12 returned `stop`.

Gen-13 then localized supervised class-error credit to manual analog and spiking output-synapse updates. Full fine-tuning and autograd readout adaptation recovered 3.269 and 2.049 points. Analog and spiking local rules recovered only 0.420 and 0.410. Removing spiking fast weights cost 0.410 points and class shuffling cost 0.468, despite exactly 20.0% trace density and zero source forgetting. Gen-13 returned `stop`.

Gen-14 moved to delayed scalar reward in the embodied tensor world. The oracle reached 8.381 net fitness per 1,000 steps versus 0.641 for static behavior, confirming that the sensor-action task is solvable. Spiking eligibility improved relative to its own cold-start phase, but static behavior improved more over the same phase transition. The learned spiking policy finished -0.750 versus static and -0.161 versus shuffled reward. Activity remained healthy at 20.0% and weights did not saturate. Gen-14 returned `stop`: reward-specific local learning is rejected, and the baseline-to-evaluation rise is treated as phase non-stationarity.

Gen-15 rebuilt each baseline and final evaluation from identical seeded state. Static behavior reproduced exactly, while conventional correct-reward REINFORCE gained +0.992 fitness per 1,000 steps and finished +1.267 above agent-shuffled reward. The final mean remained -0.271 and the improvement was seed-sensitive. Gen-15 validates the delayed reward and identity protocol, not Gen-14 or an AMMC local-learning mechanism.

## Claim ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| Standalone state-only LIF is competitive on SHD | rejected | State-only LIF trails Conv1D by 8.613 points. |
| Residual LIF is viable on SHD | supported | Residual LIF changes accuracy by +0.942 points versus Conv1D. |
| Sample-specific LIF state contributes on SHD | supported | Removing state costs 6.419 points; shuffling state costs 4.167 points. |
| The residual-state contribution replicates on SSC | supported | Removing state costs 11.271 points; shuffling state costs 2.980 points. |
| Residual LIF matches the stronger SSC temporal baseline | rejected | Matched dilated TCN leads by 3.253 points. |
| Residual LIF is faster in the current T4 implementation | rejected | Residual LIF delivers 0.314x TCN throughput. |
| Residual LIF has a lower dense-operation proxy | proxy_only | Dense MAC proxy is 11.569% lower than TCN. |
| The current results establish hardware energy efficiency | not_tested | No direct power or energy measurement was performed; dense PyTorch is not event-driven. |
| Hierarchical residual scaling closes the SSC accuracy gap | rejected | Hierarchical analog and LIF trail TCN validation by 8.567 and 13.467 points. |
| The current Gen-5 architecture qualifies for hardware optimization | rejected | Milestone A promoted dilated_tcn and returned status=stop with 0 qualified causal arms. |
| The Gen-6 shared residual LIF preserves TCN predictive accuracy | supported | Shared residual LIF changes SSC accuracy by -0.065 points versus the matched TCN. |
| The Gen-6 LIF correction is beneficially sample-specific | rejected | Removing state costs 0.386 points (1/3 seeds pass), while shuffling state changes accuracy by +0.657 points in the shuffled model's favor (0/3 seeds pass). |
| The Gen-6 successor qualifies for hardware optimization | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-7 paired future prediction improves state alignment | supported | Paired LIF future alignment is 0.2928 versus -0.0017 under shuffled-target training. |
| Gen-7 paired predictive LIF matches TCN accuracy | supported | Paired predictive LIF changes accuracy by +0.417 points versus TCN. |
| Gen-7 uses predictive state beneficially by sample identity and temporal order | rejected | Shuffling state changes accuracy by +1.022 points in the shuffled model's favor; reversing state costs only 0.165 points. |
| The Gen-7 successor qualifies for hardware optimization | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-8 time-local analog binding introduces temporal-order sensitivity | supported | Reversing analog state costs 0.561 points with 2/3 seeds passing. |
| Gen-8 time-local analog binding uses the correct sample identity | rejected | Shuffling analog state costs only 0.118 points with 0/3 seeds passing. |
| The Gen-8 paired time-local LIF candidate is stable enough for confirmation | rejected | The candidate screened at 7.267% validation accuracy with a 50.656% spike rate. |
| The Gen-8 successor qualifies for hardware optimization | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-9 sensor damage creates a non-trivial distribution shift | supported | Static TCN accuracy falls by 9.364 points across confirmation seeds. |
| The Gen-9 predictive LIF representation is source-competent | rejected | Predictive LIF trails TCN screening validation by 6.467 points with 8.964% spike activity. |
| A frozen TCN representation adapts through a trainable readout | supported | Readout adaptation gains 5.601 points with 3/3 seeds passing, at 2.831 points forgetting. |
| Gen-9 qualifies for STW/LTW memory experiments | rejected | The terminal decision is status=stop with 0 qualified arms; only dilated_tcn passed source screening. |
| Sensor dropout improves conventional robustness in Gen-10 | supported | Dropout TCN changes clean accuracy by +2.684 points and damaged accuracy by +8.763 points. |
| The Gen-10 masked residual analog representation is source-competent | rejected | Residual analog trails dropout TCN screening by 5.500 clean and 3.200 damaged points. |
| The Gen-10 masked residual LIF representation is source-competent | rejected | Residual LIF trails dropout TCN screening by 9.200 clean and 6.733 damaged points with 11.538% spikes. |
| Gen-10 qualifies a spiking representation for adaptation | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-11 state adapters improve damaged-task accuracy by the preregistered margin | rejected | The analog and LIF adapters gain 1.353 and 0.783 points, versus 2.330 for readout adaptation. |
| Gen-11 LIF adaptation depends on sample-specific spiking state | rejected | Removing LIF state costs 0.783 points, but shuffling sample identity costs only 0.011 points. |
| Gen-11 qualifies for synaptic STW/LTW consolidation | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-12 prototype memory provides useful fast adaptation | rejected | Dense and spiking prototypes gain 0.250 and 0.278 points, versus 3.564 for readout adaptation. |
| Gen-12 spiking memory depends on correct class associations | rejected | Removing memory costs 0.278 points and shuffling class associations costs 0.417 points, with 20.000% event density. |
| Gen-12 qualifies for context-free consolidation | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-13 local output plasticity provides useful adaptation | rejected | Analog and spiking local rules gain 0.420 and 0.410 points, versus 2.049 for autograd readout adaptation. |
| Gen-13 spiking fast weights are causally class-specific | rejected | Removing spiking fast weights costs 0.410 points and shuffling output classes costs 0.468 points, with 20.000% trace density. |
| Gen-13 qualifies for STW/LTW consolidation | rejected | The terminal decision is status=stop with 0 qualified arms. |
| Gen-14 embodied sensor-to-action mapping is solvable | supported | The oracle reaches 8.381 versus 0.641 static net fitness per 1,000 steps. |
| Gen-14 baseline-to-evaluation improvement identifies local learning | rejected | Spiking eligibility rises by 2.798, but the unchanged static arm rises by 3.983; the phase comparison is non-stationary. |
| Gen-14 spiking eligibility depends on correctly assigned reward | rejected | Correctly rewarded spiking eligibility finishes -0.750 versus static and -0.161 versus shuffled reward. |
| Gen-14 qualifies for reward-eligibility confirmation | rejected | The terminal decision is status=stop; next_milestone=close_reward_eligibility_screen. |
| Gen-15 identical-reset evaluation removes phase non-stationarity | supported | The unchanged static policy has exactly +0.000 fitness gain under replayed seeded evaluation. |
| Gen-15 delayed scalar reward supports conventional learning | supported | Correct-reward REINFORCE gains +0.992 fitness per 1,000 steps on 2/3 positive-gain seeds. |
| Gen-15 conventional learning depends on agent-specific reward | supported | Correct reward finishes +0.992 versus static and +1.267 versus shuffled reward. |
| Gen-15 validates an AMMC local-learning mechanism | not tested | Gen-15 tests only a conventional autograd REINFORCE baseline; final mean fitness remains -0.271 and no local AMMC rule is present. |

## Defensible contribution

The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. Later generations establish predictive alignment, partial analog order sensitivity, a valid damage-adaptation task, strong sensor-dropout robustness, conventional few-shot adaptation, a solvable embodied sensor-action control, and a stationary delayed-reward protocol that supports weak identity-specific conventional learning. Frozen causal gates reject end-to-end spiking state, bounded state adapters, associative class prototypes, supervised three-factor output plasticity, and reward-modulated eligibility as currently implemented. These are qualified mechanism, protocol, and negative-selection results—not a best-SNN, Transformer-replacement, continuous-learning, synaptic-memory, or hardware-efficiency result.

## Next-generation roadmap

| Priority | Workstream | Objective | Success measure |
| ---: | --- | --- | --- |
| 1 | gen15_reward_protocol_closeout | Package the stationary embodied reward diagnostic and its seed-level limitations. | A reproducible 17-source ledger whose claims distinguish protocol learning from local learning. |
| 2 | publication_package | Report the supported mechanism chain through Gen-15 without architecture-superiority claims. | Exact protocols, seeds, checkpoints, causal controls, and negative gates are publication-ready. |
| 3 | gen16_local_score_equivalence | Test the exact local score-function rule against a matched autograd policy before adding spikes. | Gradient parity, behavioral equivalence, and reward identity all pass under the frozen Gen-16 gate. |
| 4 | complex_plasticity_remains_gated | Keep spiking translation, STW/LTW, replay, and structural plasticity closed until Gen-16 passes. | No complex biological mechanism is added before local credit is mathematically and behaviorally validated. |

## Evidence sources

- phase44: `gen5\outputs\shd_calibrated_baselines_cuda_2026-08-10\shd_calibrated_baselines.json`
- phase45: `gen5\outputs\shd_spiking_temporal_conv_cuda_2026-08-10\shd_spiking_temporal_conv.json`
- phase46: `gen5\outputs\shd_state_placement_diagnostic_cuda_2026-08-10\shd_state_placement_diagnostic.json`
- phase47: `gen5\outputs\shd_residual_state_contribution_cuda_2026-08-10\shd_residual_state_contribution.json`
- phase48: `gen5\outputs\ssc_residual_lif_replication_cuda_2026-08-10\ssc_residual_lif_replication.json`
- phase49: `gen5\outputs\ssc_efficiency_baselines_cuda_2026-08-10\ssc_efficiency_baselines.json`
- milestone_a: `gen5\outputs\milestone_a_architecture_cuda_2026-08-10\milestone_a_architecture.json`
- gen6: `gen5\outputs\gen6_successor_cuda_2026-08-10\gen6_successor.json`
- gen7: `gen5\outputs\gen7_predictive_state_cuda_2026-08-10\gen7_predictive_state.json`
- gen8: `gen5\outputs\gen8_temporal_binding_cuda_2026-08-10\gen8_temporal_binding.json`
- gen9: `gen5\outputs\gen9_continual_adaptation_cuda_2026-08-10\gen9_continual_adaptation.json`
- gen10: `gen5\outputs\gen10_robust_representation_cuda_2026-08-10\gen10_robust_representation.json`
- gen11: `gen5\outputs\gen11_plastic_adapter_cuda_2026-08-10\gen11_plastic_adapter.json`
- gen12: `gen5\outputs\gen12_associative_memory_cuda_2026-08-10\gen12_associative_memory.json`
- gen13: `gen5\outputs\gen13_local_plasticity_cuda_2026-08-10\gen13_local_plasticity.json`
- gen14: `gen5\outputs\gen14_reward_eligibility_cuda_2026-08-10\gen14_reward_eligibility.json`
- gen15: `gen5\outputs\gen15_reward_baseline_cuda_2026-08-10\gen15_reward_baseline.json`
