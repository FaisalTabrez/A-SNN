# AMMC Gen-5 through Gen-20 evidence report

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

Gen-16 derived the exact score-function update on a matched linear policy. The manual gradient matched autograd within 2.794e-09, and both policies finished at +0.300 with zero behavioral gap. The local rule gained +0.183, finished +0.249 above shuffled reward, and passed identity on 3/3 seeds. This validates analog linear local credit, not sparse spiking or memory.

Gen-17 translated that rule to one Bernoulli sensory event per channel and decision step. Event activity remained healthy at 6.369% during training and 12.078% during evaluation, while the manual gradient error remained 3.725e-09. Nevertheless, correct-reward spiking credit changed fitness by -0.391 and finished -1.052 relative to shuffled reward. The analog reference itself gained only +0.004 on the fresh seeds. Gen-17 therefore rejects this sparse translation and reopens analog-credit replication.

Gen-18 then held the analog rule fixed across ten untouched seeds. Correct reward improved mean fitness by +0.796 and finished +0.510 above shuffled reward. However, only 5/10 seeds met the gain gate and 6/10 met reward identity; the lower 95% bounds were -0.016 and -0.013. The local reward-credit program is therefore closed despite its positive mean.

Gen-19 transferred the frozen residual-state test to N-MNIST event vision. Conv1D reached 96.860% and residual LIF reached 96.317%, while removing state cost 15.210 points. However, shuffling state between samples improved accuracy by 2.300 points and zero of three seeds passed identity. The external replication therefore stopped: sample-specific residual-state benefit is supported on SHD/SSC event audio, not N-MNIST event vision.

Gen-20 then attempted to translate the successful dense N-MNIST spatial-temporal representation into multiscale residual PLIF state. The dense teacher screened at 99.117%, while ConvPLIF, multiscale PLIF, and distilled multiscale PLIF reached 96.216%, 96.366%, and 96.333%. The best new arm missed the frozen promotion gate by 1.134 points; distillation changed accuracy by -0.033 points. Its 12.692% activity and 74.37x operation proxy are operational strengths, not substitutes for the failed accuracy gate. No arm was promoted, so causal state and time-order controls were not tested.

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
| Gen-16 manual score-function gradient matches autograd | supported | Maximum analytic gradient error is 2.794e-09. |
| Gen-16 local reward credit is behaviorally equivalent to autograd | supported | Manual and autograd policies finish with a fitness gap of 0.000 per 1,000 steps. |
| Gen-16 local learning depends on agent-specific reward | supported | The local rule finishes +0.183 versus static and +0.249 versus shuffled reward on 3/3 identity-qualified seeds. |
| Gen-16 establishes sparse-spiking or structural continuous learning | not tested | Gen-16 uses a dense linear analog policy; spikes, STW/LTW, replay, and topology changes are absent. |
| Gen-17 sparse event generation and local gradient are operational | supported | Training/evaluation spike density is 6.369%/12.078% and maximum gradient error is 3.725e-09. |
| Gen-16 analog local-credit gain replicates on Gen-17 seeds | rejected | The analog reference gains +0.004 with 1/3 qualified seeds. |
| Gen-17 Bernoulli sparse translation preserves local learning | rejected | The correct-reward spiking policy gains -0.391 and trails the analog gain by 0.396 fitness per 1,000 steps. |
| Gen-17 sparse local learning depends on correctly assigned reward | rejected | Correct reward finishes -0.391 versus static and -1.052 versus shuffled reward. |
| Gen-18 stationary controls and manual-gradient implementation remain valid | supported | Static reset is exact, oracle fitness is +9.358, and maximum gradient error is 3.725e-09. |
| Gen-16 analog local-credit behavior replicates across ten held-out seeds | rejected | Mean gain is +0.796 with lower 95% bound -0.016 and 5/10 qualified seeds. |
| Gen-18 local behavior depends reliably on correctly assigned reward | rejected | Correct minus shuffled reward is +0.510 with lower 95% bound -0.013 and 6/10 qualified seeds. |
| The tested local reward-credit program qualifies for further mechanism expansion | rejected | Gen-18 returned status=stop and next_milestone=close_local_reward_credit_program. |
| Gen-19 establishes a learnable parameter-matched N-MNIST benchmark | supported | Conv1D reaches 96.860% and residual LIF reaches 96.317%. |
| Residual LIF state is causally used on N-MNIST | supported | Removing state costs 15.210 points on average with 3/3 qualifying seeds. |
| Residual LIF state is beneficially sample-specific on N-MNIST | rejected | Full minus shuffled-state accuracy is -2.300 points with 0/3 qualifying seeds. |
| The event-audio residual-state result generalizes to event vision | rejected | Gen-19 returned status=stop and next_milestone=limit_residual_state_claim_to_event_audio. |
| Gen-20 retains the strong dense N-MNIST spatial-temporal representation | supported | The dense teacher reaches 99.117% validation accuracy against the 97.5% screen gate. |
| Gen-20 multiscale residual PLIF closes the N-MNIST representation gap | rejected | The best new spiking arm reaches 96.366%, missing promotion by 1.134 points. |
| Gen-20 teacher distillation improves the multiscale spiking translation | rejected | Distillation changes validation accuracy by -0.033 points. |
| Gen-20 proposed arms maintain sparse activity and a low operation proxy | supported | The best arm has 12.692% activity and a 74.37x activity-scaled operation reduction versus the teacher. |
| Gen-20 establishes causal temporal state use on N-MNIST | not tested | No new arm passed the screen, so confirmation, state removal, and temporal-order controls did not run. |
| Gen-20 qualifies the program for an automatic Gen-21 architecture phase | rejected | Gen-20 returned status=stop, reason=no_new_spiking_arm_passed_screen, and next_milestone=evidence_synthesis. |

## Defensible contribution

The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary and beneficially sample-specific on two event-audio datasets. Gen-19 and Gen-20 define the event-vision boundary: state shuffling improves the former, while the latter's more ambitious sparse translation fails its promotion gate despite healthy activity and a low operation proxy. Later generations establish predictive alignment, partial analog order sensitivity, a valid damage-adaptation task, strong sensor-dropout robustness, conventional few-shot adaptation, a solvable embodied sensor-action control, a stationary delayed-reward protocol, and an exact manual score-function gradient. Frozen causal gates reject reliable behavioral replication of that local-credit rule, end-to-end spiking state, bounded state adapters, associative class prototypes, supervised three-factor output plasticity, the earlier reward-modulated eligibility rule, and the Gen-17 one-sample Bernoulli translation. Local continual learning, structural plasticity, dual memory, learned-delay benefit, sparse-spiking credit, and hardware energy remain unproven. These are qualified mechanism, protocol, boundary-condition, and negative-selection results—not a best-SNN, Transformer-replacement, continuous-learning, synaptic-memory, or hardware-efficiency result.

## Next-generation roadmap

| Priority | Workstream | Objective | Success measure |
| ---: | --- | --- | --- |
| 1 | publication_evidence_closeout | Package the supported event-audio mechanism and the Gen-19/20 event-vision boundary conditions. | A reproducible 22-source ledger reports positive, negative, and untested gates without post-hoc rescue. |
| 2 | matched_causal_mechanism_benchmark | Test one supported event-audio residual-state backbone with factorial adaptive-mechanism ablations. | Dynamic topology, dual memory, learned delays, and local reward credit are each compared under matched parameters, active operations, seeds, and optimization budgets. |
| 3 | event_vision_theory_reset | Require a genuinely new representation hypothesis before reopening N-MNIST state identity. | No Gen-19 or Gen-20 rescue sweep is labeled confirmatory evidence. |
| 4 | complex_plasticity_remains_gated | Keep strong continuous-learning and hardware-energy claims closed until factorial causal gates pass. | Each mechanism must add replicated task, adaptation, or retention value beyond matched static controls; energy requires direct measurement. |

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
- gen16: `gen5\outputs\gen16_local_score_credit_cuda_2026-08-10\gen16_local_score_credit.json`
- gen17: `gen5\outputs\gen17_sparse_spiking_credit_cuda_2026-08-10\gen17_sparse_spiking_credit.json`
- gen18: `gen5\outputs\gen18_local_credit_replication_cuda_2026-08-10\gen18_local_credit_replication.json`
- gen19: `gen5\outputs\gen19_nmnist_state_replication_log_recovery_2026-08-10\gen19_nmnist_state_replication.json`
- gen20: `gen5\outputs\gen20_spiking_spatiotemporal_cuda_2026-08-20\gen20_spiking_spatiotemporal.json`
