# AMMC Gen-5/Gen-6/Gen-7/Gen-8 evidence report

## Executive conclusion

Across SHD and SSC, a residual LIF state carries sample-specific information that complements direct temporal convolution features. The mechanism is causal under feature-removal and shuffled-state tests, but it is neither standalone nor computationally competitive in the current dense PyTorch implementation.

The strongest matched SSC baseline, a dilated TCN, exceeds residual LIF by 3.253 accuracy points and runs at 3.18x its throughput. The residual model uses a lower dense-MAC proxy, but no hardware-energy claim is justified.

Milestone A then tested whether residual and hierarchical state models could clear a preregistered validation screen. Only the dilated TCN was promoted. Gen-6 subsequently preserved that predictor and added a zero-initialized, weight-shared LIF correction. It matched TCN accuracy within 0.065 points and learned a non-zero gate with healthy spiking, but removing state cost only 0.386 points and shuffled state improved accuracy by 0.657 points. Gen-6 therefore returned `stop` with no qualified causal arms.

The accuracy-preservation hypothesis is supported; the beneficial sample-specific-correction hypothesis is rejected. Hardware optimization remains closed under the preregistered rule.

Gen-7 then assigned state a paired future-prediction objective and a sample-conditioned gate. Paired LIF leads TCN by +0.417 points and its future-alignment margin reaches 0.2928, but shuffled state improves accuracy by 1.022 points and time reversal costs only 0.165 points. Representation learning succeeded; beneficial identity/order-specific use did not. Gen-7 returned `stop`.

Gen-8 moved prediction and fusion to aligned timesteps. Its paired local LIF candidate screened at 7.267% with a 50.656% spike rate and was not confirmed. The analog local binder remained within 0.456 TCN points and reversal cost 0.561 points, but state shuffling cost only 0.118 points. Local fusion introduces partial order sensitivity without beneficial identity-specific spiking use. Gen-8 returned `stop`.

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

## Defensible contribution

The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. Gen-6 shows that zero-initialized shared correction preserves the conventional predictor. Gen-7 shows strong future-aligned predictive state without beneficial identity-specific output use. Gen-8 adds evidence that pre-pooling local fusion can create temporal-order sensitivity in an analog state path, while its paired LIF candidate is unstable and sample identity remains non-causal. These are qualified mechanism and representation results accompanied by negative architecture-selection decisions, not a best-SNN, Transformer-replacement, or hardware-efficiency result.

## Next-generation roadmap

| Priority | Workstream | Objective | Success measure |
| ---: | --- | --- | --- |
| 1 | gen8_terminal_closeout | Package predictive alignment and analog order sensitivity together with the failed LIF screen and identity gate. | A reproducible final ledger whose claims match the Gen-8 stop decision. |
| 2 | publication_package | Report the supported Gen-5 mechanism, Gen-6 parity, Gen-7 predictive representation, and Gen-8 partial analog order result without architecture-superiority claims. | Exact protocols, seeds, checkpoints, causal controls, and negative gates are publication-ready. |
| 3 | hardware_work_deferred | Keep event-driven kernel optimization closed after the Gen-8 terminal failure. | No hardware-efficiency claim is pursued for an architecture with no qualified causal arm. |
| 4 | new_program_requires_new_hypothesis | Prevent an automatic LIF stabilization or temporal-binding rescue sweep on Gen-8. | Any future generation starts from a separately approved hypothesis and preregistration. |

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
