# AMMC Gen-5/Gen-6 evidence report

## Executive conclusion

Across SHD and SSC, a residual LIF state carries sample-specific information that complements direct temporal convolution features. The mechanism is causal under feature-removal and shuffled-state tests, but it is neither standalone nor computationally competitive in the current dense PyTorch implementation.

The strongest matched SSC baseline, a dilated TCN, exceeds residual LIF by 3.253 accuracy points and runs at 3.18x its throughput. The residual model uses a lower dense-MAC proxy, but no hardware-energy claim is justified.

Milestone A then tested whether residual and hierarchical state models could clear a preregistered validation screen. Only the dilated TCN was promoted. Gen-6 subsequently preserved that predictor and added a zero-initialized, weight-shared LIF correction. It matched TCN accuracy within 0.065 points and learned a non-zero gate with healthy spiking, but removing state cost only 0.386 points and shuffled state improved accuracy by 0.657 points. Gen-6 therefore returned `stop` with no qualified causal arms.

The accuracy-preservation hypothesis is supported; the beneficial sample-specific-correction hypothesis is rejected. Hardware optimization remains closed under the preregistered rule.

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

## Defensible contribution

The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. Milestone A does not support retaining that implementation as a competitive architecture. Gen-6 demonstrates that a zero-initialized shared correction can preserve the conventional predictor, but its learned state is not beneficially sample-specific under shuffling. This is a causal mechanism result accompanied by two negative architecture-selection results, not a best-SNN, Transformer-replacement, or hardware-efficiency result.

## Next-generation roadmap

| Priority | Workstream | Objective | Success measure |
| ---: | --- | --- | --- |
| 1 | gen6_terminal_closeout | Package predictive parity together with the failed state-specificity gate. | A reproducible final ledger whose claims match the Gen-6 stop decision. |
| 2 | publication_package | Report the supported cross-dataset Gen-5 mechanism and negative Gen-6 successor result without architecture-superiority claims. | Exact protocols, seeds, checkpoints, causal controls, and negative gates are publication-ready. |
| 3 | hardware_work_deferred | Keep event-driven kernel optimization closed after the Gen-6 terminal failure. | No hardware-efficiency claim is pursued for an architecture with no qualified causal arm. |
| 4 | new_program_requires_new_hypothesis | Prevent an automatic Gen-7 rescue sweep on the rejected shared-residual design. | Any future generation starts from a separately approved hypothesis and preregistration. |

## Evidence sources

- phase44: `gen5\outputs\shd_calibrated_baselines_cuda_2026-08-10\shd_calibrated_baselines.json`
- phase45: `gen5\outputs\shd_spiking_temporal_conv_cuda_2026-08-10\shd_spiking_temporal_conv.json`
- phase46: `gen5\outputs\shd_state_placement_diagnostic_cuda_2026-08-10\shd_state_placement_diagnostic.json`
- phase47: `gen5\outputs\shd_residual_state_contribution_cuda_2026-08-10\shd_residual_state_contribution.json`
- phase48: `gen5\outputs\ssc_residual_lif_replication_cuda_2026-08-10\ssc_residual_lif_replication.json`
- phase49: `gen5\outputs\ssc_efficiency_baselines_cuda_2026-08-10\ssc_efficiency_baselines.json`
- milestone_a: `gen5\outputs\milestone_a_architecture_cuda_2026-08-10\milestone_a_architecture.json`
- gen6: `gen5\outputs\gen6_successor_cuda_2026-08-10\gen6_successor.json`
