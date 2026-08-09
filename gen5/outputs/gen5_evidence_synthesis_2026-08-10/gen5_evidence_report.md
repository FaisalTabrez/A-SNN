# AMMC Gen-5 evidence report

## Executive conclusion

Across SHD and SSC, a residual LIF state carries sample-specific information that complements direct temporal convolution features. The mechanism is causal under feature-removal and shuffled-state tests, but it is neither standalone nor computationally competitive in the current dense PyTorch implementation.

The strongest matched SSC baseline, a dilated TCN, exceeds residual LIF by 3.253 accuracy points and runs at 3.18x its throughput. The residual model uses a lower dense-MAC proxy, but no hardware-energy claim is justified.

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

## Defensible contribution

The supported contribution is a residual temporal architecture in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. This is an architectural mechanism result, not a best-SNN, Transformer-replacement, or hardware-efficiency result.

## Next-generation roadmap

| Priority | Workstream | Objective | Success measure |
| ---: | --- | --- | --- |
| 1 | compiled_event_driven_kernel | Replace the Python LIF loop and dense state execution with compiled event-driven kernels. | Measured accelerator or neuromorphic energy and latency, not MAC estimates. |
| 2 | accuracy_scaling | Combine residual state with a stronger hierarchical temporal front end. | Close the matched TCN gap without losing causal state contribution. |
| 3 | non_audio_replication | Test the mechanism on an independent event-vision or control dataset. | Replicate removal and shuffled-state gates outside event audio. |
| 4 | continual_plasticity_reintegration | Reintroduce LTW/STW and structural plasticity into the validated residual architecture. | Adaptation with retention under preregistered continual-learning baselines. |

## Evidence sources

- phase44: `gen5\outputs\shd_calibrated_baselines_cuda_2026-08-10\shd_calibrated_baselines.json`
- phase45: `gen5\outputs\shd_spiking_temporal_conv_cuda_2026-08-10\shd_spiking_temporal_conv.json`
- phase46: `gen5\outputs\shd_state_placement_diagnostic_cuda_2026-08-10\shd_state_placement_diagnostic.json`
- phase47: `gen5\outputs\shd_residual_state_contribution_cuda_2026-08-10\shd_residual_state_contribution.json`
- phase48: `gen5\outputs\ssc_residual_lif_replication_cuda_2026-08-10\ssc_residual_lif_replication.json`
- phase49: `gen5\outputs\ssc_efficiency_baselines_cuda_2026-08-10\ssc_efficiency_baselines.json`
