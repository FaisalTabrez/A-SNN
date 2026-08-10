# Gen-7 predictive-state analysis

## Provenance

- Source archive: `gen7_predictive_state_cuda-20260810T080026Z-1-001.zip`
- Archive SHA-256: `5AF3B42A569EADEB5CA56E7E33005334E73D93820B298BA77E4850A62DFB67F0`
- Runtime: Colab CUDA
- Confirmation: complete official SSC splits, 15 epochs, seeds 142–144

## Screen

All five registered arms were promoted. The paired predictive LIF candidate
led the screen at 45.033% validation and 43.767% descriptive test accuracy,
ahead of TCN at 38.300% validation and 39.033% test. It maintained 12.920%
spiking, a 0.1752 mean sample-gate magnitude, and strong positive future
alignment. The promotion therefore exercised the intended automatic controls
rather than rescuing a failed screen.

## Confirmation

| Arm | Test accuracy | vs TCN | State removal | State shuffle | Time reversal | Future alignment | Activity | Examples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired predictive LIF | 58.807% ± 1.093 | +0.417 | +1.009 | -1.022 | +0.165 | 0.2928 | 7.204% spikes | 16,004 |
| Dilated TCN | 58.390% ± 1.848 | — | — | — | — | — | 38.406% ReLU | 53,556 |
| LIF without prediction | 57.583% ± 1.567 | -0.806 | +1.117 | -0.118 | +0.219 | -0.0033 | 8.588% spikes | 14,606 |
| Shuffled-target predictive LIF | 57.160% ± 3.130 | -1.230 | +1.634 | -0.402 | +0.545 | -0.0017 | 7.884% spikes | 13,876 |
| Paired predictive analog | 56.625% ± 0.056 | -1.765 | +17.530 | -0.852 | +3.608 | 0.2334 | 78.245% analog | 18,751 |

The Gen-7 candidate passes several important gates:

- it exceeds TCN by 0.417 mean test point;
- state removal costs 1.009 points and passes on two of three seeds;
- future alignment is 0.2928 on all three seeds and exceeds shuffled-target
  training by 0.2945;
- spike activity and sample-gate magnitude are non-degenerate.

It fails both identity/order gates:

- batch-shuffling state *improves* accuracy by 1.022 mean points, and no seed
  passes the required specificity loss;
- time reversal costs only 0.165 mean point, and no seed passes the registered
  0.5-point threshold.

The shuffled-state result is consistent across the candidate's three seeds:
changes are -1.128, -0.064, and -1.874 points. This is not a single-seed sign
error. The paired prediction objective successfully learns sample identity in
its auxiliary space, but the output correction does not use that identity
beneficially. The likely bottleneck has moved from state representation to the
pooled additive decoder/interface.

## Gate decision and sanity check

The stored decision is `status=stop`, `best_arm=lif_paired_predictive`, zero
qualified arms, and `next_milestone=close_gen7_predictive_state`. This follows
the preregistered protocol.

The scientifically defensible conclusion is narrower than architecture
success: paired future prediction creates a strong state representation and
can improve mean predictive accuracy, but it does not establish beneficial
sample-specific or temporally ordered use of that state. The current dense LIF
path is also only 0.299x TCN throughput.

No loss-weight, horizon, threshold, or decoder rescue sweep is authorized.
The gate-selected next work is an updated reproducible evidence ledger and
research closeout. A new temporal-binding architecture would require a
separately approved and preregistered generation.
