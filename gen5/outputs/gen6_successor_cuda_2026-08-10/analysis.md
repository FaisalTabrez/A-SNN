# Gen-6 successor analysis

## Provenance

- Source archive: `gen6_successor_cuda-20260810T044712Z-1-001.zip`
- Archive SHA-256: `F644395B407CCA4A33B820EE34C62C729CFE1B4EBF24949E5524C6FA74AF83CD`
- Runtime: Colab CUDA
- Confirmation: complete official SSC splits, 15 epochs, seeds 142–144

## Screen

All three arms passed the reduced-data promotion screen. The shared residual
LIF candidate reached 39.767% descriptive test accuracy, ahead of TCN at
39.233%, with a healthy 13.063% spike rate and a non-zero 0.0939 mean absolute
gate. This justified full confirmation; it was not the terminal result.

## Confirmation

| Arm | Test accuracy | State removal | State shuffle | Activity | Gate | Examples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dilated TCN | 59.082% ± 0.208 | — | — | 39.260% ReLU | — | 54,966 |
| Shared residual LIF | 59.016% ± 0.159 | +0.386 points | -0.657 points | 6.020% spikes | 0.1898 | 15,162 |
| Shared residual analog | 58.684% ± 0.390 | +21.651 points | -4.345 points | 87.878% analog | 0.2009 | 21,627 |

The LIF model is only 0.065 accuracy points behind TCN, so the predictive
parity gate passes. Its spike rate and learned gate also pass. However:

- removing the LIF correction costs only 0.386 mean points, below the required
  0.5, and only one of three seeds clears that threshold;
- shuffling state identity *improves* mean accuracy by 0.657 points, so the
  required state-specific loss fails on all three seeds;
- the current dense implementation achieves 0.276x TCN throughput, meaning
  TCN is approximately 3.63x faster.

The analog control reinforces the diagnosis: it has a large aggregate state
contribution, but shuffled state improves accuracy even more strongly. The
correction branches learned non-trivial signals, yet those signals are not
beneficially tied to the corresponding sample.

## Gate decision

The stored decision is `status=stop`, `best_arm=dilated_tcn`, zero qualified
arms, and `next_milestone=close_gen6_successor`. This exactly follows the
preregistered terminal rule.

## Sanity conclusion

Gen-6 solved the Gen-5 representation-preservation problem: a zero-initialized,
weight-shared LIF correction can retain TCN-level predictive performance. It
did not establish that the correction provides beneficial sample-specific
temporal state. Therefore the defensible result is predictive parity plus a
negative causal-specificity finding—not a superior SNN, a hardware-efficiency
result, or a reason to reopen the hardware milestone.

No rescue sweep or automatic Gen-7 experiment is authorized. The gate-selected
next work is final evidence synthesis and publication-oriented closeout. Any
new generation must begin from a genuinely different, separately approved and
preregistered hypothesis.
