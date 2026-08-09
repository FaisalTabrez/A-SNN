# Phase 48: residual-LIF replication on SSC

Archive SHA-256:
`2575EA0A0098C7E0CDF38AA97B69DD86D30AB68246D58A9A71C00FCE624C7477`

## Result

| Metric | Mean |
| --- | ---: |
| Residual-LIF test accuracy | 56.498% ± 0.360 pt |
| Matched Conv1D accuracy | 49.248% |
| Full gain over Conv1D | +7.250 pt |
| Direct-only accuracy | 45.226% |
| State-only accuracy | 6.216% |
| Shuffled-state accuracy | 53.518% |
| Full minus direct-only | +11.271 pt |
| Full minus shuffled state | +2.980 pt |
| Residual-LIF spike rate | 4.813% |

All three residual-LIF seeds are within two points of Conv1D; in fact, they
beat it by 5.662, 5.279, and 10.809 points. Removing state and shuffling its
sample identity each cost more than one point on all seeds. The causal
contribution therefore replicates on the complete official SSC splits.

State-only performance remains weak, so this is a cooperative hybrid result.
The 4.813% spike rate is much lower than the SHD rate, indicating that the
state effect does not require uniformly high firing activity.

## Sanity check

The evidence now supports complementary direct temporal and LIF-state features
on two event-audio datasets. It does not support standalone SNN, state-of-the-art,
or efficiency claims. Residual LIF runs at about 15,361 test examples/s versus
48,003 for Conv1D in dense PyTorch.

## Decision

Run one final matched audit against a two-layer dilated TCN and report measured
throughput plus clearly labeled operation proxies. Do not treat dense MAC
estimates as hardware energy measurements. After that audit, synthesize the
positive mechanism result and negative efficiency/standalone findings without
further architecture tuning.
