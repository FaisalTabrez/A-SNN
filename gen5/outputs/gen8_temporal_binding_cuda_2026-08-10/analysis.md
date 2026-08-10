# Gen-8 time-local predictive-binding analysis

Archive SHA-256:
`5D3087148BF735FE921E6895B06318B979F3371454095B57170F2B6286493E79`.

## Terminal result

The stored preregistered decision is `status=stop` with no qualified arms and
`next_milestone=close_gen8_temporal_binding`. The paired and shuffled
time-local LIF arms failed the screen and were never exposed to confirmatory
test selection. This is the intended operation of the promotion gate, not
missing data.

## Screening

| Arm | Validation | Difference from TCN | Activity | Outcome |
| --- | ---: | ---: | ---: | --- |
| dilated TCN | 38.100% | — | 33.169% ReLU | promoted |
| pooled predictive LIF | 37.200% | -0.900 point | 14.039% spikes | promoted |
| analog time-local binding | 43.000% | +4.900 points | 78.029% analog | promoted |
| shuffled-target time-local LIF | 10.133% | -27.967 points | 41.091% spikes | rejected |
| paired time-local LIF | 7.267% | -30.833 points | 50.656% spikes | rejected |

Both local LIF arms fail the accuracy screen by a wide margin and exceed the
registered 30% maximum spike rate. Their parameter ratios remain matched, so
the failure is not caused by an accidental capacity mismatch. The paired arm's
large binding activity (`0.6357`) also rules out an inactive correction; its
dynamics are active but destructive.

## Confirmation

- Pooled predictive LIF reaches `60.684% ± 0.283`, exceeding TCN
  (`59.149% ± 0.585`) by `+1.536` points. Its state removal cost is only
  `0.221` point with 0/3 seeds passing. Batch shuffling improves accuracy by
  `2.069` points and reversal costs only `0.203` point. Alignment remains high
  at `0.3094`. The representation is predictive, but its output use is again
  non-specific.
- Analog time-local binding reaches `58.692% ± 2.176`, `0.456` point below
  TCN. State removal costs `0.502` point but replicates on only 1/3 seeds.
  Reversal costs `0.561` point and replicates on 2/3 seeds, showing that local
  fusion can make output depend on order. State shuffling costs only `0.118`
  point and passes on 0/3 seeds, so correct sample identity remains absent.
- Throughput is `53,167` examples/s for TCN, `21,207` for analog binding
  (`0.399x`), and `14,071` for pooled LIF (`0.265x`). No hardware-efficiency
  claim is supported.

## Goal sanity check

The project now has reproducible evidence for predictive LIF representations
and partial evidence that pre-pooling local fusion introduces temporal-order
sensitivity in an analog state path. It still lacks a stable, parameter-matched
spiking candidate whose benefit depends on the correct sample identity and
temporal order. The results do not support a best-SNN, Transformer-alternative,
continuous-learning, or hardware-efficiency claim.

The Gen-8 hypothesis is closed according to its preregistration. Retuning the
horizon, temperature, loss weight, threshold, or spike gate would be a rescue
sweep and is not authorized by these results. The evidence-selected next step
is a final ten-source claim ledger and publication closeout. Any later model
generation requires a new task-level hypothesis and independent
preregistration.
