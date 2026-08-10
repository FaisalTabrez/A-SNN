# Gen-11 plastic state-adapter analysis

Status (2026-08-10): terminal `stop`; zero qualified arms.

## Result

The fixed 35% sensor failure remained a valid shift: the frozen sensor-dropout
TCN lost 3.910 accuracy points. Across seeds 154–156, full fine-tuning recovered
3.295 points and classifier-only adaptation recovered 2.330 points. The analog
and LIF state adapters recovered only 1.353 and 0.783 points, respectively, so
neither reached the preregistered two-point adaptation threshold.

The LIF adapter retained source accuracy and maintained a healthy 16.782% spike
rate. Removing its correction erased 0.783 points, but batch-shuffling state
identity changed accuracy by only 0.011 point. Analog state showed the same
pattern: 1.353 points removed versus 0.018 point shuffled. State is therefore
being used as a generic correction, not as a beneficial sample-specific signal.

## Sanity check against the project goal

Supported:

- sensor dropout provides a source-competent and damage-robust sensory backbone;
- conventional readout and full-model adaptation recover accuracy;
- a frozen backbone can preserve prior-task performance during downstream adaptation;
- the LIF adapter is active and trainable.

Not supported:

- a parametric analog or LIF adapter matches conventional adaptation;
- adapter improvement depends on the identity of the current sample;
- Gen-11 demonstrates continuous synaptic learning;
- STW/LTW, replay, neuromodulation, or structural plasticity should be opened.

This narrows the bottleneck: adaptation requires an explicitly sample-indexed
memory mechanism, not another pooled correction layer. Gen-12 therefore starts
a separate associative-memory hypothesis rather than sweeping Gen-11 width,
gate, leak, threshold, mask fraction, or learning rate.

## Evidence

- Archive SHA-256: `150d6d350ed14c55a3eed44e4980264e5d3f460b6404ad35653017870a488a24`
- Extracted payload: `gen5/outputs/gen11_plastic_adapter_cuda_2026-08-10/`
- Frozen protocol: `gen5/docs/GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md`
