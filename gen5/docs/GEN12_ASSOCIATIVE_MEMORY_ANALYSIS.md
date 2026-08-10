# Gen-12 associative-memory analysis

Status (2026-08-10): terminal `stop`; zero qualified arms.

## Result

The frozen sensor-dropout TCN lost 5.158 points under the fixed 35% sensor
failure. Across seeds 157–159, classifier-only adaptation recovered 3.564
points and full fine-tuning recovered 4.767 points, each improving on all three
seeds.

Dense and spiking prototype memories recovered only 0.250 and 0.278 point.
The spiking memory held the registered 20% event density, stored 16,800 active
cells, incurred zero source forgetting through explicit context gating, and
required no trainable parameters. These healthy diagnostics isolate the
failure to associative usefulness rather than inactivity or missing storage.

Removing spiking memory cost 0.278 point and shuffling class associations cost
0.417 point. Neither intervention reached 0.5 point on any seed. Dense memory
showed the same pattern. One class-average prototype therefore collapses too
much task-relevant variation to replace synaptic credit assignment.

## Goal sanity check

Supported:

- the frozen robust backbone remains source competent;
- ordinary output-synapse adaptation reliably repairs sensor damage;
- a context gate can eliminate source forgetting;
- sparse event-coded memory is technically stable and inexpensive to update.

Not supported:

- prototype retrieval provides useful fast adaptation;
- Gen-12 demonstrates autonomous context recognition;
- associative prototypes justify consolidation, replay, or structural plasticity;
- the current system is a competitive brain-like continual learner.

The positive control is now unambiguous: useful adaptation occurs when the
classifier weights receive task-specific credit. Gen-13 therefore tests
whether that credit can be implemented as an explicit local three-factor
synaptic rule, without backpropagating through the frozen sensory backbone.

## Evidence

- Archive SHA-256: `7121cbd458b4f1a10736d6c8ac3e1b0a2d02f2b97c25d85897c32eda2f109717`
- Extracted payload: `gen5/outputs/gen12_associative_memory_cuda_2026-08-10/`
- Frozen protocol: `gen5/docs/GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md`
