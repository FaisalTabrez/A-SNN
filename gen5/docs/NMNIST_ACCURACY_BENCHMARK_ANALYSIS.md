# N-MNIST accuracy benchmark analysis

Date: 2026-08-11  
Status: completed / both frozen accuracy gates passed  
Evidence: `gen5/outputs/nmnist_accuracy_benchmark_log_recovery_2026-08-11/`

## Result

The full-resolution spatial-temporal CNN achieved **99.4767% mean test
accuracy** across seeds 211–213. Its standard deviation was **0.0125
percentage points**, its weakest seed reached **99.46%**, and all three seeds
were evaluated from validation-selected checkpoints. This passes both the
preregistered 99.0% practical gate and the 99.4% stretch gate.

The frame CNN achieved **99.1233% mean test accuracy**. The extra temporal
modeling therefore bought **0.3533 percentage points**, but required 2.02× as
many parameters and 18.15× the dense MAC proxy. The measured frame-CNN
throughput was 11.31× higher. The spatial-temporal model is the accuracy winner;
the frame model is the clear efficiency winner under this implementation.

## What this proves—and what it does not

This run proves that the native 34×34 polarity-time representation and the
benchmark pipeline can support highly competitive N-MNIST accuracy. The result
is reproducible across the three preregistered confirmation seeds and is not a
single lucky checkpoint.

It does **not** establish an SNN record. ConvPLIF screened at 93.07% validation
accuracy, 5.63 points behind the 98.70% spatial-temporal screen result, and was
correctly not promoted under the frozen one-point rule. The confirmed 99.4767%
result belongs to a conventional spatial-temporal CNN. Direct energy efficiency
was not measured, and the MAC counts remain analytical proxies.

## Gen-20 handoff

The bounded accuracy side track is closed. Gen-20 should return to the core
research question with a separately preregistered **spiking spatial-temporal
translation** experiment: preserve the successful native event representation
and spatial-temporal receptive field while testing whether multi-timescale LIF
dynamics can close the observed 6.4-point screen gap without losing sparse
activity. The successful spatial-temporal CNN becomes a teacher/upper control,
not evidence for the SNN claim.

Gen-20 must remain one unified screen/confirmation package rather than another
open-ended phase chain. Minimum reporting should include accuracy, spike/event
density, parameter count, operation proxy, throughput, and direct state-removal
and time-order controls. Local plasticity, structural growth, and hardware-energy
claims remain gated until this representation problem is solved.

## Artifact limitation

The supplied log reports that the canonical JSON, CSVs, figure, manifest, and
checksummed bundle were saved in Google Drive. Only the log was supplied here,
so the repository contains an explicitly marked recovery with the exact reported
screen, confirmation, aggregate, dataset, and decision values. The canonical
`nmnist_accuracy_benchmark_bundle.zip` should be imported later for complete
per-epoch and timing provenance.
