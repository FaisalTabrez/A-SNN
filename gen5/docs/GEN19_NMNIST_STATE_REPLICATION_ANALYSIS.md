# Gen-19 N-MNIST residual-state replication analysis

Status (2026-08-10): completed with preregistered decision `stop`.

## Artifact provenance

The Colab terminal log reports that the full run completed and wrote JSON,
records CSV, summary CSV, plot, and progress JSON under
`/content/drive/MyDrive/A-SNN/gen5_outputs/gen19_nmnist_state_replication_cuda`.
Those original files were not retrieved. The aggregate terminal JSON is
preserved in
`gen5/outputs/gen19_nmnist_state_replication_log_recovery_2026-08-10/` with
the source-log SHA-256 and explicit recovery limitations. No missing per-seed
values were reconstructed.

## Result

The conventional temporal Conv1D learned the official N-MNIST split at
96.860% mean test accuracy. The parameter-matched residual LIF reached
96.317%, a gap of only -0.543 percentage point, and its 17.052% spike activity
was inside the frozen 1-30% interval.

The state-removal ablation was strongly positive: full residual accuracy was
15.210 points above direct-only accuracy, with all three seeds meeting the
registered effect threshold. The direct branch was also necessary: full
accuracy exceeded state-only accuracy by 19.507 points.

The sample-identity test failed in the opposite direction. Shuffling the LIF
state between samples increased accuracy from 96.317% to 98.617%, so the mean
full-minus-shuffled effect was -2.300 points and zero of three seeds met the
identity gate. State is therefore causally useful as a feature block in this
model, but the experiment does not show that its sample-specific content is
beneficial. A plausible interpretation is that it supplies a generic
calibration or regularization signal while its sample identity is a nuisance;
that interpretation is not itself proven by this experiment.

The residual model processed 41,046 test examples/s versus 147,509 for the
Conv1D, about 27.8% of the conventional throughput (the Conv1D was 3.59x
faster). This is a dense PyTorch measurement and not a neuromorphic energy
benchmark.

## Frozen decision

- Dataset learnability: pass.
- Matched accuracy: pass.
- State contribution: pass.
- Sample-specific state identity: **fail**.
- Spike activity: pass.
- Overall: **stop**.

The cross-modal claim is not supported. Causal, beneficial sample-specific
residual LIF state remains supported on the prior SHD and SSC event-audio
experiments, but Gen-19 does not extend that claim to N-MNIST event vision.
Local reward credit, STW/LTW, replay, and structural plasticity remain closed.

## Next decision

Do not tune Gen-19 after observing this result. The next work should be a
publication/evidence closeout that reports both the event-audio positive result
and the N-MNIST boundary condition. Any new architecture intended to solve
event-vision state identity must begin from a new hypothesis and a separately
frozen protocol rather than being presented as a rescue sweep.
