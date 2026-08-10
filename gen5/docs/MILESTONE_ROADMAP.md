# Gen-5 milestone roadmap

> Status (2026-08-10): Milestone A returned `stop` with no qualified causal
> arms. The current architecture branch is closed. Milestones B and C are
> deferred; they must not be run against this rejected architecture.

> Gen-6 update (2026-08-10): the separately preregistered shared-residual
> successor preserved TCN accuracy but failed its state-removal and
> shuffled-state causal gates. It also returned `stop` with zero qualified
> arms. Hardware and continual-learning milestones remain closed. The active
> workstream is evidence/publication closeout, not an automatic Gen-7 rescue.

> Gen-7 update: the user explicitly authorized a new mechanism-level
> hypothesis after the Gen-6 closeout. Gen-7 is not a rescue sweep: it assigns
> state a paired future-prediction objective and replaces the static class gate
> with sample-conditioned direct/state interaction. Its independent terminal
> protocol is frozen in `GEN7_PREDICTIVE_STATE_PREREGISTRATION.md`; the hardware
> milestone remains closed until that protocol returns `pass`.

> Gen-7 result (2026-08-10): paired predictive LIF learned strong future
> alignment and led TCN mean accuracy, but shuffled state improved accuracy and
> temporal reversal had negligible cost. The terminal gate returned `stop`
> with zero qualified arms. Hardware work remains closed. A temporal-binding
> decoder is a possible new hypothesis, not an authorized rescue phase.

> Gen-8 update (2026-08-10): the user explicitly authorized that independent
> hypothesis. Gen-8 binds direct and state traces at matched timesteps before
> aggregation and trains prediction at matched future timesteps. The frozen
> arms and terminal gate are in `GEN8_TEMPORAL_BINDING_PREREGISTRATION.md`.
> Runtime/hardware work remains closed unless this new gate passes.

> Gen-8 result (2026-08-10): the paired time-local LIF candidate failed its
> screen with 7.267% validation accuracy and a 50.656% spike rate. The analog
> binder gained replicated order sensitivity but not identity specificity.
> The stored decision is `stop`; hardware and automatic successor work remain
> closed. The active phase is final evidence closeout.

> Gen-9 update (2026-08-10): after closing the static architecture branch, the
> user authorized a new continual-learning program aligned with AMMC's original
> goal. Its first milestone tests predictive-LIF versus TCN adaptation under a
> deterministic 35% sensor-bank failure. STW/LTW, modulation, replay, and
> structural plasticity remain gated behind this representation-level result.
> The protocol is frozen in `GEN9_CONTINUAL_ADAPTATION_PREREGISTRATION.md`.

> Gen-9 result (2026-08-10): the shift was valid and both TCN adaptation
> controls improved damaged-task accuracy. Predictive LIF nevertheless trailed
> TCN screening validation by 6.467 points and was not promoted. The terminal
> decision is `stop` with zero qualified arms. STW/LTW, replay, modulation,
> structural plasticity, and automatic damage sweeps remain closed. Any new
> continual-learning program requires a separately preregistered,
> source-competent representation hypothesis.

> Gen-10 update (2026-08-10): the user authorized that representation reset.
> Gen-10 compares ordinary and sensor-dropout TCN controls against masked-
> sensor residual analog and LIF state. The state candidates receive a
> parameter-free clean-target alignment objective. Clean/damaged parity and
> replicated state-removal and state-shuffling costs are required before a new
> adaptation experiment can open. Memory mechanisms remain closed. The frozen
> protocol is `GEN10_ROBUST_REPRESENTATION_PREREGISTRATION.md`.

> Gen-10 result (2026-08-10): sensor dropout strongly improved conventional
> clean and damaged TCN accuracy, but residual analog and LIF missed the frozen
> promotion margins. LIF activity and parameter matching were healthy. The
> stored decision is `stop`; no causal state confirmation was run. The user
> authorized a new functional-separation hypothesis rather than a Gen-10
> hyperparameter sweep.

> Gen-11 update (2026-08-10): the newly authorized hypothesis freezes the
> proven sensor-dropout TCN as a sensory backbone and compares conventional
> readout/full adaptation with matched analog and LIF correction adapters. A
> zero correction gate preserves identical source behavior. Only matched
> adaptation/retention plus replicated state-removal and state-shuffling costs
> can open STW/LTW. The frozen protocol is
> `GEN11_PLASTIC_ADAPTER_PREREGISTRATION.md`.

Gen-5 no longer advances through one experiment per numbered phase. Work is
grouped into three decision milestones. Each milestone owns a cheap screen, an
automatic promotion rule, a confirmatory run, causal controls, and a terminal
decision. Unpromising branches stop inside the same run.

## Milestone A — accuracy and architecture

Question: can a parameter-matched AMMC residual-state architecture retain a
causal spiking-state contribution while remaining within two SSC test points
of the strongest conventional temporal control?

The unified runner screens five arms on one seed and reduced official SSC
subsets:

- Conv1D and dilated TCN controls;
- the established residual LIF model;
- a hierarchical residual analog control;
- a hierarchical residual LIF candidate.

Only the best conventional control and causal candidates within two validation
points, within 95–105% of the parameter budget, and with non-degenerate LIF
activity are promoted. Confirmation uses all official splits and three seeds.
Full, direct-only, state-only, and batch-shuffled-state inference are evaluated
from each selected causal checkpoint without retraining.

Pass requires a causal LIF arm to:

- remain within two mean test points of the best conventional model;
- lose at least one mean point without state;
- lose at least one mean point when state identity is shuffled;
- reproduce both causal losses on at least two of three seeds;
- maintain a 1–30% mean spike rate when the arm is spiking.

No arm passed. The dilated TCN was the sole promoted model, so the architecture
branch is closed and no rescue threshold sweep is authorized. A future
successor must begin with a separate preregistration and a genuinely new causal
hypothesis.

That successor is now preregistered separately in
[`GEN6_SUCCESSOR_PREREGISTRATION.md`](GEN6_SUCCESSOR_PREREGISTRATION.md). It
does not change the terminal interpretation of Gen-5 evidence.

## Milestone B — runtime and hardware

Question: does compiled event-driven execution convert the existing operation
proxy into realized latency, throughput, memory, or energy benefits?

Screen kernel layouts and batch sizes on one fixed checkpoint. Confirm only the
best implementation against the same TCN on matched hardware, reporting warm
and cold latency, throughput, peak memory, and measured board power or energy
when available. Dense-MAC proxies alone cannot pass this milestone.

## Milestone C — generalization and continual learning

Question: does the frozen Milestone A/B system transfer beyond event audio and
does gated structural plasticity improve adaptation without destroying prior
skills?

Use one event-vision benchmark and one preregistered distribution shift. Run
static, full-plasticity, and gated-plasticity conditions under identical seeds.
Report task accuracy, adaptation time, retention, active edges, and realized
runtime. This milestone replaces a series of isolated dataset and plasticity
phases with one factorial experiment.

## Reporting rule

Each milestone produces one JSON record, screen and confirmation CSVs, a
summary plot, and an explicit `pass` or `stop` decision. Results are copied to
`gen5/outputs/`, analyzed once, recorded in `research.md`, and then pushed.
Long runs checkpoint after every arm/seed pair and resume only when the stored
run signature exactly matches the requested configuration.
