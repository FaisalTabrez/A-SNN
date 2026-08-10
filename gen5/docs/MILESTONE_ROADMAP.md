# Gen-5 milestone roadmap

> Status (2026-08-10): Milestone A returned `stop` with no qualified causal
> arms. The current architecture branch is closed. Milestones B and C are
> deferred; they must not be run against this rejected architecture.

> Gen-6 update (2026-08-10): the separately preregistered shared-residual
> successor preserved TCN accuracy but failed its state-removal and
> shuffled-state causal gates. It also returned `stop` with zero qualified
> arms. Hardware and continual-learning milestones remain closed. The active
> workstream is evidence/publication closeout, not an automatic Gen-7 rescue.

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
