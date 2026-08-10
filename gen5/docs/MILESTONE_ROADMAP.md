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

> Gen-11 result (2026-08-10): conventional readout and full fine-tuning adapted,
> but analog/LIF adapters missed the two-point gain gate. LIF state removal
> erased its small gain while identity shuffling cost only 0.011 point. The
> terminal decision is `stop`; synaptic STW/LTW remains closed.

> Gen-12 update (2026-08-10): the user authorized a distinct fast-memory
> hypothesis rather than an adapter sweep. A frozen robust TCN is paired with
> dense and rank-order spiking associative prototypes. Memory removal and
> class-association shuffling are mandatory causal controls. A pass opens only
> context-discovery and consolidation testing. The protocol is frozen in
> `GEN12_ASSOCIATIVE_MEMORY_PREREGISTRATION.md`.

> Gen-12 result (2026-08-10): dense and spiking class prototypes gained only
> 0.250 and 0.278 point, far below the conventional adaptation controls.
> Activity, storage, and context-gated retention were healthy, but removal and
> association-shuffle costs missed their causal gates. The terminal decision
> is `stop`; prototype memory and automatic consolidation are closed.

> Gen-13 update (2026-08-10): the next registered mechanism localizes the
> successful output credit assignment as a manual three-factor rule. Analog
> and sparse-spiking traces are tested against autograd readout and full
> fine-tuning with fast-weight removal and class-shuffle controls. A pass opens
> only a separate STW/LTW consolidation preregistration. See
> `GEN13_LOCAL_PLASTICITY_PREREGISTRATION.md`.

> Gen-13 result (2026-08-10): conventional readout and full fine-tuning
> adapted, but analog and spiking local rules gained only 0.420 and 0.410
> point. Healthy activity, broad fast-weight occupancy, and zero forgetting
> rule out an execution collapse. Weight-removal and class-shuffle effects
> missed their replicated gates. The terminal decision is `stop`; Gen-9–13
> continual adaptation is closed. The next program must use a genuinely new
> hypothesis, with reward-modulated embodiment recommended.

> Gen-14 update (2026-08-10): the recommended new program is implemented as a
> terminal reward-eligibility screen. Delayed scalar food/toxin reward
> modulates local sensor/action traces; no label, target action, or autograd
> gradient reaches fast weights. Static, oracle, analog, spiking, and
> shuffled-reward arms share a matched tensorized swarm. A pass opens only a
> causal confirmation under a separate preregistration. See
> `GEN14_REWARD_ELIGIBILITY_PREREGISTRATION.md`.

> Gen-14 result (2026-08-10): the oracle reached +8.381 net fitness per 1,000
> steps, validating the world, while spiking eligibility finished at -0.109,
> below static (+0.641) and shuffled reward (+0.052). Activity and weight range
> were healthy. The terminal decision is `stop`; reward-specific eligibility
> is rejected. Empirical mechanism expansion is paused for a 16-source
> evidence freeze and matched reward-baseline/evaluation-protocol redesign.

> Gen-15 update (2026-08-10): the authorized diagnostic uses independent
> matched worlds, identical seeded pre/post resets, and shared-policy
> REINFORCE with correct versus agent-shuffled reward. It asks whether the
> delayed scalar reward itself supports identity-specific conventional
> learning. No new local-plasticity mechanism is included. See
> `GEN15_REWARD_BASELINE_PREREGISTRATION.md`.

> Gen-15 result (2026-08-10): all registered diagnostic gates passed. Static
> reset was exact, the oracle reached +10.661, and correct REINFORCE improved
> by +0.992 fitness per 1,000 steps while finishing +1.267 above shuffled
> reward. The learner remained slightly negative and its gain was dominated by
> one seed. The result validates the delayed reward and identity protocol, not
> Gen-14 or a local mechanism.

> Gen-16 update (2026-08-10): the next frozen experiment isolates credit
> assignment using one matched 8-to-4 policy. Autograd REINFORCE is compared
> with the exact manual score-function rule `return × sensor × (chosen -
> probability)`, plus static, oracle, and shuffled-reward controls. Gradient
> parity and replicated behavioral identity are required before sparse-spiking
> translation can open. See `GEN16_LOCAL_SCORE_CREDIT_PREREGISTRATION.md`.

> Gen-16 result (2026-08-10): every registered gate passed. The manual
> score-function gradient matched autograd within 2.79e-9, both policies
> finished at +0.300 with zero behavioral gap, and correct reward beat
> shuffled reward on all three seeds. Mean learning gain was only +0.183, so
> the result validates exact analog linear credit assignment rather than a
> capable SNN or continuous-learning system.

> Gen-17 update (2026-08-10): the validated rule is translated to eight
> parameter-matched Bernoulli sensory events. Analog, static, oracle, activity,
> and shuffled-reward controls are frozen in one run. A pass opens only larger
> sparse-spiking replication before memory. See
> `GEN17_SPARSE_SPIKING_CREDIT_PREREGISTRATION.md`.

> Gen-17 result (2026-08-10): `stop`. Sparse activity and analytic-gradient
> parity passed, but correct-reward spiking learning lost 0.391 fitness per
> 1,000 steps and finished 1.052 below shuffled reward. The analog reference
> gained only 0.004 on its fresh seeds, so analog robustness is again the
> blocking question.

> Gen-18 update (2026-08-10): ten untouched seeds replicate the unchanged
> analog local-credit rule with confidence-bound and 7/10-seed gates. No new
> spike encoding, memory, or topology mechanism is authorized before this
> replication. See `GEN18_LOCAL_CREDIT_REPLICATION_PREREGISTRATION.md`.

> Gen-18 result (2026-08-10): `stop`. Mean gain was +0.796 and mean
> correct-minus-shuffled margin was +0.510, but their lower 95% bounds were
> -0.016 and -0.013; only 5/10 and 6/10 seeds qualified. The tested local
> reward-credit program is closed without a rescue sweep.

> Gen-19 update (2026-08-10): a distinct external-generalization program now
> tests the supported residual-LIF state mechanism on real N-MNIST event
> vision. Accuracy cannot pass alone; state removal, sample-identity shuffling,
> and activity gates remain mandatory. See
> `GEN19_NMNIST_STATE_REPLICATION_PREREGISTRATION.md`.

> Gen-19 result (2026-08-10): `stop`. The residual model matched N-MNIST
> accuracy and state removal cost 15.210 points, but shuffled state improved
> accuracy by 2.300 points. The cross-modal identity claim is rejected. The
> next milestone is evidence/publication closeout; a new architecture requires
> a separate theory and preregistration rather than a rescue sweep.

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
