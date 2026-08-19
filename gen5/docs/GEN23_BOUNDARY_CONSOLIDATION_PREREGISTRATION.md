# Gen-23 boundary-gated consolidation preregistration

Date frozen: 2026-08-20

Gen-23 replaces the algebraically inert continuous-transfer rule with a
nontrivial boundary mechanism. After adapting to lesion A, the largest-magnitude
50% of active STW slots are transferred to LTW without changing immediate
effective output, then protected from lesion-B updates. Remaining STW is bounded
to ±0.5 and decays by 0.5% per update.

Five arms share the frozen SSC residual-LIF backbone and readout allocation:
static, single memory, bounded STW without consolidation, boundary-selective
dual memory, and boundary-randomized dual memory. Five seeds use the same
disjoint A/B lesions and data partition as Gen-22.

The selective arm must, in aggregate and at least three of five seeds:

1. retain A by at least +1.0 point versus single memory;
2. remain within 1.0 point of single-memory B accuracy;
3. lose at least 0.5 point of A retention when LTW is removed;
4. exceed shuffled selection by 0.5 point on mean A/B accuracy;
5. change A accuracy by no more than 0.1 point at the consolidation boundary.

These gates are frozen before execution. Passing supports only boundary-gated
consolidation in the residual readout. Failure closes this dual-memory branch
without a rescue sweep.
