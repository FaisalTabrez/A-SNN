# Gen-27 trained threshold-robustness preregistration

Date frozen: 2026-08-20

Gen-27 trains three validation-selected Phase-49 residual-LIF networks on SSC,
then substitutes the Gen-25 FP32 sparse input operator without changing weights
or state equations. Dense inference, sparse inference, and a batch-shuffled
current-error control share all examples.

The primary question is behavioral rather than bitwise: across every seed,
sparse test accuracy must remain within 0.1 point of dense accuracy, prediction
agreement must be at least 99.9%, and spike disagreement must be at most 0.01%.
Maximum current/logit errors and the fraction of state updates within `1e-3` of
threshold are diagnostic. Passing authorizes a custom Triton kernel with these
semantics; failure moves to explicit threshold-margin training. It does not
authorize energy claims.
