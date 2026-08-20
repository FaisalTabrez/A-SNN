# Gen-26 sparse numerical-fidelity analysis

Date analyzed: 2026-08-20

Gen-26 stopped. SSC bins were already binary: the measured nonbinary fraction
was zero and binary/count dense predictions agreed exactly. FP64 COO did not
reduce error. All three variants reached approximately `3.32e-4` maximum
current deviation and `3.29e-2` maximum logit deviation while retaining exact
predicted classes. State dynamics amplified current differences by up to 131x.

The Gen-25 error is therefore not an encoding problem or insufficient sparse
accumulator precision. It is an accumulation-order difference between dense
Conv1d and sparse matrix multiplication, magnified by hard threshold crossings.
No operator can be selected from the random-weight diagnostic. Gen-27 must test
whether the differences alter spikes, decisions, or accuracy after actual
validation-selected SSC training.
