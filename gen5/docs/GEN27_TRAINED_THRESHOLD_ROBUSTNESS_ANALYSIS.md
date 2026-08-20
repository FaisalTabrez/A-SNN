# Gen-27 trained threshold-robustness analysis

Date analyzed: 2026-08-20

The supplied bundle was restored for one omitted summary file and then passed
its complete SHA-256 manifest. Gen-27 passed all behavioral gates. Sparse and
dense mean accuracy were identical at 48.0792%. Minimum prediction agreement
was 99.9625%, and mean spike disagreement was 0.00401%, below the registered
0.01% ceiling. Mean accuracy change was exactly zero.

Current and logit differences can be numerically large (`0.00304` and `0.1518`
maximum) while remaining behaviorally negligible because only 0.01333% of state
updates lie within `1e-3` of threshold. A shuffled-error control caused roughly
twice the spike disagreement without meaningful accuracy change. These results
authorize the registered custom Triton event-kernel audit, but not an energy
claim.
