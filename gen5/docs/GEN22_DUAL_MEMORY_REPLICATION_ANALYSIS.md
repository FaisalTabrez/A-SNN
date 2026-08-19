# Gen-22 dual-memory replication analysis

Date: 2026-08-20

Status: `stop`; complete JSON/CSV/plot/manifest artifacts verified locally.

Across five seeds, single memory and dual memory were identical: both retained
34.9225% accuracy on lesion A after learning B, reached 45.1025% on B, forgot
11.1975 A points, and lost 3.7900 clean points. Dual memory gained exactly zero
over single memory on A, B, and their joint score for every seed.

The mean LTW-removal margin was only 0.34 point, below the 0.5-point gate, and
one seed was negative. Selective identity was not tested in this version;
class-shuffled continuous consolidation slightly exceeded the nominal dual arm
on A retention while sacrificing B. No seed qualified.

## Algebraic explanation

The continuous rule transferred `c × STW` to LTW and then multiplied STW by
`1-c` at every update. Consequently:

`effective_after = LTW + c·STW + (1-c)·STW = LTW + STW`.

With `c=0.02`, the decomposition changed but the effective weight did not.
This exactly explains the paired equality with single memory. Gen-21's positive
LTW-removal effect showed that useful weight mass had been relabeled as LTW;
it did not establish a functional two-timescale mechanism.

Decision: close the continuous conservation rule. One final corrected readout
test may use event-boundary consolidation, utility-selected protected pathways,
bounded STW, a bounded-STW no-consolidation control, and shuffled selection.
No coefficient sweep is authorized.
