# Phase 40 analysis: SHD analog dynamics and topology

Archive SHA-256: `EE66EB871400680B16803533061F296BDFF35C82F2B62ED703003DBB6D6AD27F`

## Registered gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Dense recurrent analog vs dense recurrent LIF | +2 points; two +1 seeds | -2.709; zero +1 seeds | Fail |
| Sparse leaky vs dense feedforward analog | +2 points; two +1 seeds | +5.639; three +1 seeds | Pass |
| Sparse leaky vs sparse instant | +1 point; two positive seeds | +2.061; three positive seeds | Pass |
| Sparse leaky vs raw temporal | +2 points | +3.180 | Pass |

## Interpretation

Sparse leaky analog reaches `81.140%`, the highest mean in this experiment.
Sparse instant analog reaches `79.078%`, establishing a reproducible benefit
from temporal leak. Dense feedforward analog reaches only `75.501%`; the sparse
model's gain is therefore not explained by analog activation alone.

Dense recurrent analog falls to `71.555%`, below dense recurrent LIF at
`74.264%`. Recurrence again fails, and high analog activation (`94.91%`) is
consistent with saturation or poorly conditioned recurrent state.

The surviving model uses 700 fixed sparse edges and 132,931 trained readout
parameters. It beats the dense feedforward analog despite having dramatically
fewer dynamics parameters, but runs about two times slower because the current
sparse implementation is not kernel optimized.

## Goal sanity check

The evidence supports sparse expansion plus leaky state as a useful SHD feature
transform. It does not support spiking, plasticity, recurrence, biological
memory claims, neuromorphic energy efficiency, or state-of-the-art status.
Phase 41 tests whether sparse width is a real fixed-budget scaling effect and
locates its useful capacity knee.
