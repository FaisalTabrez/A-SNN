# Phase 39 analysis: sparse SHD mechanisms

Archive SHA-256: `B4A021423C00270E4B29014250D196DAC10AD4CD7539E116CA72399F7BEEF002`

## Registered gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Sparse LIF vs raw temporal | +2 mean points | +1.649 | Fail |
| LIF vs matched analog | +2 mean points; two +1 seeds | -1.016; zero positive seeds | Fail |
| Trainable vs frozen LIF LTW | +1 mean point; two positive seeds | -0.015; one positive seed | Fail |

## Interpretation

Raw temporal decoding reaches `77.959%`. Sparse LIF reaches `79.623%` with
frozen LTWs and `79.608%` with trainable LTWs. Sparse analog reaches `80.624%`
for both frozen and trainable variants to displayed precision.

The result falsifies the proposed mechanism under this setup. Hard threshold
spikes reduce accuracy by `1.016` points relative to matched analog dynamics.
LTW optimization produces measurable weight movement without an accuracy gain.
The only surviving result is the frozen sparse analog model's `+2.665` point
gain over raw temporal. It is also faster than sparse LIF (`8,847` versus
`7,325` test examples/s for the frozen arms).

This does not invalidate AMMC as an engineering exploration, but it changes the
scientific claim. The current SHD performance belongs to a sparse analog leaky
representation until a spiking or plasticity contribution is demonstrated.

## Next decision

Phase 40 directly compares dense and sparse analog topologies and removes leak
from a paired sparse arm. Structural plasticity remains frozen. If sparse width
and leak do not pass their gates, the project should stop attributing SHD gains
to the AMMC core and treat the temporal decoder as the principal result.
