# SNN literature review: project-specific inferences

Date: 2026-08-09

Status: targeted narrative review of primary papers, not a systematic review,
meta-analysis, or patent search. The purpose is to interpret AMMC Phase 24-27
results and choose discriminating experiments.

## Executive conclusion

The literature does not suggest that AMMC should simply add more neurons or
more random synapses. Strong temporal SNN results usually combine several of
the following:

1. neuron state with a slow adaptive timescale;
2. carefully controlled surrogate gradients or eligibility traces;
3. task-aligned input connectivity and synaptic delays;
4. repeated sparse prune-regrow cycles rather than one-shot growth;
5. firing-rate or excitability homeostasis; and
6. evaluation on genuinely temporal data with spike, operation, latency, and
   memory accounting.

This explains our current pattern. Recurrence helps when input is sequential,
but extra recurrent edges do not. Sensor LTWs and sensor sprouts matter more,
showing that information entry and temporal alignment are the immediate
bottlenecks. The present LIF reservoir has only membrane leak as slow state,
and its `delay_steps` metadata are not yet executed by the Phase 24-27 forward
path. Topology scaling alone therefore adds capacity without adding the richer
temporal mechanisms used by leading SNNs.

## Papers and direct implications

### 1. Adaptive neuron state can matter more than additional connectivity

Bellec et al., *Long short-term memory and learning-to-learn in networks of
spiking neurons* (NeurIPS 2018), showed that spike-frequency adaptation gives
RSNN neurons a slow state and that LSNNs trained with BPTT plus rewiring can
approach LSTM performance on temporal tasks.

Source: https://proceedings.neurips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html

Deckers et al., *Co-learning synaptic delays, weights and adaptation in spiking
neural networks* (2023), reported complementary gains from learnable neuronal
adaptation and synaptic delays on speech benchmarks.

Source: https://arxiv.org/abs/2311.16112

Project inference: before another neuron-count or recurrent-edge sweep, add an
adaptive LIF/LSNN arm with the same topology and parameter budget. Our current
64-neuron LIF graph may be state-limited rather than node-limited.

### 2. Online biological plausibility requires eligibility traces plus learning signals

Bellec et al., *A solution to the learning dilemma for recurrent networks of
spiking neurons* (Nature Communications 2020), factorizes learning into local
eligibility traces and neuron-specific learning signals. Reward-based e-prop
combines these with reward prediction error.

Source: https://www.nature.com/articles/s41467-020-17236-y

Project inference: AMMC's STW is a natural location for an eligibility trace,
while the astrocyte grid can be tested as a spatially smoothed learning-signal
field. This is a more defensible interpretation than saying astrocytes replace
a global loss. A future ablation should compare BPTT, exact/global learning
signals, and astrocyte-local approximations on the same task.

### 3. One-shot sprouting is a diagnostic, not the standard dynamic-sparse recipe

Chen et al., *Pruning of Deep Spiking Neural Networks through Gradient
Rewiring* (2021), jointly optimizes connectivity and weights through Grad R and
reports deep-SNN compression with substantial sparsity.

Source: https://arxiv.org/abs/2105.04916

Shen et al., *Improving the Sparse Structure Learning of Spiking Neural
Networks from the View of Compression Efficiency* (ICLR 2025), adapts the
rewiring ratio during repeated pruning and regrowth rather than choosing a
single fixed sparsity intervention.

Source: https://openreview.net/forum?id=gcouwCx7dG

Knight, Senk, and Nowotny, *A flexible framework for structural plasticity in
GPU-accelerated sparse spiking neural networks* (2025), combines sparse GPU
execution, e-prop, and DEEP R; its reported benefit depends on maintaining
actual sparse execution rather than masking a dense tensor.

Source: https://arxiv.org/abs/2510.19764

Project inference: Phase 27's zero-weight gradient ranking is close in spirit
to gradient regrowth, but it is only a one-shot, four-batch selection test. If
it fails, that rejects this selector and schedule, not structural plasticity in
general. If it passes, the next structural experiment should hold edge count
fixed and periodically prune-regrow, with selection stability and churn logged.

### 4. Structural plasticity is normally stabilized by homeostasis

Yuan et al., *Incorporating structural plasticity into self-organization
recurrent networks for sequence learning* (Frontiers in Neuroscience 2023),
combines reward-modulated STDP, structural plasticity, and homeostatic
plasticity on sequence tasks.

Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1224752/full

Spiess et al., *Structural Plasticity Denoises Responses and Improves Learning
Speed* (Frontiers in Computational Neuroscience 2016), studies structural
plasticity together with STDP and reports denoising and faster learning.

Source: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00093/full

Project inference: our event ratios stayed bounded in Phase 26, but linear arms
increased activity by roughly 25-38%. A target-rate homeostatic variable should
become an explicit control before continuous growth is enabled. Growth should
respond to persistent credit/activity mismatch, not simply unused capacity.

### 5. Delays are a learnable computational resource, not only metadata

Mészáros, Knight, and Nowotny, *Learning Delays Through Gradients and
Structure* (2024), evaluates learnable delays and delay-selecting dynamic
pruning on Raw Heidelberg Digits and finds useful spatiotemporal structure.

Source: https://arxiv.org/abs/2407.18917

Project inference: AMMC's polychronization claim is currently ahead of its
Gen-5 implementation. `delay_steps` are serialized but the Phase 24-27 sparse
forward path does not route through delay buckets. Implementing fixed and then
learnable delay buckets is higher-value than adding more undelayed recurrent
edges.

### 6. MNIST is a diagnostic; temporal benchmarks should decide the architecture

Stanojevic et al., *High-performance deep spiking neural networks with 0.3
spikes per neuron* (Nature Communications 2024), demonstrates that carefully
trained feedforward time-to-first-spike SNNs can match corresponding ANNs on
several image datasets while using very sparse spike communication.

Source: https://www.nature.com/articles/s41467-024-51110-5

The LSNN, adaptation, and delay papers evaluate temporal MNIST, speech, SHD,
SSC, or related sequential tasks. Static-image accuracy therefore does not
demonstrate the distinctive value of recurrence, delays, or continual
plasticity.

Project inference: our 56.4% recurrent MLP versus 94.2% raw-pixel MLP is a
useful engineering warning, not a competitive benchmark. Finish Phase 27, then
move the main architecture comparison to SHD/SSC or another event-native
sequence. Keep row-sequential MNIST as a fast causal regression test.

### 7. Memory consolidation must be tested as continual learning

Tadros et al., *Sleep-like unsupervised replay reduces catastrophic forgetting
in artificial neural networks* (Nature Communications 2022), evaluates sleep
replay on sequential tasks and includes an SNN implementation.

Source: https://www.nature.com/articles/s41467-022-34938-7

Zenke, Agnes, and Gerstner, *Diverse synaptic plasticity mechanisms
orchestrated to form and retrieve memories in spiking neural networks* (Nature
Communications 2015), combines Hebbian, heterosynaptic, transmitter-induced,
and inhibitory plasticity to stabilize assemblies and consolidation.

Source: https://www.nature.com/articles/ncomms7922

Project inference: STW/LTW is biologically motivated but remains an engineering
hypothesis until evaluated on task sequences. The correct experiment is not
ordinary single-task accuracy: it is adaptation, backward transfer, forgetting,
and retention with replay/no-replay and STW/LTW/no-split ablations.

### 8. Sparse topology and low spike count do not alone prove energy efficiency

NeuroBench (Nature Communications 2025) distinguishes dense and effective
synaptic operations, activation sparsity, timing, and hardware-dependent
energy/latency measurements.

Source: https://www.nature.com/articles/s41467-025-56739-4

Project inference: active-edge count and agent-steps/s are necessary but not
sufficient. Add spikes/sample, effective accumulate operations, dense-equivalent
operations, latency, peak memory, and—when hardware permits—energy per sample.
Any energy claim must name the execution backend because masked sparse tensors
do not automatically save work on all hardware.

## Prioritized roadmap inferred from the literature

1. **Run Phase 27 unchanged.** It tests whether candidate selection explains
   the Phase 26 seed sensitivity.
2. **Adaptive-neuron ablation.** Compare LIF with adLIF/LSNN using identical
   topology, edge count, readout, training budget, and seeds.
3. **Executable delay buckets.** Make `delay_steps` affect propagation; compare
   no-delay, fixed-distance delay, and learnable-delay arms.
4. **Move to an event-native temporal benchmark.** Prefer SHD first because it
   is compact and widely used for SNN temporal processing; add SSC afterward.
5. **Only then test periodic rewiring.** Keep active-edge budget fixed, protect
   a core only if justified, and compare random, magnitude, gradient, and
   activity/credit-mismatch regrowth.
6. **Add homeostatic control.** Track per-neuron target firing rate, silent and
   hyperactive fractions, topology churn, and edge lifetime distributions.
7. **Run continual-learning ablations.** Separate replay, STW/LTW, structural
   plasticity, and astrocyte-like modulation on a task sequence.
8. **Adopt NeuroBench-style accounting.** Accuracy alone cannot support the
   project's efficiency claim.

## What not to infer

- Phase 26 does not show that structural plasticity broadly works; it shows a
  small linear-readout benefit from one sensor-growth dose.
- A Phase 27 failure would not disprove gradient rewiring generally.
- Biological inspiration does not establish biological plausibility.
- Sparse weights or spikes do not establish energy savings without operation
  and backend measurements.
- The existing evidence does not support a Transformer-replacement claim.

## Most useful near-term scientific question

> With topology, parameter budget, and optimizer held constant, does adding a
> slow adaptive neuron state or executable synaptic delays improve temporal
> accuracy and accuracy per effective synaptic operation more than adding
> sensor or recurrent edges?

This question directly distinguishes temporal mechanism from raw capacity and
is more informative than another neuron-count sweep.
