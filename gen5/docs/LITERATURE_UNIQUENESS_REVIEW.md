# AMMC-SNN / Gen-5 literature uniqueness review

Date: 2026-06-26

Status: first-pass literature reconnaissance, not a formal systematic review or
patent search.

## Executive assessment

AMMC is not unique because it uses spiking neurons, STDP, pruning, dopamine,
astrocyte-like modulation, sleep replay, or neuroevolution. All of those have
clear prior art.

AMMC may be unique, or at least unusually positioned, as an integrated
closed-loop framework that combines:

1. embodied sparse SNN agents,
2. structural plasticity through sprouting and pruning,
3. STW/LTW memory separation,
4. astrocyte-like low-frequency chemical modulation,
5. day/night offline replay,
6. multi-agent evolutionary selection,
7. champion serialization and browser/PyTorch weight bridge,
8. accelerator-oriented vectorized execution on TPU/XLA.

The defensible novelty claim should therefore be architectural integration and
benchmarkable continual-learning behavior, not invention of any single
biological mechanism.

## Nearby prior art by theme

### 1. Structural plasticity and pruning in SNNs

Closest prior work:

- Rathi, Panda, Roy, **"STDP Based Pruning of Connections and Weight
  Quantization in Spiking Neural Networks for Energy Efficient Recognition"**
  (2017). This work prunes low-correlation SNN synapses and keeps critical
  ones for energy/area efficiency.
  Source: https://arxiv.org/abs/1710.04734
- Pan, Zhao, Zeng, Han, **"Adaptive structure evolution and biologically
  plausible synaptic plasticity for recurrent spiking neural networks"**
  (2023). This combines adaptive/evolutionary LSM structure search with
  dopamine-modulated BCM plasticity for decision tasks and rule reversal.
  Source: https://arxiv.org/abs/2304.01015

AMMC distinction:

- AMMC uses structural plasticity as an ongoing embodied runtime process, not
  only as an offline topology search or compression method.
- AMMC ties pruning/sprouting to organism-level reward/stress and sleep
  consolidation.

Novelty risk:

- Structural evolution plus dopamine-modulated plasticity is already close
  conceptually, especially Pan et al. 2023. AMMC must differentiate through
  multi-agent embodiment, astrocyte overlay, STW/LTW consolidation, and
  reproducible throughput/retention evidence.

### 2. Dynamic sparse training and rewiring

Closest prior work:

- Bellec et al., **"Deep Rewiring: Training very sparse deep networks"**
  (2017/ICLR 2018). DEEP R rewires sparse networks during supervised training
  while keeping connection count strictly bounded.
  Source: https://arxiv.org/abs/1711.05136
- Evci et al., **"Rigging the Lottery: Making All Tickets Winners"**
  (RigL, 2019/ICML 2020). RigL trains sparse networks with fixed parameter
  count and fixed compute, updating topology via magnitude and intermittent
  gradient information.
  Source: https://arxiv.org/abs/1911.11134
- Sokar et al., **"Dynamic Sparse Training for Deep Reinforcement Learning"**
  (2021). Dynamic sparse topology adapts during DRL and reduces parameters and
  FLOPs.
  Source: https://arxiv.org/abs/2106.04217

AMMC distinction:

- AMMC's topology changes are biologically framed as synaptogenesis/pruning in
  spiking embodied agents rather than gradient/magnitude-driven sparse ANN
  training.
- AMMC explicitly separates volatile STW and durable LTW and serializes the
  connectome for visual inspection and external training import.

Novelty risk:

- Fixed-capacity masked sparse pools are not novel by themselves. AMMC should
  not claim novelty merely for dynamic sparse tensors.

### 3. Astrocyte / tripartite-synapse computation

Closest prior work:

- Tewari and Majumdar, **"A Mathematical Model of Tripartite Synapse:
  Astrocyte Induced Synaptic Plasticity"** (2011). Biophysically detailed
  astrocyte modulation of short-term synaptic plasticity.
  Source: https://arxiv.org/abs/1105.0866
- Tang et al., **"Introducing Astrocytes on a Neuromorphic Processor:
  Synchronization, Local Plasticity and Edge of Chaos"** (2019). Astrocyte
  module on Loihi that senses synaptic activity, can switch STDP on/off, and
  monitors ordered/chaotic regimes.
  Source: https://arxiv.org/abs/1907.01620
- Shen et al., **"Astrocyte-Enabled Advancements in Spiking Neural Networks for
  Large Language Modeling"** (2023). Astrocyte-Modulated Spiking Unit for
  memory retention and long-term dependency tasks.
  Source: https://arxiv.org/abs/2312.07625
- Yang et al., **"Characterizing Learning in Spiking Neural Networks with
  Astrocyte-Like Units"** (2025). Adds astrocyte-like units to a liquid-state
  machine and studies learning rate effects.
  Source: https://arxiv.org/abs/2503.06798

AMMC distinction:

- AMMC uses astrocytes as a low-frequency spatial chemical field over an
  embodied, structurally plastic, evolving population.
- Astrocyte state modulates excitability, pruning/sprouting eligibility, and
  reward/stress feedback.

Novelty risk:

- Astrocyte-modulated SNNs are active prior art. AMMC should claim "integrated
  astrocyte overlay for embodied structural-plasticity control," not "first
  astrocyte SNN."

### 4. Sleep, replay, and consolidation

Closest prior work:

- Whelan, Prescott, Vasilaki, **"A Robotic Model of Hippocampal Reverse Replay
  for Reinforcement Learning"** (2021). Reverse replay accelerates learning and
  improves stability in robotic navigation.
  Source: https://arxiv.org/abs/2102.11914
- Pietras, Schmutz, Schwalger, **"Mesoscopic description of hippocampal replay
  and metastability in spiking neural networks with short-term plasticity"**
  (2022). Bottom-up spiking model of hippocampal replay and metastable replay
  dynamics.
  Source: https://arxiv.org/abs/2204.01675
- Massey et al., **"Sleep-Based Homeostatic Regularization for Stabilizing
  Spike-Timing-Dependent Plasticity in Recurrent Spiking Neural Networks"**
  (2026). Sleep/wake phases suppress input and stabilize STDP with offline
  stochastic activity and homeostatic regularization.
  Source: https://arxiv.org/abs/2601.08447

AMMC distinction:

- AMMC's sleep phase transfers STW into LTW and changes structural survival
  rules, rather than only renormalizing weights.
- The replay mechanism is embedded inside a multi-agent evolutionary organism
  loop and tied to environmental day/night state.

Novelty risk:

- Sleep-like stabilization and replay are existing ideas. AMMC must show that
  STW/LTW consolidation plus structural plasticity improves retention under
  perturbation.

### 5. Embodied dopamine learning and neurorobotics

Closest prior work:

- Evans, **"Reinforcement Learning in a Neurally Controlled Robot Using
  Dopamine Modulated STDP"** (2015). Embodied robot food-foraging with
  dopamine-modulated STDP; the robot learns food attraction and can unlearn
  behavior when the environment changes.
  Source: https://arxiv.org/abs/1502.06096
- Tang, Kumar, Michmizos, **"Reinforcement co-Learning of Deep and Spiking
  Neural Networks for Energy-Efficient Mapless Navigation with Neuromorphic
  Hardware"** (2020). SNN actor with deep critic for robotic navigation and
  Loihi deployment.
  Source: https://arxiv.org/abs/2003.01157

AMMC distinction:

- AMMC combines embodied reward learning with structural connectome churn,
  astrocyte modulation, sleep consolidation, and multi-agent evolution.

Novelty risk:

- Food/toxin foraging with dopamine-STDP is not unique. AMMC must avoid
  presenting the foraging task itself as novel.

### 6. Neuroevolution and co-evolution

Closest prior work:

- Stanley and Miikkulainen, **NEAT** (2002). Evolves neural topology and
  weights, starts simple, complexifies over time, and preserves innovation.
  Primary paper link from UT Austin:
  http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
- Miikkulainen and Stanley, **"Competitive Coevolution through Evolutionary
  Complexification"** (2011). Applies NEAT-style complexification in a
  coevolutionary robot duel domain.
  Source: https://arxiv.org/abs/1107.0037
- Wang et al., **"TensorNEAT: A GPU-accelerated Library for NeuroEvolution of
  Augmenting Topologies"** (2025). Tensorizes NEAT-like populations for
  hardware acceleration with JAX/Brax/gymnax.
  Source: https://arxiv.org/abs/2504.08339

AMMC distinction:

- AMMC uses tensorized populations but with per-agent sparse SNN connectomes,
  structural plasticity, sleep, astrocytes, and embodied chemical feedback.

Novelty risk:

- Tensorized neuroevolution exists. AMMC should not claim "first tensorized
  neuroevolution." The claim is "tensorized neuromorphic organism runtime with
  biological plasticity layers."

## Uniqueness matrix

| AMMC component | Prior art strength | AMMC uniqueness assessment |
|---|---:|---|
| Spiking neurons / LIF dynamics | Very high | Not unique |
| STDP / dopamine-modulated learning | Very high | Not unique |
| Synaptic pruning | High | Not unique |
| Synaptogenesis / rewiring | High | Not unique |
| Dynamic sparse masked pools | High | Engineering choice, not unique |
| Astrocyte-like modulation | Medium-high | Not unique alone |
| Sleep/replay consolidation | Medium-high | Not unique alone |
| STW/LTW explicit exportable split | Medium | Somewhat distinctive in this stack |
| Embodied food/toxin agent | High | Not unique |
| Multi-agent evolutionary swarm | High | Not unique alone |
| Browser connectome visualization + Colab/PyTorch weight bridge | Low-medium | Distinctive engineering layer |
| Full integration of all above | Low observed overlap | Strongest uniqueness claim |
| TPU/XLA-first sparse embodied SNN swarm benchmark | Low observed overlap | Potentially distinctive if validated |

## Best defensible claim today

AMMC should be positioned as:

> An integrated sparse spiking-organism framework for embodied continual
> learning, combining structural plasticity, astrocyte-like chemical modulation,
> sleep consolidation, and tensorized evolutionary populations with explicit
> connectome serialization.

Avoid claiming:

- first structural-plasticity SNN,
- first astrocyte SNN,
- first sleep SNN,
- first neuroevolution framework,
- first sparse/dynamic topology neural framework,
- Transformer replacement before external benchmarks.

## What would make AMMC clearly publishable

1. Show statistically significant adaptation and retention advantages over:
   - static SNN,
   - dynamic sparse non-spiking ANN,
   - dopamine-STDP SNN without astrocytes,
   - SNN with sleep but no structural plasticity,
   - PPO/MLP baseline,
   - NEAT/TensorNEAT-style topology evolution baseline.
2. Run ablations that isolate:
   - structural plasticity,
   - astrocyte modulation,
   - sleep consolidation,
   - STW/LTW split,
   - evolution,
   - embodiment.
3. Demonstrate the same mechanism on more than one environment.
4. Report active edge count, memory, throughput, retention, and fitness across
   multiple seeds.
5. Validate TPU/XLA throughput for both sparse-prior and saturated/champion-like
   topologies.

## Bottom line

The project is not unique at the level of biological ingredients. The project
is plausibly unique as a system-level integration and engineering/research
platform.

That is not a weakness. It is the right kind of novelty for a framework paper:
combine known biological mechanisms into a reproducible runtime, then prove
that the combination creates measurable continual-learning behavior that the
components alone do not.
