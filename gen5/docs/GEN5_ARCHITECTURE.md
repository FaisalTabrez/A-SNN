# AMMC Gen-5 Architecture

## 1. Paradigm shift

Gen-4 was a browser-native biological proof. Gen-5 is the production runtime:
a sparse, dynamic, embodied, continuously learning neuromorphic framework meant
to compete with static ANN blocks in agentic workloads.

The Gen-5 API is Python-first, with TPU/XLA as the near-term Colab accelerator
target and C++/CUDA extension points reserved for later kernels that PyTorch
cannot express efficiently:

- dynamic sparse allocation
- edge deletion with memory reclamation
- local sprouting under spatial constraints
- delayed event routing
- slow chemical-field modulation

## 2. Lessons carried forward from Gen-4

### The PyTorch topology wall

Standard `nn.Parameter` tensors expect stable shapes. True synaptogenesis and
pruning change the number of edges. A production AMMC runtime therefore needs a
dynamic sparse backend, not a dense `Linear` layer and not a full `N x N`
matrix.

The Gen-5 scaffold handles this by exposing an allocator API:

- active edge slots are used in the forward pass
- pruned edge slots are masked and reusable
- sprouted edge slots are inserted into available slots
- the future CUDA backend can replace masked slots with real VRAM reclamation
- the current XLA backend benefits from stable fixed-capacity shapes

### The STW/LTW bridge works

Gen-4 showed that separating short-term and long-term weights gives us a clean
working-memory/permanent-memory distinction:

- STW changes quickly under reward, replay, and local coincidence
- LTW changes slowly through consolidation and external training import
- effective weight is `STW + LTW`
- pruning decisions should be based on LTW, not transient STW

### Astrocytes are low-frequency spatial loss

The tripartite synapse can be treated as a low-resolution chemical field:

- high local spike load pushes astrocytes toward GABA/suppression
- reward pushes astrocytes toward dopamine/excitability
- the chemical grid modulates thresholds, leak, and plasticity rates

This becomes a low-frequency convolution over the fast electrical graph.

## 3. Core modules

### Core 1: Dynamic sparse tensor backend

`DynamicSparseLinear` is the Sprint 1 prototype.

Current prototype:

- sparse edge list stored as source indices, target indices, active mask, sign,
  STW, LTW, and optional delay slots
- custom autograd function accumulates sparse current into target neurons
- structural changes happen through explicit `sprout` and `prune` calls outside
  the autograd step

Near-term XLA target:

- fixed-capacity edge-list tensors
- masked active/inactive slots instead of runtime shape changes
- static-shape sprouting/pruning candidate tensors
- explicit XLA step boundaries through `ammc_gen5.runtime`

Later CUDA target:

- dynamic edge-list memory allocator
- edge compaction/reuse on GPU
- local spatial sprouting kernel
- delayed routing buffer kernel

Important constraint: changing tensor shape during a PyTorch backward pass is
not optimizer-safe and is also hostile to XLA compilation. The production
design should keep Python-facing handles stable while XLA uses static masked
pools now and a later CUDA allocator manages the underlying sparse pool.

### Core 2: Dual-frequency tensor processing

The fast graph runs electrical ticks. The slow grid runs chemical ticks.

Fast graph:

- sparse electrical propagation
- membrane updates
- delayed spike/event buffers

Slow grid:

- dense chemical state tensor
- low-frequency smoothing
- dopamine/GABA reward and suppression fields
- modulation sampled back into neuron/synapse kernels

### Core 3: Vectorized embodiment

The framework should present an environment interface compatible with:

- Isaac Gym / Isaac Sim vectorized robotics
- MuJoCo-style vectorized tasks
- custom market/audio/video streams

The AMMC runtime receives sensor tensors and emits motor/action tensors, while
astrocytes receive reward, collision, stress, and novelty fields.

### Core 4: Spatiotemporal delays as attention

AMMC substitutes sparse physical delays for dense Transformer attention:

- edge delays route events through time
- sequential alignment emerges from polychronous timing
- memory is in sparse topology and delay buffers, not a quadratic KV cache

Target complexity: near `O(E)` per tick, where `E` is the active sparse edge
count, rather than `O(N^2)`.

## 4. Development milestones

### Sprint 1: Dynamic sparse autograd prototype

- Implement sparse edge-list forward/backward.
- Provide edge insertion/deletion API.
- Keep PyTorch-facing parameter shapes stable for optimizer safety.
- Define the TPU/XLA compatibility boundary and the later C++/CUDA replacement
  boundary.

### Sprint 2: DualTensor manager

- Add slow astrocyte grid state.
- Add sensing from activity tensors.
- Add modulation tensors for electrical thresholds and plasticity multipliers.

### Sprint 3: LTW/STW optimizer wrapper

- Make STW and LTW first-class optimizer groups.
- Allow different learning rates, decay rates, and consolidation schedules.
- Add import/export of LTW for accelerator/JS bridge compatibility.

### Sprint 4: Tensor environment

- Keep all agent, food, toxin, velocity, and fitness state in tensors.
- Use broadcast math for nearest-object sensing and collision detection.
- Avoid Python loops over agents so the environment can scale to 10,000+
  agents on XLA/CUDA.

Prototype artifact: `ammc_gen5.tensor_environment.TensorEnvironment2D`.

### Sprint 5: Vectorized transducer

- Convert batched environment tensors into sensor-neuron input channels.
- Decode motor-neuron spikes into batched physical actions.
- Couple environment and sparse brain into a headless loop.

Prototype artifacts:

- `ammc_gen5.transducer.VectorizedTransducer`
- `ammc_gen5.transducer.HeadlessAMMCLoop`

### Sprint 6: Batched culling and broadcasting

- Rank agent fitness with `torch.argsort`.
- Keep the top population slice as survivors.
- Overwrite bottom performers by indexed tensor assignment from survivor rows.
- Preserve all genome fields together: source, target, active mask, sign, delay,
  STW, and LTW.

Prototype artifact: `ammc_gen5.evolver.TensorEvolver`.

### Sprint 7: Tensor topology mutation

- Apply Gaussian LTW noise to active child edge slots.
- Randomly prune active child edge slots.
- Randomly sprout inactive child edge slots by filling source/target/sign/delay
  tensors and flipping the active mask.
- Keep mutation entirely tensorized over child rows and edge slots.

### Sprint 8: Evolving headless loop

- Bind tensor physics, transduction, per-agent sparse brains, motor decoding,
  and epoch evolution into one runtime.
- Keep all per-agent work tensorized.
- Trigger epochs through an `epoch_steps` counter.
- At epoch boundary:
  - read `TensorEnvironment2D.fitness`
  - call `TensorEvolver.evolve(fitness)`
  - reset environment positions/scores
  - clear neural membrane state
  - increment generation

Prototype artifact: `ammc_gen5.evolving_loop.EvolvingHeadlessAMMCLoop`.

### Headless telemetry

Because Gen-5 has no visual UI, epoch-level telemetry is part of the runtime
contract. `EvolutionTelemetryLogger` records:

- max fitness
- mean population fitness
- mean active synapses
- sprout/prune/mutation counts

It writes JSON/CSV for analysis and can produce a matplotlib PNG plot in Colab.

### Champion export

The all-time best genome must be snapshotted during evolution because telemetry
does not contain topology tensors. `EvolvingHeadlessAMMCLoop` stores
`best_genome_snapshot`, `best_fitness`, and `best_generation`.

`ChampionExporter` converts that snapshot into:

- a Gen-4 browser connectome (`AMMC-SNN/connectome`)
- a matching browser weight import file (`AMMC-SNN/colab-weights`)
- a Gen-5 sparse adjacency analysis file

Load order in the browser remains important:

1. Load `champion_connectome.json`
2. Import `colab_weights.json`

## 5. Validation gates

- shape-safe autograd smoke test
- edge insertion/pruning contract test
- STW/LTW consolidation test
- chemical modulation sanity test
- vectorized environment shape test
- environment-to-brain transducer shape test
- tensorized culling/broadcast test
- tensor topology mutation test
- evolving loop epoch-reset test
- telemetry JSON/CSV contract test
- champion export schema contract test
- deterministic seeded run for reproducibility
- later: CUDA kernel parity against the PyTorch prototype
