# Program sanity check after Gen-21

Date: 2026-08-20

The goal remains to determine whether brain-inspired adaptive mechanisms make
independent causal contributions after architectural, parameter, compute,
seed, and selection confounds are controlled.

Gen-21 advances that goal. It did not reward a larger architecture: all
readout arms shared the same frozen SSC residual-LIF backbone, allocation,
active-slot budget, data, and updates. Three of four proposed mechanisms did
not advance. Dual memory alone passed its registered adaptation, retention,
and LTW-removal gates on three confirmation seeds.

The sanity boundary is equally important. Identical accuracy between dual
memory and ordinary gradient adaptation means that Gen-21 supports useful LTW
storage, not yet a superior two-timescale memory system. It says nothing yet
about whole-network consolidation, replay, dynamic topology, delay learning,
local reward credit, hardware energy, or general intelligence.

The next phase is therefore Gen-22, a direct sequential-shift replication of
dual memory. A second disjoint lesion creates interference. The decisive claim
is improved retention of shift A after adapting to shift B without materially
reducing B accuracy, plus loss under LTW removal and correctly paired versus
shuffled consolidation. Only a pass should authorize moving dual memory from
the readout into backbone synapses.
