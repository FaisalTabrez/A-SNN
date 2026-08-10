# Gen-11 frozen sensory backbone plus plastic state adapter preregistration

Status (2026-08-10): completed with terminal `stop`; zero qualified arms. See
`GEN11_PLASTIC_ADAPTER_ANALYSIS.md`.

## New hypothesis

Gen-10 showed that sensor dropout produces a source-competent robust TCN while
asking a residual spiking path to relearn the sensory classifier destroys
accuracy. Gen-11 tests functional separation instead: a sensor-dropout TCN is
the frozen sensory cortex, while a small downstream analog or LIF circuit
learns only bounded correction logits during sensor-damage adaptation.

This is not a Gen-10 mask/alignment sweep. The backbone, fixed 35% damage mask,
adaptation stream, and conventional controls are held constant. STW/LTW,
replay, neuromodulation, and structural plasticity remain closed.

## Frozen strategies

- `dropout_tcn_static`: no adaptation.
- `dropout_tcn_readout`: update only the conventional classifier.
- `dropout_tcn_full_finetune`: all-parameter BPTT upper control.
- `analog_state_adapter`: freeze the backbone; adapt analog dynamics, gate,
  and a classifier-sized correction matrix.
- `lif_state_adapter`: freeze the backbone; adapt LIF dynamics, gate, and the
  identically sized correction matrix.

The adapter begins with a zero correction gate, so all strategies share the
same source behavior at zero adaptation samples. The adapter receives the
backbone's time-resolved hidden trace. Its trainable parameter count is matched
closely to the readout control rather than the full backbone.

## Protocol

Each seed 154–156 trains one TCN source backbone for 15 epochs with independent
20% sensor dropout and clean-validation checkpoint selection. The shifted task
uses Gen-9/10's fixed 35% mask at seed 909. Official validation examples form
the sequential non-replayed adaptation stream at cumulative budgets
`0, 64, 256, 1024, 4096`; official test examples measure damaged accuracy and
undamaged retention only.

Each newly observed block receives three epochs at learning rate 0.001. Final
adapter evaluation includes full, direct-only, and batch-shuffled-state modes.

## Terminal LIF gate

`lif_state_adapter` passes only if:

- static sensor damage causes at least a two-point shift;
- final damaged accuracy improves by at least two points on at least two seeds;
- mean adaptation AUC and final damaged accuracy are each within one point of
  TCN readout;
- source forgetting is no more than 0.5 point worse than TCN readout;
- removing the adapter state costs at least 0.5 point on at least two seeds;
- shuffling adapter state identity costs at least 0.5 point on at least two
  seeds;
- final spike rate remains between 1% and 30%.

A pass opens a separately preregistered STW/LTW consolidation experiment. A
stop closes this adapter and forbids automatic adapter-width, gate, leak,
threshold, learning-rate, or mask sweeps.
