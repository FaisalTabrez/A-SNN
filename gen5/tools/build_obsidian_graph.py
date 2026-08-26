#!/usr/bin/env python3
"""Build the A-SNN Obsidian knowledge graph from curated repository evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SPRINTS = {
    "gen6": ("Causal State Successor", "[[Spiking Speech Commands]], [[Residual LIF State]]"),
    "gen7": ("Predictive State", "[[Spiking Speech Commands]], [[Predictive State Representation]]"),
    "gen8": ("Temporal Binding", "[[Spiking Speech Commands]], [[Temporal Binding]]"),
    "sprint18": ("Frozen Event-coded MNIST", "[[Event-coded MNIST]], [[Frozen Reservoir Representation]]"),
    "sprint19": ("Event Representation Decomposition", "[[Event-coded MNIST]], [[Temporal State Representation]]"),
    "sprint20": ("Temporal State Preservation", "[[Temporal State Representation]]"),
    "sprint21": ("Trainable Temporal MNIST", "[[Temporal State Representation]], [[LTW-STW Memory]]"),
    "sprint23": ("Causal Recurrence Ablation", "[[Temporal State Representation]], [[Sparse Recurrent Topology]]"),
    "sprint24": ("Sequential MNIST", "[[Sequential MNIST]], [[Sparse Recurrent Topology]]"),
    "sprint25": ("Trainable Sequential MNIST", "[[Sequential MNIST]], [[LTW-STW Memory]]"),
    "sprint26": ("Structural Sequential MNIST", "[[Sequential MNIST]], [[Structural Plasticity]]"),
    "sprint27": ("Utility-Gated Structural MNIST", "[[Sequential MNIST]], [[Structural Plasticity]]"),
    "sprint28": ("Adaptive Sequential MNIST", "[[Sequential MNIST]], [[Adaptive LIF Neurons]]"),
    "sprint29": ("Delayed Sequential MNIST", "[[Sequential MNIST]], [[Trainable Delays]]"),
    "sprint30": ("Trainable Delays MNIST", "[[Sequential MNIST]], [[Trainable Delays]]"),
    "sprint31": ("SHD Benchmark", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint36": ("SHD Temporal Pyramid", "[[Spiking-Heidelberg-Digits]], [[Temporal Pyramid Readout]]"),
    "sprint37": ("SHD Temporal Controls", "[[Spiking-Heidelberg-Digits]], [[Temporal Pyramid Readout]]"),
    "sprint38": ("SHD Matched Baselines", "[[Spiking-Heidelberg-Digits]], [[Matched Temporal Baselines]]"),
    "sprint39": ("SHD Sparse Mechanisms", "[[Spiking-Heidelberg-Digits]], [[Sparse Recurrent Topology]]"),
    "sprint40": ("SHD Analog Topology", "[[Spiking-Heidelberg-Digits]], [[Analog-Leaky-Topology]]"),
    "sprint41": ("SHD Sparse Width Scaling", "[[Spiking-Heidelberg-Digits]], [[Sparse-Width-Scaling]]"),
    "sprint42": ("SHD Initialization Robustness", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint43": ("SHD Validation Checkpoint", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint44": ("SHD Calibrated Baselines", "[[Spiking-Heidelberg-Digits]], [[Matched Temporal Baselines]]"),
    "sprint45": ("SHD Spiking Temporal Convolution", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint46": ("SHD State Placement", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint47": ("SHD Residual State Contribution", "[[Spiking-Heidelberg-Digits]], [[Residual LIF State]]"),
    "sprint48": ("SSC Residual LIF Replication", "[[Spiking Speech Commands]], [[Residual LIF State]]"),
    "sprint49": ("SSC Efficiency Baselines", "[[Spiking Speech Commands]], [[Compiled Residual State]]"),
    "gen9": ("Continual Adaptation", "[[Spiking Speech Commands]], [[Continual Adaptation]]"),
    "gen10": ("Robust Representation", "[[Spiking Speech Commands]], [[Sensor-dropout Representation]]"),
    "gen11": ("Plastic Adapter", "[[Spiking Speech Commands]], [[Plastic Adapter]]"),
    "gen12": ("Associative Memory", "[[Spiking Speech Commands]], [[Associative Memory]]"),
    "gen13": ("Local Plasticity", "[[Spiking Speech Commands]], [[Three-factor Local Plasticity]]"),
    "gen14": ("Reward Eligibility", "[[Embodied Delayed Reward]], [[Reward-modulated Eligibility]]"),
    "gen15": ("Reward Baseline", "[[Embodied Delayed Reward]], [[Reward Learning Baseline]]"),
    "gen16": ("Local Score Credit", "[[Embodied Delayed Reward]], [[Local Score Credit]]"),
    "gen17": ("Sparse Spiking Credit", "[[Embodied Delayed Reward]], [[Sparse Spiking Credit]]"),
    "gen18": ("Local Credit Replication", "[[Embodied Delayed Reward]], [[Local Score Credit]]"),
    "gen19": ("N-MNIST State Replication", "[[N-MNIST]], [[Residual LIF State]]"),
    "gen20": ("Spiking Spatial-temporal Translation", "[[N-MNIST]], [[Multiscale Spiking Representation]]"),
    "gen21": ("Matched Causal Mechanisms", "[[Spiking Speech Commands]], [[Matched Adaptive Mechanisms]]"),
    "gen22": ("Dual-Memory Replication", "[[Spiking Speech Commands]], [[Dual Memory Timescales]]"),
    "gen23": ("Boundary Consolidation", "[[Spiking Speech Commands]], [[Dual Memory Timescales]]"),
    "gen24": ("Compiled Residual State", "[[Spiking Speech Commands]], [[Compiled Residual State]]"),
    "gen25": ("Event-Driven Sparse Audit", "[[Spiking Speech Commands]], [[Event-driven Sparse Execution]]"),
    "gen26": ("Sparse Numerical Fidelity", "[[Spiking Speech Commands]], [[Behavioral Sparse Semantics]]"),
    "gen27": ("Trained Threshold Robustness", "[[Spiking Speech Commands]], [[Behavioral Sparse Semantics]]"),
    "gen28": ("Triton Event Kernel", "[[Spiking Speech Commands]], [[Event-driven Sparse Execution]]"),
    "gen29": ("Causal Evidence Closure", "[[Cross-benchmark Evidence Ledger]], [[Program-level Causal Synthesis]]"),
    "gen30": ("Dendritic Predictive Credit", "[[Delayed Contextual Binding]], [[Dendritic Predictive Credit]]"),
    "evidence1": ("Primary Audio Evidence Consolidation", "[[SHD-SSC Matched Audio Evidence]], [[Residual LIF State]]"),
}

CURRENT_STATE = {
    "sprint_id": "evidence1",
    "title": "Primary Audio Evidence Consolidation",
    "status": "active-protocol-defined-runner-pending",
    "summary": (
        "The project will first consolidate its supported residual-LIF temporal-audio evidence under "
        "one matched SHD/SSC protocol. Active-dendrite and astrocyte-context research is explicitly "
        "deferred until the causal, accuracy, systems, and clean-replication decisions are locked."
    ),
    "next_action": "Implement the canonical paired-seed SHD/SSC runner from Stage E1 without selecting architectures or checkpoints on final test data.",
    "benchmark": "SHD-SSC Matched Audio Evidence",
    "hypothesis": "Residual LIF State",
    "protocol": "gen5/docs/PRIMARY_EVIDENCE_TRACK_ROADMAP.md",
    "implementation": "pending canonical evidence runner",
    "guardrail": "Do not claim dense-baseline superiority, independent replication, product readiness, or energy efficiency before their separate registered gates pass.",
    "index_summary": "Primary temporal-audio evidence consolidation is active; the mechanism research track is deferred.",
}

HYPOTHESES = {
    "Causal State Successor": ("gen6", "stopped", "A causal state successor can provide a source-competent temporal representation."),
    "Predictive State Representation": ("gen7", "stopped", "Predicting future encoder state can create a causal sample-specific representation."),
    "Temporal Binding": ("gen8", "stopped", "Aligned temporal binding can establish useful identity- and order-specific state."),
    "Sparse-Width-Scaling": ("sprint41", "active", "Sparse SHD width can improve performance under bounded activity."),
    "Continual Adaptation": ("gen9", "stopped", "Residual state can support robust local continual adaptation after sensor damage."),
    "Local Score Credit": ("gen16", "stopped", "A local score-function rule gives stable reward-specific behavioral learning."),
    "Residual LIF State": ("gen19", "supported-scope-limited", "Residual LIF state has useful sample-specific temporal content across event domains."),
    "N-MNIST Accuracy Frontier": ("gen20", "completed", "Native spatial-temporal encoding supports competitive conventional N-MNIST accuracy."),
    "Multiscale Spiking Representation": ("gen20", "stopped", "Multi-timescale LIF state can close the native N-MNIST spiking representation gap."),
    "Robust Representation": ("gen10", "stopped", "Mask-trained residual representations can preserve conventional performance and causal state use."),
    "Plastic Adapter": ("gen11", "stopped", "A bounded plastic adapter can repair damaged representations with sample-specific state."),
    "Associative Memory": ("gen12", "stopped", "Fast associative prototypes can provide useful context-gated damaged-stream adaptation."),
    "Three-factor Local Plasticity": ("gen13", "stopped", "Manual local output-synapse plasticity can match conventional adaptation."),
    "Reward-modulated Eligibility": ("gen14", "stopped", "Eligibility traces can assign delayed reward credit in the embodied control task."),
    "Reward Learning Baseline": ("gen15", "supported", "The identical-reset delayed-reward protocol is learnable with a conventional REINFORCE baseline."),
    "Sparse Spiking Credit": ("gen17", "stopped", "One-event sparse coding preserves useful local reward-credit learning."),
    "Matched Adaptive Mechanisms": ("gen21", "stopped", "Dynamic topology, dual memory, learned delays, and local credit add causal value under matched controls."),
    "Dual Memory Timescales": ("gen22", "closed", "Separated short- and long-term weights improve sequential adaptation and retention over one memory."),
    "Compiled Residual State": ("gen24", "supported-scope-limited", "Compilation removes the eager-loop confound while preserving residual-LIF behavior."),
    "Behavioral Sparse Semantics": ("gen27", "supported-scope-limited", "Sparse event accumulation preserves trained predictions and spike behavior despite numerical differences."),
    "Event-driven Sparse Execution": ("gen28", "closed", "A custom event-native kernel can convert validated sparse semantics into accelerator throughput."),
    "Program-level Causal Synthesis": ("gen29", "supported", "A deterministic ledger can separate supported, rejected, untested, and proxy-only project claims."),
    "Dendritic Predictive Credit": ("gen30", "stopped-component-signal-only", "Residual apical teaching signals combined with basal eligibility traces can assign delayed hidden-layer credit without BPTT."),
}

DECISIONS = {
    "sprint42": ("stop", "Sparse SHD width gains did not survive independent initialization.", "raw 78.357%; sparse-512 78.058%; sparse-1024 77.380%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint43": ("stop", "Validation selection did not rescue the sparse-expansion branch.", "raw checkpoint 80.374%; sparse-512 78.023%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint44": ("completed", "A matched local temporal Conv1D became the honest SHD target.", "Conv1D 82.847%; raw 80.374%; dense LIF 75.103%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint45": ("stop", "Replacing direct local features with state caused the dominant accuracy loss.", "Conv1D 82.921%; analog state 76.472%; LIF 74.308%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint46": ("pass", "Residual feature preservation recovered stateful model accuracy.", "residual LIF 83.804%; Conv1D 82.862%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint47": ("pass", "Sample-specific residual LIF state contributed causally on SHD.", "full 83.908%; direct-only 77.488%; shuffled-state 79.741%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint48": ("pass", "The residual-state contribution replicated on official SSC.", "residual LIF 56.498%; direct-only 45.226%; shuffled-state 53.518%", "gen5/docs/MILESTONE_ROADMAP.md"),
    "sprint49": ("stop", "The causal mechanism survived but lost matched accuracy and software throughput to TCN.", "TCN 59.225%; residual LIF 55.973%; TCN 3.182x faster", "gen5/docs/MILESTONE_ROADMAP.md"),
    "gen20": ("stop", "The multiscale spiking N-MNIST translation missed the promotion gate.", "teacher 99.1165%; multiscale PLIF 96.3661%; gate 97.5%", "gen5/docs/GEN20_SPIKING_SPATIOTEMPORAL_TRANSLATION_PREREGISTRATION.md"),
    "gen21": ("partial", "Only dual-memory storage passed screening; it did not outperform ordinary gradient adaptation.", "static 38.0667%; gradient and dual memory 47.1708%; LTW removal -4.3333 points", "gen5/docs/GEN21_MATCHED_CAUSAL_MECHANISMS_ANALYSIS.md"),
    "gen22": ("stop", "Continuous STW-to-LTW transfer was algebraically function-preserving and gave no two-timescale advantage.", "single and dual joint score 40.0125%; gain 0", "gen5/docs/GEN22_DUAL_MEMORY_REPLICATION_ANALYSIS.md"),
    "gen23": ("stop", "Boundary consolidation improved old-shift retention but harmed new-shift learning and failed identity controls.", "A gain +1.9475 points; B loss 4.93 points; no qualified seed", "gen5/docs/GEN23_BOUNDARY_CONSOLIDATION_ANALYSIS.md"),
    "gen24": ("pass", "Compiled residual LIF removed the eager Python-loop confound.", "9.027x speedup; 243381 examples/s; 91.006% TCN parity", "gen5/docs/GEN24_COMPILED_RESIDUAL_STATE_ANALYSIS.md"),
    "gen25": ("stop", "Generic PyTorch COO was behaviorally stable but far slower than compiled dense execution.", "15201 vs 234864 examples/s; ratio 0.06473", "gen5/docs/GEN25_EVENT_DRIVEN_SPARSE_AUDIT_ANALYSIS.md"),
    "gen26": ("stop", "The sparse discrepancy came from accumulation-order error amplified at hard thresholds, not encoding or precision.", "max current error 3.32e-4; max logit error 3.29e-2; amplification 131x", "gen5/docs/GEN26_SPARSE_NUMERICAL_FIDELITY_ANALYSIS.md"),
    "gen27": ("pass", "Trained sparse substitution preserved task behavior under the preregistered gates.", "accuracy 48.0792%; min agreement 99.9625%; spike disagreement 0.00401%", "gen5/docs/GEN27_TRAINED_THRESHOLD_ROBUSTNESS_ANALYSIS.md"),
    "gen28": ("stop", "The Triton event-scatter kernel failed behavioral and throughput gates, closing the software event-sparse path.", "real native ratio 0.3276; best low-density ratio 0.5208; primary agreement 99.609%", "gen5/docs/GEN28_TRITON_EVENT_KERNEL_ANALYSIS.md"),
    "gen29": ("complete", "The deterministic claim ledger closes the current workstream without promoting a new mechanism.", "12 evidence documents; 12 explicit claims; no supported adaptive mechanism", "gen5/docs/GEN29_PROGRAM_CLOSURE_PROTOCOL.md"),
    "gen30": ("stop", "DPC isolated necessary eligibility and teaching-identity components but failed absolute learning and retention gates.", "B 52.8125%; A retained 40.6348%; drop 14.8730 points; qualified seeds 0/10", "gen5/docs/GEN30_DENDRITIC_PREDICTIVE_CREDIT_ANALYSIS.md"),
}

ARTIFACTS = (
    ("gen6", "gen6_successor_cuda_2026-08-10/gen6_successor.json", "Gen-6 Causal State Successor", "Spiking Speech Commands", "Residual LIF State", "Causal State Successor"),
    ("gen7", "gen7_predictive_state_cuda_2026-08-10/gen7_predictive_state.json", "Gen-7 Predictive State", "Spiking Speech Commands", "Predictive State Representation", "Predictive State Representation"),
    ("gen8", "gen8_temporal_binding_cuda_2026-08-10/gen8_temporal_binding.json", "Gen-8 Temporal Binding", "Spiking Speech Commands", "Temporal Binding", "Temporal Binding"),
    ("gen9", "gen9_continual_adaptation_cuda_2026-08-10/gen9_continual_adaptation.json", "Gen-9 Continual Adaptation", "Spiking Speech Commands", "Residual LIF State", "Continual Adaptation"),
    ("gen10", "gen10_robust_representation_cuda_2026-08-10/gen10_robust_representation.json", "Gen-10 Robust Representation", "Spiking Speech Commands", "Sensor-dropout Representation", "Robust Representation"),
    ("gen11", "gen11_plastic_adapter_cuda_2026-08-10/gen11_plastic_adapter.json", "Gen-11 Plastic Adapter", "Spiking Speech Commands", "Plastic Adapter", "Plastic Adapter"),
    ("gen12", "gen12_associative_memory_cuda_2026-08-10/gen12_associative_memory.json", "Gen-12 Associative Memory", "Spiking Speech Commands", "Associative Memory", "Associative Memory"),
    ("gen13", "gen13_local_plasticity_cuda_2026-08-10/gen13_local_plasticity.json", "Gen-13 Local Plasticity", "Spiking Speech Commands", "Three-factor Local Plasticity", "Three-factor Local Plasticity"),
    ("gen14", "gen14_reward_eligibility_cuda_2026-08-10/gen14_reward_eligibility.json", "Gen-14 Reward Eligibility", "Embodied Delayed Reward", "Reward-modulated Eligibility", "Reward-modulated Eligibility"),
    ("gen15", "gen15_reward_baseline_cuda_2026-08-10/gen15_reward_baseline.json", "Gen-15 Reward Baseline", "Embodied Delayed Reward", "Reward Learning Baseline", "Reward Learning Baseline"),
    ("gen16", "gen16_local_score_credit_cuda_2026-08-10/gen16_local_score_credit.json", "Gen-16 Local Score Credit", "Embodied Delayed Reward", "Local Score Credit", "Local Score Credit"),
    ("gen17", "gen17_sparse_spiking_credit_cuda_2026-08-10/gen17_sparse_spiking_credit.json", "Gen-17 Sparse Spiking Credit", "Embodied Delayed Reward", "Sparse Spiking Credit", "Sparse Spiking Credit"),
    ("gen18", "gen18_local_credit_replication_cuda_2026-08-10/gen18_local_credit_replication.json", "Gen-18 Local Credit Replication", "Embodied Delayed Reward", "Local Score Credit", "Local Score Credit"),
    ("sprint41", "shd_sparse_width_cuda_2026-08-10/shd_sparse_width.json", "SHD Sparse Width Evidence", "Spiking-Heidelberg-Digits", "Sparse Recurrent Topology", "Sparse-Width-Scaling"),
    ("gen19", "gen19_nmnist_state_replication_log_recovery_2026-08-10/gen19_nmnist_state_replication.json", "N-MNIST State Replication", "N-MNIST", "Residual LIF State", "Residual LIF State"),
    ("gen20", "nmnist_accuracy_benchmark_log_recovery_2026-08-11/nmnist_accuracy_benchmark.json", "N-MNIST Accuracy Benchmark", "N-MNIST", "Spatial-temporal CNN", "N-MNIST Accuracy Frontier"),
    ("gen29", "gen29_program_closure_2026-08-20/gen29_program_closure.json", "Gen-29 Causal Evidence Closure", "Cross-benchmark Evidence Ledger", "Deterministic Evidence Synthesis", "Program-level Causal Synthesis"),
)


def yaml_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_note(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def compact_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            qualified = f"{prefix}_{key}" if prefix else key
            if isinstance(item, (str, int, float, bool)) and len(result) < 18:
                result[qualified] = item
            elif isinstance(item, dict) and len(result) < 18:
                result.update(compact_metrics(item, qualified))
    return result


def artifact_metrics(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("aggregate", "summary", "decision"):
        if isinstance(data.get(key), dict) and (metrics := compact_metrics(data[key])):
            return metrics
    return compact_metrics(data)


def note_id(sprint_id: str) -> str:
    if sprint_id.startswith("evidence"):
        return sprint_id.replace("evidence", "Evidence-")
    return sprint_id.replace("sprint", "Sprint-").replace("gen", "Gen-")


def build_graph(vault: Path, outputs: Path) -> int:
    for sprint_id, (title, links) in SPRINTS.items():
        status = CURRENT_STATE["status"] if sprint_id == CURRENT_STATE["sprint_id"] else "historical"
        current_section = ""
        if sprint_id == CURRENT_STATE["sprint_id"]:
            current_section = f"""
## Current Research Position
**Status:** `{status}`

{CURRENT_STATE["summary"]}

**Next action:** {CURRENT_STATE["next_action"]}

**Frozen protocol:** {CURRENT_STATE["protocol"]}<br>
**Implementation:** {CURRENT_STATE["implementation"]}
"""
        write_note(vault / "Sprints" / f"{note_id(sprint_id)}.md", f"""---
type: sprint
sprint_id: {yaml_value(sprint_id)}
title: {yaml_value(title)}
status: {yaml_value(status)}
tags: [sprint, gen5]
---

# {sprint_id.upper()} - {title}

## Graph Connections
{links}

## Evidence
See linked experiment records and the repository's `gen5/docs/` preregistration or analysis for the source protocol.
{current_section}
""")
    write_note(vault / "Current State.md", f"""---
type: current-state
current_sprint: {yaml_value(CURRENT_STATE["sprint_id"])}
status: {yaml_value(CURRENT_STATE["status"])}
updated: "2026-08-26"
tags: [current-state, gen5, {CURRENT_STATE["sprint_id"]}]
---

# Current State

## Current Research Position
[[{note_id(CURRENT_STATE["sprint_id"])}]] is the current program position: **{CURRENT_STATE["title"]}**.

{CURRENT_STATE["summary"]}

## Frozen Scope
- Benchmark: [[{CURRENT_STATE["benchmark"]}]]
- Hypothesis: [[{CURRENT_STATE["hypothesis"]}]]
- Protocol: {CURRENT_STATE["protocol"]}
- Implementation: {CURRENT_STATE["implementation"]}

## Next Action
{CURRENT_STATE["next_action"]}

## Guardrail
{CURRENT_STATE["guardrail"]}

## Deferred Itinerary
The LTH-informed review refines the post-evidence research backlog without
changing the active sequence. See [[LTH-Informed Itinerary Review]].
""")
    write_note(vault / "Roadmaps" / "LTH-Informed Itinerary Review.md", """---
type: research-roadmap
status: deferred-until-evidence1-decision
updated: "2026-08-26"
tags: [roadmap, lth, dendrites, sparsity, gen5]
---

# LTH-Informed Itinerary Review

## Sequence Decision
[[Evidence-1]] remains active and must finish before this research backlog
opens. The reviewed proposal refines later experiments; it does not authorize
Gen-31 through Gen-34 as the next phases.

## Accepted Refinements
- First deferred mechanism study: context-specific dendritic supermask routing
  with a frozen backbone, explicit context, matched capacity, and causal
  context/mask controls.
- Conditional systems study: structured dendritic sparsity only with an
  operator that actually skips block or channel work and is compared with
  [[Compiled Residual State]] plus unstructured sparsity.
- Optional evolutionary diagnostics: original birth initialization versus
  reinitialization and continued mutation, followed only optionally by a
  compute-matched overparameterize-then-prune study.

## Boundaries
- [[Residual LIF State]] remains the active evidence hypothesis.
- [[Structural Plasticity]] is not reopened by this review.
- Gen-19 is negative mechanism-transfer evidence, not a direct test of a
  winning-ticket mask plus original initialization.
- No hardware-energy claim is authorized.

## Repository Source
`gen5/docs/LTH_INFORMED_ITINERARY_REVIEW.md`
""")
    for name, (origin, status, statement) in HYPOTHESES.items():
        write_note(vault / "Hypotheses" / f"{name}.md", f"""---
type: hypothesis
origin_sprint: {yaml_value(origin)}
status: {yaml_value(status)}
tags: [hypothesis, gen5]
---

# {name}

{statement}

## Graph Connections
- Origin: [[{note_id(origin)}]]
""")
    architecture_notes = {
        "Analog-Leaky-Topology": "Default analog leaky topology reference for imported Gen-5 experiment notes.",
        "Sparse Recurrent Topology": "Fixed-capacity sparse edge-list topology used for recurrent AMMC experiments.",
        "Predictive State Representation": "State representation trained to predict future encoder features.",
        "Temporal Binding": "Aligned state-and-direct temporal fusion representation.",
        "Residual LIF State": "Residual spiking state fused with a direct temporal representation.",
        "Temporal Pyramid Readout": "Multi-window temporal pooling decoder used in SHD controls.",
        "Spatial-temporal CNN": "Native-resolution conventional N-MNIST temporal upper control.",
        "Multiscale Spiking Representation": "Shared spatial stem with multi-timescale LIF temporal banks.",
        "LTW-STW Memory": "Short- and long-term weight decomposition investigated in Gen-5.",
        "Structural Plasticity": "Pruning and sprouting mechanism; currently gated by evidence.",
        "Trainable Delays": "Differentiable recurrent delay assignment in the sequential-MNIST series.",
        "Frozen Reservoir Representation": "Frozen sparse reservoir representation used as an event-coded baseline.",
        "Temporal State Representation": "Pre-reset temporal state retained for causal representation tests.",
        "Adaptive LIF Neurons": "Leaky integrate-and-fire neurons with an adaptive threshold.",
        "Matched Temporal Baselines": "Parameter-matched dense recurrent and GRU temporal comparators.",
        "Sensor-dropout Representation": "Sensor-mask-trained conventional feature representation.",
        "Plastic Adapter": "Bounded correction adapter evaluated after sensor damage.",
        "Associative Memory": "Fast prototype-association memory mechanism evaluated without gradients.",
        "Three-factor Local Plasticity": "Supervised local output-synapse learning rule evaluated in Gen-13.",
        "Reward-modulated Eligibility": "Delayed scalar-reward eligibility-trace learning mechanism.",
        "Reward Learning Baseline": "Matched conventional REINFORCE delayed-reward baseline.",
        "Sparse Spiking Credit": "One-event sparse translation of the local score-credit rule.",
        "Compiled Residual State": "The parameter-matched residual Conv1D plus LIF-state model under compiled steady-state inference.",
        "Event-driven Sparse Operator": "Event-coordinate accumulation substituted for the dense temporal input operator.",
        "Triton Event Kernel": "Custom Triton event-scatter execution path evaluated with behavioral-equivalence gates.",
        "Deterministic Evidence Synthesis": "Hashed claim-ledger generation that preserves authoritative program decisions without retraining.",
        "Dendritic Predictive Credit": "Fixed-topology three-factor hidden-layer rule using basal eligibility and residual apical teaching signals.",
    }
    for name, description in architecture_notes.items():
        write_note(vault / "Architectures" / f"{name}.md", f"---\ntype: architecture\ntags: [architecture, gen5]\n---\n\n# {name}\n\n{description}\n")
    benchmarks = {
        "Spiking-Heidelberg-Digits": "Event-based spoken-digit benchmark used by the SHD sprint series.",
        "N-MNIST": "Official event-vision MNIST conversion; native sensor is 34x34x2.",
        "Spiking Speech Commands": "Event-audio benchmark used by the continual-adaptation program.",
        "Cross-benchmark Evidence Ledger": "Hashed program-level synthesis spanning SHD, SSC, N-MNIST, embodied control, and systems audits.",
        "Embodied Delayed Reward": "Seeded tensorized control environment with delayed scalar reward.",
        "Sequential MNIST": "Row-stream MNIST temporal-memory benchmark.",
        "Event-coded MNIST": "Latency/event-coded MNIST representation benchmark.",
        "Delayed Contextual Binding": "Four-way synthetic delayed-association task with conflicting context mappings, distractors, and retention testing.",
        "SHD-SSC Matched Audio Evidence": "Canonical paired-seed evidence protocol spanning SHD and SSC causal, accuracy, systems, and reproduction gates.",
    }
    for name, description in benchmarks.items():
        write_note(vault / "Benchmarks" / f"{name}.md", f"---\ntype: benchmark\ntags: [benchmark, gen5]\n---\n\n# {name}\n\n{description}\n")

    for sprint_id, (status, conclusion, metrics, evidence) in DECISIONS.items():
        title = SPRINTS[sprint_id][0]
        write_note(vault / "Decisions" / f"{note_id(sprint_id)} Decision.md", f"""---
type: decision
sprint_id: {yaml_value(sprint_id)}
status: {yaml_value(status)}
evidence: {yaml_value(evidence)}
tags: [decision, {sprint_id}, gen5]
---

# {note_id(sprint_id)} Decision - {title}

## Decision
{conclusion}

## Key Evidence
{metrics}

## Graph Connections
- Phase: [[{note_id(sprint_id)}]]
- Source: {evidence}
""")

    write_note(vault / "Index.md", f"""---
type: index
updated: "2026-08-26"
tags: [index, gen5]
---

# A-SNN Research Knowledge Graph

> [!important] Current bounded experiment
> {CURRENT_STATE["index_summary"]} See [[Current State]].

## Program Map
- Current position: [[Current State]]
- Supported mechanism: [[Residual LIF State]]
- Supported systems result: [[Compiled Residual State]]
- Active evidence hypothesis: [[Residual LIF State]]
- Closed adaptive-mechanism branches: [[Matched Adaptive Mechanisms]] and [[Dual Memory Timescales]]
- Deferred research itinerary: [[LTH-Informed Itinerary Review]]

## Evidence Timeline
- SHD robustness and state discovery: [[Sprint-42]], [[Sprint-43]], [[Sprint-44]], [[Sprint-45]], [[Sprint-46]], [[Sprint-47]]
- Cross-dataset replication and matched efficiency: [[Sprint-48]], [[Sprint-49]]
- N-MNIST boundary: [[Gen-19]], [[Gen-20]]
- Matched adaptation and dual-memory falsification: [[Gen-21]], [[Gen-22]], [[Gen-23]]
- Compiled and event-sparse systems audit: [[Gen-24]], [[Gen-25]], [[Gen-26]], [[Gen-27]], [[Gen-28]]
- Program closure: [[Gen-29]]
- Stopped local-credit causal microtask: [[Gen-30]]
- Primary audio evidence consolidation: [[Evidence-1]]

## Claim Boundary
The internal positive neural result remains sample-specific residual LIF state on SHD and SSC. Evidence-1 must determine whether that effect survives canonical paired-seed controls and remains competitive with strong matched dense baselines. Active-dendrite research follows only after this decision. Independent replication, product value, structural-plasticity claims, and hardware-energy claims remain unauthorized.
""")

    generated = 0
    for sprint_id, relative, title, dataset, architecture, hypothesis in ARTIFACTS:
        source = outputs / relative
        if not source.exists():
            continue
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        metrics = artifact_metrics(data)
        decision = data.get("decision", {}) if isinstance(data.get("decision"), dict) else {}
        status = decision.get("status", "completed")
        metric_yaml = "\n".join(f"  {key}: {yaml_value(value)}" for key, value in metrics.items()) or "  {}"
        metric_body = "\n".join(f"- **{key}**: `{value}`" for key, value in metrics.items()) or "- No scalar summary was available."
        provenance = "recovered-provenance" if "recovery" in source.parent.name else "repository-artifact"
        write_note(vault / "Experiments" / f"EXP-{sprint_id.upper()}-{source.parent.name}.md", f"""---
type: experiment
run_id: {yaml_value(source.parent.name)}
sprint_id: {yaml_value(sprint_id)}
title: {yaml_value(title)}
dataset: "[[{dataset}]]"
architecture: "[[{architecture}]]"
hypothesis: "[[{hypothesis}]]"
status: {yaml_value(status)}
provenance: {yaml_value(provenance)}
source_artifact: {yaml_value((Path('gen5/outputs') / relative).as_posix())}
metrics:
{metric_yaml}
tags: [experiment, {sprint_id}, gen5]
---

# {title}

## Executive Summary
Imported from `{Path('gen5/outputs') / relative}`. Provenance: **{provenance}**.

## Context & Graph Connections
- Parent sprint: [[{note_id(sprint_id)}]]
- Benchmark: [[{dataset}]]
- Architecture: [[{architecture}]]
- Hypothesis: [[{hypothesis}]]

## Metrics Summary
{metric_body}
""")
        generated += 1
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic A-SNN Obsidian knowledge graph.")
    parser.add_argument("--vault-path", type=Path, default=REPOSITORY_ROOT / "obsidian_vault")
    parser.add_argument("--outputs-path", type=Path, default=REPOSITORY_ROOT / "gen5" / "outputs")
    args = parser.parse_args()
    generated = build_graph(args.vault_path, args.outputs_path)
    print(f"[obsidian_graph] Refreshed graph nodes and imported {generated} curated artifact note(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
