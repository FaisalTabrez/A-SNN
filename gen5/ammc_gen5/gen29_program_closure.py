"""Historical source-linked synthesis, never an independent metric certificate."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import zipfile


EVIDENCE_DOCUMENTS = {
    "phase47": "gen5/outputs/shd_residual_state_contribution_cuda_2026-08-10/analysis.md",
    "phase48": "gen5/outputs/ssc_residual_lif_replication_cuda_2026-08-10/analysis.md",
    "phase49": "gen5/outputs/ssc_efficiency_baselines_cuda_2026-08-10/analysis.md",
    "gen18": "gen5/docs/GEN18_LOCAL_CREDIT_REPLICATION_ANALYSIS.md",
    "gen19": "gen5/docs/GEN19_NMNIST_STATE_REPLICATION_ANALYSIS.md",
    "gen20": "gen5/docs/PROGRAM_SANITY_CHECK_AFTER_GEN20.md",
    "gen21": "gen5/docs/GEN21_MATCHED_CAUSAL_MECHANISMS_ANALYSIS.md",
    "gen22": "gen5/docs/GEN22_DUAL_MEMORY_REPLICATION_ANALYSIS.md",
    "gen23": "gen5/docs/GEN23_BOUNDARY_CONSOLIDATION_ANALYSIS.md",
    "gen24": "gen5/docs/GEN24_COMPILED_RESIDUAL_STATE_ANALYSIS.md",
    "gen25": "gen5/docs/GEN25_EVENT_DRIVEN_SPARSE_AUDIT_ANALYSIS.md",
    "gen26": "gen5/docs/GEN26_SPARSE_NUMERICAL_FIDELITY_ANALYSIS.md",
    "gen27": "gen5/docs/GEN27_TRAINED_THRESHOLD_ROBUSTNESS_ANALYSIS.md",
    "gen28": "gen5/docs/GEN28_TRITON_EVENT_KERNEL_ANALYSIS.md",
}


CLAIMS = (
    ("neural_mechanism", "Residual LIF state is beneficially sample-specific on SHD and SSC", "supported", "Phase 47 and Phase 48"),
    ("generalization", "Residual-state identity generalizes to N-MNIST event vision", "rejected", "Gen-19 and Gen-20"),
    ("competitiveness", "Residual LIF matches the best parameter-matched SSC predictor", "rejected", "Phase 49"),
    ("adaptive_mechanism", "Dynamic topology adds matched continual-learning benefit", "rejected", "Gen-21"),
    ("adaptive_mechanism", "Dual memory timescales outperform one adaptive memory", "rejected", "Gen-22 and Gen-23"),
    ("adaptive_mechanism", "Learned delays add matched causal benefit", "rejected", "Gen-21"),
    ("adaptive_mechanism", "Local reward credit adds reliable matched adaptation", "rejected", "Gen-18 and Gen-21"),
    ("systems", "Compilation removes the eager residual-LIF loop confound", "supported", "Gen-24"),
    ("systems", "Sparse accumulation preserves trained task behavior", "supported-scope-limited", "Gen-27"),
    ("systems", "Generic COO is a viable production event-sparse path", "rejected", "Gen-25"),
    ("systems", "The Triton event kernel reaches compiled-dense throughput parity", "rejected", "Gen-28"),
    ("hardware", "AMMC-SNN reduces measured hardware energy", "untested", "No wall-plug or neuromorphic measurement"),
)


@dataclass
class Gen29Result:
    sources: list[dict]
    claims: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen29_program_closure.json"
        claims_path = output / "gen29_program_closure_claims.csv"
        report_path = output / "gen29_program_closure_report.md"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(claims_path, self.claims)
        report_path.write_text(_render_report(self), encoding="utf-8")
        return {"json": str(json_path), "claims_csv": str(claims_path), "report": str(report_path)}


def run_gen29(repo_root: str | pathlib.Path) -> Gen29Result:
    root = pathlib.Path(repo_root).resolve()
    sources = []
    for phase, relative in EVIDENCE_DOCUMENTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {phase} evidence: {path}")
        text = path.read_text(encoding='utf-8')
        if len(text.strip()) < 120 or not text.lstrip().startswith('#'):
            raise ValueError(f'Empty/placeholder evidence for {phase}: {path}')
        sources.append({
            "phase": phase,
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    claims = [
        {"category": category, "claim": claim, "status": status, "evidence": evidence,
         "verification": "historical_source_statement_not_recomputed_metric"}
        for category, claim, status, evidence in CLAIMS
    ]
    supported_adaptive = [
        row["claim"] for row in claims
        if row["category"] == "adaptive_mechanism" and row["status"] == "supported"
    ]
    decision = {
        "status": "historical_synthesis_requires_review",
        "metric_validation": "not_performed",
        "source_integrity_only": True,
        "required_evidence_count": len(EVIDENCE_DOCUMENTS),
        "claim_count": len(claims),
        "original_research_question_answered": "partially",
        "supported_adaptive_mechanisms": supported_adaptive,
        "supported_neural_mechanism": "cooperative sample-specific residual LIF state on SHD and SSC",
        "supported_systems_path": "historical Gen24 frozen-workload result, not Evidence1 fidelity certification",
        "closed_systems_paths": ["generic COO", "current Triton event-scatter kernel"],
        "hardware_energy_claim_authorized": False,
        "next_milestone": "reviewed_evidence1_decision_before_new_mechanisms",
    }
    return Gen29Result(sources=sources, claims=claims, decision=decision)


def bundle_gen29_artifacts(paths: dict[str, str], output_dir: str | pathlib.Path) -> dict[str, str]:
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(path) for path in paths.values()]
    manifest = output / "gen29_program_closure_manifest.json"
    manifest.write_text(json.dumps({
        "files": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in files
        ]
    }, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen29_program_closure_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files + [manifest]:
            bundle.write(path, arcname=path.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _render_report(result: Gen29Result) -> str:
    lines = [
        "# Gen-29 historical evidence synthesis",
        "",
        "Source integrity only. Claim statuses below reproduce historical statements; this generator does not recompute experimental metrics or issue a research decision lock.",
        "",
        "## Executive conclusion",
        "",
        "The original biological-learning objective remains unachieved. Matched controls do not support dynamic topology, dual-memory advantage, learned-delay benefit, or reliable local reward credit in the tested implementations. The replicated positive neural result is cooperative, sample-specific residual LIF state on SHD and SSC.",
        "",
        "Gen24's compiled result has frozen-workload scope. Current Evidence1 numerical fidelity remains unconfirmed. Generic COO and the tested Triton kernel remain closed, and no hardware-energy claim is authorized.",
        "",
        "## Claim ledger",
        "",
        "| Category | Claim | Status | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in result.claims:
        lines.append(f"| {row['category']} | {row['claim']} | {row['status']} | {row['evidence']} |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Status: {result.decision['status']}",
        f"- Next milestone: {result.decision['next_milestone']}",
        "- This historical synthesis does not authorize new experiments or supersede the later Gen30 stop/Evidence1 program.",
        "",
        "## Hashed evidence",
        "",
    ])
    for row in result.sources:
        lines.append(f"- {row['phase']}: {row['path']} ({row['sha256']})")
    return "\n".join(lines) + "\n"


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
