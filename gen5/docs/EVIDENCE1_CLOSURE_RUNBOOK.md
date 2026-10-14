# Evidence-1 closure tooling and honest remaining gates

Implementation date: 2026-10-14. The user authorized completion and publication.
Evidence-1 stays active; the recorded decision remains `causal_audit_only`.

## Exact original source and public numeric evidence

`gen5/repro/evidence1_original_sources.zip` contains all 73 original package
modules. The previously uncommitted v1 runner was recovered from the original
audit's executed notebook snapshot. Each source byte hash matches the original
protocol. Other modules were recovered from the original Git base, not recreated
from later hardened code. The package is deterministic and includes its receipt.

This frozen source intentionally retains the known v1 defects. It is for
review/exact-source replay, not the default new-run implementation. Replays must
use new output directories; receipts cannot repair old defects retroactively.

`gen5/docs/evidence1_release/` retains the original protocol and the two complete
numeric result records. Student-t intervals and every registered gate recompute
without training or checkpoint loading. The same ten seeds must not be pooled as
twenty independent observations. Full historical dependency/driver metadata was
not returned and cannot be reconstructed honestly. The original large checkpoint
ZIPs are preserved locally and excluded from Git; SHA-256 identities are pinned
in `evidence1_closure.py`. Source/metadata publication is not a claim that all
checkpoint/data binaries are hosted in the repository.

## No-training CUDA diagnosis

Open `gen5/notebooks/evidence1_fidelity_diagnostic.ipynb` on a CUDA host. Set
`RELEASE_REVISION` to the full published commit SHA. Point `BUNDLE` and
`DATA_ROOT` at the original result ZIP and original encoded caches. Leave
`PREFLIGHT_ONLY=True` for integrity checks. To execute, choose a new output folder
and set it to false. Dataset downloads and retraining are not needed.

Equivalent command:

```text
python gen5/tools/evidence1_fidelity_diagnostic.py --bundle research_restart_results.zip --data-root PATH_TO_ORIGINAL_DATA --output-dir NEW_DIAGNOSTIC_FOLDER
```

The subprocess imports the exact old package and loads only manifest-verified
SHD310/SSC311 validation checkpoints with `weights_only=True`. It consumes
hash-matching SHD train/SSC validation caches; no final-test input or label is
evaluated. Every validation batch, including the tail shape, records absolute
logit differences, prediction agreement, worst sample/output and eager-repeat
drift. Registered reduce-overhead, default Inductor and AOT-eager graph capture
are separate diagnostic cases, not new gate winners. Runtime/backend metadata is
saved. There is no throughput or energy claim.

There is no CUDA device on the local audit host. Preflight verifies the original
bundle and all 73 sources; a real discrepancy/cause remains unresolved until a
GPU run returns. Different CUDA/Torch versions do not establish reproduction of
the original L4 environment. Even a passing new diagnostic does not rewrite the
old systems failure or unlock a competitive claim.

## Clean-source software verification

```text
python gen5/tools/verify_clean_checkout.py --repository https://github.com/FaisalTabrez/A-SNN.git --revision FULL_COMMIT_SHA --work-dir NONEXISTENT_FOLDER
```

This creates a detached clean checkout, runs source coverage, all CPU tests,
offline notebook execution and strict graph checks. It reuses the current
Python dependencies. It does not certify an independent executor, fresh runtime
installation, CUDA research replication or human approval. The committed CPU
CI workflow additionally installs its declared environment on GitHub.

## Missing raw evidence

```text
python gen5/tools/audit_historical_recovery.py --search-root PATH_TO_EXPORTED_DRIVE --output NEW_RECOVERY_REPORT.json
```

The scanner rejects unsafe/duplicate ZIP members and CRC faults. A matching
filename is only a candidate. It never certifies missing per-seed metrics or
reopens a stop decision. The local Documents search found no Gen19, N-MNIST
accuracy, Gen26 or Gen30 candidate bundle. Original Drive exports are still
needed; a new run is not a substitute for missing history.

## Human review and later controls

Review the original outcomes and audit with the mentor, record the reviewer's
name/date and chosen claim boundary, and sign/hash the decision document. Agents
cannot manufacture this approval or independently review their own execution.
The decision must preserve accuracy/systems failures and historical test exposure.

`EVIDENCE1_FOLLOWUP_PROTOCOL_DRAFT.md` describes within-class shuffling, separated
spike/membrane ablations, retrained analog/direct controls and matched plasticity
confounds. Control primitives and unit tests exist, but no new study is registered
or run. Dataset, margins, power and multiplicity decisions require preregistration
after the reviewed lock. Human approval and an untouched holdout remain entry gates.

Graphify is refreshed structurally. No backend was approved for a new semantic
pass; a complete semantic-freshness claim is still deliberately withheld.
