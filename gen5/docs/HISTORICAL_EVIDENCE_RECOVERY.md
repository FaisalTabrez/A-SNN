# Missing historical raw-evidence recovery checklist

Do not reconstruct missing per-seed values or label reported console summaries
as verified canonical runs. Recovery is read-only inspection of the original
Drive artifacts, followed by manifest/gate verification in a new import folder.
Stop decisions stay closed regardless of whether the originals can be recovered.

| Run | Requested original bundle | Current evidence |
| --- | --- | --- |
| Gen19 | gen19_nmnist_state_replication_bundle.zip and JSON/records/progress | Explicit terminal aggregate recovery only |
| N-MNIST accuracy | nmnist_accuracy_benchmark_bundle.zip and JSON/CSVs/timing/checkpoints | Three-seed conventional-CNN log recovery only |
| Gen26 | gen26 numerical-fidelity JSON/CSV/manifest/bundle | Analysis text only |
| Gen30 | dendritic predictive-credit JSON/per-seed records/manifest/bundle | Ten-seed console summary only |

1. Export the original folder without overwriting it or retraining in place.
2. Import the ZIP into a separate review directory. Check duplicate/unsafe member
   paths, CRC, manifest coverage and hashes before inspecting JSON.
3. Recompute metrics and registered gate decisions from original per-seed rows.
4. Preserve the outer ZIP checksum, original source/runtime/protocol metadata,
   recovery origin, and any discrepancies in an analysis note.
5. If artifacts are unavailable, retain the weaker provenance classification.
   A new run would be new evidence and needs separate authorization/protocol.

No original Drive session or missing bundle was available during the local
hardening work; this checklist is not a completed recovery.
