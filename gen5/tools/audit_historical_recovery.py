"""Find potential missing bundles without extracting or treating filenames as recovered proof."""
from pathlib import Path
import argparse
import json
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gen5'))
from ammc_gen5.evidence1_closure import safe_members, sha256_file
TERMS = {'gen19': ('gen19',), 'nmnist_accuracy': ('nmnist', 'accuracy'),
         'gen26': ('gen26',), 'gen30': ('gen30',)}


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        names = safe_members(archive)
        manifests = [n for n in names if n.endswith('manifest.json')]
        # CRC and manifest presence alone never establish original gate outcomes.
        return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path),
                'member_count': len(names), 'manifest_candidates': manifests,
                'crc_and_paths_pass': True, 'metrics_recomputed': False,
                'recovery_complete': False, 'classification': 'candidate_requires_source_and_gate_review'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 1, 'search_roots': [str(p.resolve()) for p in args.search_root],
              'runs': {name: [] for name in TERMS}, 'decision_claims_reopened': False}
    seen = set()
    for root in args.search_root:
        if not root.is_dir(): raise ValueError('Missing explicitly selected search root: ' + str(root))
        for path in root.rglob('*.zip'):
            resolved = path.resolve()
            if resolved in seen: continue
            seen.add(resolved)
            for name, terms in TERMS.items():
                if all(term in path.name.lower() for term in terms):
                    try: result = inspect(path)
                    except (ValueError, OSError, zipfile.BadZipFile) as error:
                        result = {'path': str(resolved), 'error': str(error), 'recovery_complete': False}
                    report['runs'][name].append(result)
    report['status'] = 'candidates_require_review' if any(report['runs'].values()) else 'missing_originals_not_found'
    if args.output.exists(): raise ValueError('Use a new report filename; no existing evidence is overwritten')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['status'])


if __name__ == '__main__': main()
