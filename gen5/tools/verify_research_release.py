"""Fail if the release lacks implementation/tests or notebook snapshots drift."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gen5/tools'))
from build_research_notebook import SNAPSHOTS


def verify(root=ROOT, require_tracked=False):
    root = Path(root)
    required = SNAPSHOTS + ['gen5/tools/build_research_notebook.py', 'gen5/tools/verify_research_notebook.py',
        'gen5/tools/verify_research_release.py', 'gen5/tests/test_evidence1_audio_contract.py',
        'gen5/tests/test_evidence1_audit_regressions.py', 'gen5/docs/EVIDENCE1_AUDIO_PREREGISTRATION.md',
        'gen5/docs/EVIDENCE1_HARDENING_2026_10_14.md', 'gen5/notebooks/evidence1_research_restart.ipynb',
        'evidence1_research_restart.ipynb', 'requirements-research.txt', 'pyproject.toml',
        '.github/workflows/research-contracts.yml', 'gen5/ammc_gen5/evidence1_closure.py',
        'gen5/ammc_gen5/evidence1_followup_controls.py', 'gen5/tools/build_original_evidence_source.py',
        'gen5/tools/evidence1_fidelity_diagnostic.py', 'gen5/tools/audit_historical_recovery.py',
        'gen5/tools/verify_clean_checkout.py', 'gen5/tools/build_diagnostic_notebook.py',
        'gen5/tests/test_evidence1_closure_contract.py', 'gen5/tests/test_evidence1_followup_controls.py',
        'gen5/notebooks/evidence1_fidelity_diagnostic.ipynb', 'gen5/repro/evidence1_original_sources.zip',
        'gen5/docs/evidence1_release/original_protocol.json', 'gen5/docs/evidence1_release/primary_results.json',
        'gen5/docs/evidence1_release/repeat_results.json', 'gen5/docs/EVIDENCE1_CLOSURE_RUNBOOK.md']
    missing = [name for name in required if not (root / name).is_file()]
    errors = []
    untracked = []
    uncommitted = []
    if not missing:
        notebook = json.loads((root / 'gen5/notebooks/evidence1_research_restart.ipynb').read_text())
        snapshots = None
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            text = ''.join(cell['source']); tree = ast.parse(text)
            if text.startswith('SNAPSHOTS = '): snapshots = ast.literal_eval(tree.body[0].value)
        if snapshots is None or set(snapshots) != set(SNAPSHOTS): errors.append('Snapshot set mismatch')
        else:
            for name, row in snapshots.items():
                data = base64.b64decode(row['base64'], validate=True)
                if hashlib.sha256(data).hexdigest() != row['sha256'] or data != (root / name).read_bytes():
                    errors.append('Snapshot drift: ' + name)
        if (root / 'evidence1_research_restart.ipynb').read_bytes() != (root / 'gen5/notebooks/evidence1_research_restart.ipynb').read_bytes():
            errors.append('Notebook copies differ')
    if require_tracked:
        for name in required:
            if subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', name],
                              capture_output=True).returncode:
                untracked.append(name)
            if (subprocess.run(['git', '-C', str(root), 'cat-file', '-e', 'HEAD:' + name], capture_output=True).returncode or
                subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', name], capture_output=True).returncode):
                uncommitted.append(name)
    return {'missing': missing, 'snapshot_errors': errors, 'untracked': untracked, 'uncommitted': uncommitted,
            'release_committed_verified': require_tracked and not (missing or errors or untracked or uncommitted),
            'passed': not (missing or errors or untracked or uncommitted)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tracked', action='store_true')
    result = verify(require_tracked=parser.parse_args().require_tracked)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
