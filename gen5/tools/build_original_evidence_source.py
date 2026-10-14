"""Package exact original source from Git + a hash-matching original notebook snapshot."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gen5'))
from ammc_gen5.evidence1_closure import verify_return_bundle, recompute_results


def build(bundle, notebook, destination):
    evidence = verify_return_bundle(bundle)
    payload = json.loads(Path(notebook).read_text(encoding='utf-8'))
    cell = next(c for c in payload['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith('SNAPSHOTS = '))
    snapshots = ast.literal_eval(ast.parse(''.join(cell['source'])).body[0].value)
    revision = evidence['environment']['git_commit']
    sources = {}
    for name, expected in evidence['protocol']['source_sha256'].items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Unexpected source layout')
        relative = 'gen5/ammc_gen5/' + name
        if relative in snapshots:
            content = base64.b64decode(snapshots[relative]['base64'], validate=True)
        else:
            content = subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + relative])
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Original source cannot be recovered exactly: ' + name)
        sources['ammc_gen5/' + name] = content
    receipt = {'schema': 1, 'kind': 'frozen_original_v1_not_hardened_v2', 'original_git_base': revision,
               'protocol_sha256': evidence['protocol']['sha256'], 'package_source_sha256': evidence['protocol']['source_sha256'],
               'original_runtime': evidence['environment'], 'runtime_complete': False,
               'warning': 'Preserves known v1 defects for exact-source replay. Never reuse old outputs or claim a new decision.'}
    sources['source_receipt.json'] = (json.dumps(receipt, indent=2) + '\n').encode()
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise ValueError('Destination exists; no original package is overwritten')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return evidence, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--snapshot-notebook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path)
    args = parser.parse_args()
    evidence, receipt = build(args.bundle, args.snapshot_notebook, args.output)
    if args.metadata_dir:
        args.metadata_dir.mkdir(parents=True, exist_ok=True)
        for name, value in [('original_protocol.json', evidence['protocol']), ('primary_results.json', evidence['results']),
                            ('original_source_receipt.json', receipt), ('primary_recomputation.json', recompute_results(evidence['results']))]:
            destination = args.metadata_dir / name
            if destination.exists(): raise ValueError('Metadata destination exists: ' + str(destination))
            destination.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Exact original package sources verified:', len(receipt['package_source_sha256']))


if __name__ == '__main__': main()
