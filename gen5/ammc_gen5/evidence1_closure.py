"""Read-only closure checks. Receipts establish integrity, not execution independence."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import statistics
import zipfile

ORIGINAL_BUNDLES = {
    'primary': '9c97119530f9396dcdc889e5959c9e72528f48915a3e2e281e05c2c087e9b702',
    'repeat': 'f254b552374cb20c695ec5173438e166e2d9a954c98399703869ed9a98b58f41',
}
ORIGINAL_PROTOCOL = '5684278f73ff052064cc6be14d2988a94331a74ca01be99662e46ff0c5ed40c3'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive):
    """Reject traversal, duplicate names, symlinks and resource-exhausting packages."""
    names = archive.namelist()
    if len(names) != len(set(names)) or len(names) > 20000:
        raise ValueError('Duplicate or excessive archive members')
    if sum(i.file_size for i in archive.infolist()) > 10 * 1024**3:
        raise ValueError('Archive expands beyond the 10 GiB inspection limit')
    for info in archive.infolist():
        name = info.orig_filename
        path = PurePosixPath(name)
        if any(ord(c) < 32 for c in name) or chr(92) in name or ':' in name or path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'Unsafe archive member: {name}')
    bad = archive.testzip()
    if bad:
        raise ValueError(f'Archive CRC failed: {bad}')
    return names


def verify_return_bundle(path, *, pinned=True):
    checksum = sha256_file(path)
    if pinned and checksum not in ORIGINAL_BUNDLES.values():
        raise ValueError('Not one of the two frozen original Evidence-1 bundles')
    with zipfile.ZipFile(path) as archive:
        names = safe_members(archive)
        manifest = json.loads(archive.read('return_manifest.json'))
        manifest = manifest.get('files', manifest)
        if set(names) != set(manifest) | {'return_manifest.json'}:
            raise ValueError('Outer manifest coverage mismatch')
        for name, receipt in manifest.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if archive.getinfo(name).file_size != receipt['bytes'] or digest.hexdigest() != receipt['sha256']:
                raise ValueError(f'Outer manifest mismatch: {name}')
        prefix = 'evidence1_confirmation/'
        protocol = json.loads(archive.read(prefix + 'protocol.json'))
        unsigned = {k: v for k, v in protocol.items() if k != 'sha256'}
        signature = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if signature != protocol['sha256'] or (pinned and signature != ORIGINAL_PROTOCOL):
            raise ValueError('Frozen protocol identity mismatch')
        # Verify the nested manifest as well, not just its outer checksum.
        inner = json.loads(archive.read(prefix + 'manifest.json'))['files']
        covered = {prefix + name for name in inner}
        actual = {n for n in names if n.startswith(prefix)} - {prefix + 'manifest.json'}
        if covered != actual:
            raise ValueError('Inner manifest coverage mismatch')
        for name, receipt in inner.items():
            if manifest[prefix + name] != receipt:
                raise ValueError(f'Inner/outer manifest disagree: {name}')
        results = json.loads(archive.read(prefix + 'results.json'))
        rows = results['records']
        expected = {(d, s, a) for d in ('shd', 'ssc') for s in range(310, 320)
                    for a in ('residual_lif', 'conv1d', 'tcn', 'gru')}
        keys = [(r['dataset'], r['seed'], r['arm']) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Missing, duplicate or unexpected frozen trial keys')
        for row in rows:
            trial = f"{prefix}{row['dataset']}_{row['seed']}_{row['arm']}/record.json"
            if json.loads(archive.read(trial)) != row:
                raise ValueError('Aggregate/per-trial record mismatch')
        return {'bundle_sha256': checksum, 'manifest': manifest, 'protocol': protocol,
                'environment': json.loads(archive.read(prefix + 'environment.json')),
                'data': {d: json.loads(archive.read(prefix + d + '_data.json')) for d in ('shd', 'ssc')},
                'results': results, 'member_count': len(names)}


def estimate(values):
    from scipy.stats import t
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Paired interval needs finite independent seed rows')
    mean, deviation = statistics.fmean(values), statistics.stdev(values)
    half = float(t.ppf(.975, len(values) - 1)) * deviation / math.sqrt(len(values))
    return {'mean': mean, 'standard_deviation': deviation, 'ci95_lower': mean - half,
            'ci95_upper': mean + half, 'n': len(values)}


def recompute_results(results):
    """Independent arithmetic implementation; never loads a checkpoint or test input."""
    datasets, maximum_error = {}, 0.
    for dataset in ('shd', 'ssc'):
        groups = {a: {r['seed']: r for r in results['records'] if r['dataset'] == dataset and r['arm'] == a}
                  for a in ('residual_lif', 'conv1d', 'tcn', 'gru')}
        full = groups['residual_lif']; seeds = range(310, 320)
        fields = {'accuracy': {a: estimate([g[s]['accuracy'] for s in seeds]) for a, g in groups.items()},
                  'causal_deltas': {mode: estimate([full[s]['accuracy'] - full[s][mode + '_accuracy'] for s in seeds])
                                    for mode in ('zero_state', 'shuffled_state')},
                  'paired_baseline_deltas': {a: estimate([full[s]['accuracy'] - groups[a][s]['accuracy'] for s in seeds])
                                            for a in ('conv1d', 'tcn', 'gru')}}
        for field, entries in fields.items():
            for arm, stats in entries.items():
                for key, value in stats.items():
                    reported = results['summary']['datasets'][dataset][field][arm][key]
                    if not isinstance(reported, (int, float)) or not math.isfinite(reported):
                        raise ValueError('Nonfinite or invalid reported statistic')
                    maximum_error = max(maximum_error, abs(value - reported))
        causal = all(v['ci95_lower'] > 0 for v in fields['causal_deltas'].values())
        accuracy = all(v['ci95_lower'] > -.01 for v in fields['paired_baseline_deltas'].values())
        systems = all(r['systems']['status'] == 'measured' and
                      math.isfinite(r['systems']['maximum_logit_difference']) and
                      r['systems']['maximum_logit_difference'] <= 1e-4 and
                      r['systems']['prediction_agreement'] == 1. and r['systems']['speedup'] > 1.
                      for r in full.values())
        for key, value in [('causal_gate', causal), ('accuracy_gate', accuracy), ('systems_gate', systems)]:
            if results['summary']['datasets'][dataset][key] is not value:
                raise ValueError('Registered gate recomputation mismatch')
            fields[key] = value
        datasets[dataset] = fields
    if maximum_error > 1e-10:
        raise ValueError('Reported Student-t statistics do not recompute')
    decision = ('stop_causal_claim' if not all(d['causal_gate'] for d in datasets.values()) else
                'causal_audit_only' if not all(d['accuracy_gate'] for d in datasets.values()) else
                'model_gates_pass_systems_unconfirmed' if not all(d['systems_gate'] for d in datasets.values()) else
                'model_and_systems_gates_pass_replication_pending')
    if decision != results['summary']['decision']:
        raise ValueError('Registered decision recomputation mismatch')
    return {'datasets': datasets, 'decision': decision, 'maximum_statistic_error': maximum_error,
            'execution_independence_certified': False, 'human_reviewed_lock': False}
