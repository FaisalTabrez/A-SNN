"""Validation-only diagnosis of frozen SHD310/SSC311. Never trains or evaluates test data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]


def compare_logits(eager, compiled):
    import torch
    if eager.shape != compiled.shape or eager.ndim != 2 or eager.shape[0] == 0:
        raise ValueError('Invalid fidelity output shape')
    if not torch.isfinite(eager).all() or not torch.isfinite(compiled).all():
        raise FloatingPointError('Nonfinite fidelity logits')
    delta = (eager - compiled).abs()
    if not torch.isfinite(delta).all(): raise FloatingPointError('Nonfinite fidelity differences')
    flat = int(delta.argmax()); sample, output = divmod(flat, eager.shape[1])
    return {'samples': len(eager), 'maximum_logit_difference': float(delta.max()),
            'mean_absolute_difference': float(delta.mean()),
            'class_agreements': int((eager.argmax(1) == compiled.argmax(1)).sum()),
            'worst_sample_in_batch': sample, 'worst_output_index': output,
            'eager_worst_logit': float(eager[sample, output]),
            'compiled_worst_logit': float(compiled[sample, output])}


def verify_source_archive(path, protocol):
    expected = protocol['source_sha256']
    with zipfile.ZipFile(path) as archive:
        from ammc_gen5.evidence1_closure import safe_members
        names = safe_members(archive)
        if set(names) != {'ammc_gen5/' + n for n in expected} | {'source_receipt.json'}:
            raise ValueError('Original source package coverage mismatch')
        receipt = json.loads(archive.read('source_receipt.json'))
        if receipt['protocol_sha256'] != protocol['sha256']:
            raise ValueError('Original source protocol mismatch')
        content = {}
        for name, checksum in expected.items():
            data = archive.read('ammc_gen5/' + name)
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError('Original source checksum mismatch: ' + name)
            content['ammc_gen5/' + name] = data
        return content


def worker(job_path):
    # Import the frozen package first in a fresh subprocess. The current package
    # never shadows historical model/loaders, even when a caller imported v2.
    job = json.loads(Path(job_path).read_text())
    sys.path.insert(0, job['source_root'])
    import torch
    from ammc_gen5 import evidence1_audio as original
    import importlib.util
    spec = importlib.util.spec_from_file_location('asnn_current_receipts', ROOT / 'gen5/ammc_gen5/artifact_integrity.py')
    receipts = importlib.util.module_from_spec(spec); spec.loader.exec_module(receipts)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA disappeared after preflight')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config_values = job['protocol']['config']
    config_values.update({key: tuple(config_values[key]) for key in ('datasets', 'seeds', 'temporal_levels')})
    config = original.EvidenceConfig(**config_values)
    report = {'schema': 1, 'kind': 'validation_only_diagnostic_not_new_research_decision',
              'bundle_sha256': job['bundle_sha256'], 'protocol_sha256': job['protocol']['sha256'],
              'runtime': receipts.runtime_identity(torch), 'original_runtime': job['original_runtime'],
              'trained': False, 'test_evaluations': 0, 'registered_tolerance': 1e-4, 'trials': []}
    modes = [('registered_reduce_overhead', {'fullgraph': True, 'mode': 'reduce-overhead'}),
             ('inductor_default', {'fullgraph': True, 'mode': 'default'}),
             ('aot_eager_capture', {'fullgraph': True, 'backend': 'aot_eager'})]
    for trial in job['trials']:
        encoded = torch.load(trial['cache'], map_location='cpu', weights_only=True)
        events, labels = encoded['events'], encoded['labels']
        if trial['dataset'] == 'shd':
            _, _, validation, _ = original._stratified_split(events, labels, config.validation_fraction, config.split_seed)
        else: validation = events
        if list(validation.shape) != trial['validation_shape']:
            raise ValueError('Frozen validation split shape mismatch')
        original.seed_everything(trial['seed'])
        model = original.make_model('residual_lif', config, 700, trial['classes'], trial['duration']).cuda().eval()
        checkpoint = torch.load(trial['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['best_state'], strict=True)
        result = {'dataset': trial['dataset'], 'seed': trial['seed'], 'cases': []}
        for name, options in modes:
            case = {'backend_case': name, 'batches': [], 'status': 'measured'}
            try:
                torch._dynamo.reset()
                compiled = torch.compile(model, **options)
                with torch.inference_mode():
                    for offset in range(0, len(validation), config.batch_size):
                        probe = validation[offset:offset + config.batch_size].cuda()
                        eager = model(probe)
                        compiled(probe)  # Warm each input shape without timing claims.
                        row = compare_logits(eager, compiled(probe))
                        row['validation_offset'] = offset
                        row['eager_repeat_maximum_difference'] = compare_logits(eager, model(probe))['maximum_logit_difference']
                        case['batches'].append(row)
                case['maximum_logit_difference'] = max(b['maximum_logit_difference'] for b in case['batches'])
                case['class_agreement'] = sum(b['class_agreements'] for b in case['batches']) / len(validation)
                case['diagnostic_within_registered_logit_tolerance'] = case['maximum_logit_difference'] <= 1e-4
            except Exception as error:
                case['status'] = 'numerical_failure' if isinstance(error, FloatingPointError) else 'diagnostic_failed'
                case['error'] = f'{type(error).__name__}: {error}'
            result['cases'].append(case)
            report['trials'] = [*report['trials'], result] if result not in report['trials'] else report['trials']
            receipts.atomic_json(Path(job_path).parent / 'diagnostic_results.json', report)
        del model, validation, events, encoded
        torch.cuda.empty_cache()
    print('Validation-only diagnostic complete; no registered decision changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, default=ROOT / 'gen5/repro/evidence1_original_sources.zip')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'gen5'))
    from ammc_gen5.evidence1_closure import verify_return_bundle, sha256_file
    from ammc_gen5.artifact_integrity import atomic_json
    import torch
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new diagnostic directory; no prior evidence is overwritten')
    evidence = verify_return_bundle(args.bundle)
    sources = verify_source_archive(args.source_archive, evidence['protocol'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {'source_files_verified': len(sources), 'bundle_sha256': evidence['bundle_sha256'],
                 'cuda_available': torch.cuda.is_available(), 'training_requested': False, 'test_evaluations': 0,
                 'status': 'preflight_only' if args.preflight_only else 'ready' if torch.cuda.is_available() else 'blocked_no_cuda'}
    atomic_json(args.output_dir / 'preflight.json', preflight)
    if args.preflight_only:
        print(json.dumps(preflight)); return
    if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU; only integrity preflight completed')
    if args.data_root is None: raise ValueError('--data-root with the original verified encoded caches is required')
    trials = []
    with zipfile.ZipFile(args.bundle) as archive:
        for dataset, seed in [('shd', 310), ('ssc', 311)]:
            data = evidence['data'][dataset]
            cache_name = ('shd_train_t64_c700_d1400ms.pt' if dataset == 'shd' else 'ssc_validation_t64_c700_d1000ms.pt')
            cache = args.data_root.resolve() / dataset / cache_name
            if sha256_file(cache) != data['cache_sha256'][cache_name]:
                raise ValueError('Frozen validation cache identity changed: ' + str(cache))
            member = f'evidence1_confirmation/{dataset}_{seed}_residual_lif/checkpoint.pt'
            checkpoint = args.output_dir.resolve() / f'{dataset}_{seed}_checkpoint.pt'
            checkpoint.write_bytes(archive.read(member))  # Already verified against both manifests.
            trials.append({'dataset': dataset, 'seed': seed, 'checkpoint': str(checkpoint), 'cache': str(cache),
                           'classes': data['classes'], 'duration': data['duration'], 'validation_shape': data['split_shapes'][2]})
    source_root = args.output_dir.resolve() / 'frozen_source'
    for name, content in sources.items():
        target = source_root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    job = {'source_root': str(source_root), 'trials': trials, 'protocol': evidence['protocol'],
           'bundle_sha256': evidence['bundle_sha256'], 'original_runtime': evidence['environment']}
    job_path = args.output_dir.resolve() / 'diagnostic_job.json'; atomic_json(job_path, job)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(job_path)], check=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker': worker(sys.argv[2])
    else: main()
