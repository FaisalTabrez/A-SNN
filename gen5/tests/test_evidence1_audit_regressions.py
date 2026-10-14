"""CPU reproductions of the October audit failures, with fail-closed assertions."""
from dataclasses import replace
from pathlib import Path
import contextlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from ammc_gen5 import evidence1_audio as e
from ammc_gen5.artifact_integrity import bind_protocol, seal_files, verify_files


def records(config):
    return [{'dataset': d, 'seed': s, 'arm': a, 'accuracy': .8,
             'zero_state_accuracy': .6, 'shuffled_state_accuracy': .6,
             'systems': {'status': 'measured', 'speedup': 2.,
                         'prediction_agreement': 1., 'maximum_logit_difference': 0.}}
            for d in config.datasets for s in config.seeds for a in e.ARMS]


class AuditRegressionTests(unittest.TestCase):
    def test_fits_cached_summary_requires_bound_unchanged_artifacts(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import fits_validation_reference as fits
        training = types.ModuleType('common.training')
        training.create_optimizer = lambda *args: self.fail('Cached result must not retrain')
        training.run_epoch = lambda *args: self.fail('Cached result must not reevaluate')
        common = types.ModuleType('common'); common.training = training
        cfg = types.SimpleNamespace(epochs=1, sizes=(2, 3))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'common': common, 'common.training': training}):
            root = Path(directory)
            names = ['validation_reference_summary.json', 'validation_reference_checkpoint.pt', 'validation_reference_history.json']
            summary = {'selection': 'validation_only', 'test_evaluations': 1}
            (root / names[0]).write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'Unbound FiTS'):
                fits.validation_fit(None, cfg, root, 'cpu', None, None, valid_loader=object())
            (root / 'validation_reference_config.json').write_text(json.dumps(vars(cfg)))
            (root / names[1]).write_bytes(b'opaque-test-checkpoint-not-loaded')
            (root / names[2]).write_text('[]')
            binding = {'schema': 2, 'configuration': fits.identity(vars(cfg)), 'protocol': None}
            seal_files(root, names, 'validation_reference_complete_integrity.json', binding)
            self.assertEqual(fits.validation_fit(None, cfg, root, 'cpu', None, None, valid_loader=object()), summary)
            (root / names[0]).write_text(json.dumps({**summary, 'test_accuracy': .99}))
            with self.assertRaisesRegex(ValueError, 'integrity/binding mismatch'):
                fits.validation_fit(None, cfg, root, 'cpu', None, None, valid_loader=object())

    def test_fly_refuses_unbound_or_changed_cached_output_before_simulation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        import fly_reference as fly
        brian = types.ModuleType('brian2'); brian.__version__ = 'mock'
        brian.prefs = types.SimpleNamespace(codegen=types.SimpleNamespace(target=None))
        modules = {'brian2': brian, 'pandas': types.ModuleType('pandas'),
                   'model': types.ModuleType('model'), 'utils': types.ModuleType('utils')}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory); upstream = root / 'upstream'; upstream.mkdir()
            output = root / 'output'; output.mkdir()
            for name in ['2023_03_23_completeness_630_final.csv', '2023_03_23_connectivity_630_final.parquet']:
                (upstream / name).write_bytes(b'mock-input')
            (output / 'sugarR.parquet').write_bytes(b'old-unbound-data')
            argv = ['fly', '--upstream', str(upstream), '--output-dir', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(fly.subprocess, 'check_output', side_effect=[fly.REVISION, '']):
                with self.assertRaisesRegex(ValueError, 'Unbound nonempty'):
                    fly.main()
            # Simulate an existing matching protocol, but never bless its missing receipt.
            with patch.object(sys, 'argv', argv), patch.object(fly.subprocess, 'check_output', side_effect=[fly.REVISION, '']), patch.object(fly, 'bind_protocol'):
                with self.assertRaisesRegex(ValueError, 'Missing integrity receipt'):
                    fly.main()

    def test_summary_rejects_extra_duplicate_and_nonfinite_trials(self):
        config = e.EvidenceConfig.confirmation()
        rows = records(config)
        for bad in [rows + [dict(rows[0], seed=999)], rows + [rows[0]],
                    [dict(rows[0], accuracy=float('nan')), *rows[1:]]]:
            with self.assertRaises(ValueError):
                e.summarize(bad, config)
        self.assertEqual(e.summarize(rows[:-1], config)['decision'], 'incomplete_or_modified_protocol')
        with self.assertRaises(ValueError):
            e.confidence([1., float('inf')])

    def test_evaluation_rejects_nan_logits_and_activity(self):
        class Broken(torch.nn.Module):
            def __init__(self, logits):
                super().__init__(); self.bad_logits = logits
            def forward(self, x, return_event_rate=False):
                out = torch.full((len(x), 3), float('nan') if self.bad_logits else 0.)
                return out, torch.tensor(0. if self.bad_logits else float('nan'))
        config = e.EvidenceConfig()
        parts, _ = e.synthetic_data(config, 'shd')
        for logits in [True, False]:
            with self.assertRaises(FloatingPointError):
                e.evaluate(Broken(logits), parts[4], parts[5], config, torch.device('cpu'))

    def test_compiler_nan_is_a_numerical_failure_not_zero_error(self):
        class Validation:
            def __len__(self): return 4
            def __getitem__(self, key): return self
            def to(self, device): return torch.zeros(4, 8, 12)
        class Finite(torch.nn.Module):
            def forward(self, x): return torch.tensor([[1., 0.]]).repeat(len(x), 1)
        def nan_output(x): return torch.tensor([[float('nan'), 0.]]).repeat(len(x), 1)
        with patch.object(torch.cuda, 'synchronize'), patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                patch.object(torch, 'compile', return_value=nan_output):
            result = e.benchmark(Finite(), Validation(), e.EvidenceConfig.confirmation(), torch.device('cuda'))
        self.assertEqual(result['status'], 'numerical_failure')
        self.assertNotIn('maximum_logit_difference', result)

    def test_checkpoint_rejects_other_seed_and_altered_bytes(self):
        config = e.EvidenceConfig(epochs=1)
        parts, _ = e.synthetic_data(config, 'shd')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'checkpoint.pt'
            e.seed_everything(310, torch.device('cpu'))
            first = e.make_model('conv1d', config, 12, 3, 1.)
            e.train(first, parts, config, 310, torch.device('cpu'), path)
            e.seed_everything(311, torch.device('cpu'))
            wrong = e.make_model('conv1d', config, 12, 3, 1.)
            with self.assertRaises(ValueError):
                e.train(wrong, parts, config, 311, torch.device('cpu'), path)
            path.write_bytes(b'not a checkpoint')
            with self.assertRaises(ValueError):
                e.train(first, parts, config, 310, torch.device('cpu'), path)

    def test_completed_record_and_checkpoint_cannot_be_resealed_by_resume(self):
        config = e.EvidenceConfig(datasets=('shd',), epochs=1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            e.run(config, root / 'base', root / 'data', 'cpu')
            for kind in ['record', 'checkpoint']:
                output = root / kind
                shutil.copytree(root / 'base', output)
                trial = output / 'shd_310_conv1d'
                if kind == 'record':
                    path = trial / 'record.json'
                    row = json.loads(path.read_text())
                    row['accuracy'], row['initial_parameter_sha256'] = .99999, 'not-a-hash'
                    path.write_text(json.dumps(row))
                else:
                    (trial / 'checkpoint.pt').write_bytes(b'invalid')
                with self.assertRaises(ValueError):
                    e.run(config, output, root / 'data', 'cpu')

    def test_changed_cache_rejected_before_new_output_can_adopt_it(self):
        config = e.EvidenceConfig.confirmation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'shd'; root.mkdir()
            for split in ['train', 'test']:
                with h5py.File(root / f'shd_{split}.h5', 'w') as f:
                    f['labels'] = np.array([0, 0, 1, 1])
                    group = f.create_group('spikes')
                    times = group.create_dataset('times', (4,), dtype=h5py.vlen_dtype(np.dtype('float64')))
                    units = group.create_dataset('units', (4,), dtype=h5py.vlen_dtype(np.dtype('int64')))
                    for i in range(4): times[i], units[i] = [.01], [i]
            e.official_data(config, 'shd', Path(directory))
            e.official_data(config, 'shd', Path(directory))
            cache = root / 'shd_train_t64_c700_d1400ms.pt'
            row = torch.load(cache, weights_only=True); row['events'].zero_(); torch.save(row, cache)
            with self.assertRaises(ValueError):
                e.official_data(config, 'shd', Path(directory))

    def test_unbound_reference_output_and_modified_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'existing.parquet').write_bytes(b'old output')
            with self.assertRaises(ValueError):
                bind_protocol(root / 'protocol.json', {'schema': 2})
            seal_files(root, ['existing.parquet'], 'receipt.json', {'run': 'a'})
            verify_files(root, ['existing.parquet'], 'receipt.json', {'run': 'a'})
            with self.assertRaises(ValueError):
                verify_files(root, ['existing.parquet'], 'receipt.json', {'run': 'b'})
            (root / 'existing.parquet').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_files(root, ['existing.parquet'], 'receipt.json', {'run': 'a'})

    def test_clean_notebook_colab_detection_handles_missing_parent(self):
        import ast
        path = Path(__file__).resolve().parents[1] / 'tools/build_research_notebook.py'
        text = path.read_text()
        self.assertIn('except (ImportError, ModuleNotFoundError, ValueError)', text)
        notebook = json.loads((path.parents[1] / 'notebooks/evidence1_research_restart.ipynb').read_text())
        cell = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'def module_available' in ''.join(c['source']))
        tree = ast.parse(cell)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'module_available')
        import importlib.util
        scope = {'importlib': __import__('importlib')}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<colab detection>', 'exec'), scope)
        with patch.object(importlib.util, 'find_spec', side_effect=ModuleNotFoundError):
            self.assertFalse(scope['module_available']('google.colab'))


if __name__ == '__main__': unittest.main()
