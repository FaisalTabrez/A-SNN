from __future__ import annotations
import contextlib
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ammc_gen5.evidence1_audio import (
    ARMS, CausalResidual, EvidenceConfig, bundle, confidence, derangement,
    make_model, run, summarize, synthetic_data, train,
)
from ammc_gen5.runtime import seed_everything
import torch


class Evidence1Contract(unittest.TestCase):
    def test_confirmation_capacity(self):
        config = EvidenceConfig.confirmation()
        for classes, duration in [(20, 1.4), (35, 1.0)]:
            for arm in ARMS:
                model = make_model(arm, config, 700, classes, duration)
                count = sum(p.numel() for p in model.parameters())
                self.assertLessEqual(count, config.target_parameters)
                self.assertGreaterEqual(count, 0.95 * config.target_parameters)

    def test_derangement_and_singleton_tail(self):
        for n in [2, 3, 19, 257]:
            order = derangement(n, 310)
            self.assertEqual(sorted(order.tolist()), list(range(n)))
            self.assertTrue(bool((order != torch.arange(n)).all()))
            self.assertTrue(torch.equal(order, derangement(n, 310)))
        with self.assertRaises(ValueError):
            derangement(1, 310)

    def test_frozen_ablation_identity_and_no_mutation(self):
        config = EvidenceConfig()
        model = make_model('residual_lif', config, 12, 3, 1.0)
        from ammc_gen5.shd_state_placement_diagnostic import ResidualTemporalConvStateClassifier
        events = synthetic_data(config, 'shd')[0][0][:4]
        original = {k: v.clone() for k, v in model.state_dict().items()}
        self.assertTrue(torch.equal(model(events), ResidualTemporalConvStateClassifier.forward(model, events)))
        zero = model(events, intervention='zero_state')
        model.set_ablation_mode('direct_only')
        self.assertTrue(torch.equal(zero, ResidualTemporalConvStateClassifier.forward(model, events)))
        model.set_ablation_mode('full')
        self.assertTrue(torch.equal(model(events), model(events, intervention='shuffled_state', partner_events=events)))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_epoch_resume_matches_uninterrupted(self):
        config = EvidenceConfig(epochs=3)
        device = torch.device('cpu')
        parts = synthetic_data(config, 'shd')[0]
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            seed_everything(310, device)
            full = make_model('residual_lif', config, 12, 3, 1.0)
            full_info = train(full, parts, config, 310, device, Path(temporary) / 'full.pt')
            seed_everything(310, device)
            resumed = make_model('residual_lif', config, 12, 3, 1.0)
            train(resumed, parts, replace(config, epochs=1), 310, device, Path(temporary) / 'resume.pt')
            seed_everything(310, device)
            resumed = make_model('residual_lif', config, 12, 3, 1.0)
            info = train(resumed, parts, config, 310, device, Path(temporary) / 'resume.pt')
            self.assertEqual(full_info['best_epoch'], info['best_epoch'])
            self.assertEqual(full_info['initial_parameter_sha256'], info['initial_parameter_sha256'])
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]))

    @staticmethod
    def records(config, causal=True, accuracy=True, systems=True):
        return [{"dataset": d, "seed": s, "arm": a,
                 "accuracy": 0.8 if a == 'residual_lif' or accuracy else 0.95,
                 "zero_state_accuracy": 0.6 if causal else 0.8,
                 "shuffled_state_accuracy": 0.6 if causal else 0.8,
                 "systems": {"status": 'measured' if systems else 'not_measured', 'speedup': 2.,
                             'prediction_agreement': 1., 'maximum_logit_difference': 0.}}
                for d in config.datasets for s in config.seeds for a in ARMS]

    def test_decisions_cannot_pass_smoke_or_modified_protocol(self):
        config = EvidenceConfig.confirmation()
        records = self.records(config)
        self.assertEqual(summarize(records, config)['decision'], 'model_and_systems_gates_pass_replication_pending')
        self.assertEqual(summarize(self.records(config, causal=False), config)['decision'], 'stop_causal_claim')
        self.assertEqual(summarize(self.records(config, accuracy=False), config)['decision'], 'causal_audit_only')
        self.assertEqual(summarize(self.records(config, systems=False), config)['decision'], 'model_gates_pass_systems_unconfirmed')
        self.assertEqual(summarize(records[:-1], config)['decision'], 'incomplete_or_modified_protocol')
        self.assertEqual(summarize(records, replace(config, epochs=14))['decision'], 'incomplete_or_modified_protocol')
        smoke = EvidenceConfig()
        self.assertEqual(summarize(self.records(smoke), smoke)['decision'], 'smoke_only')

    def test_confidence_not_fake_with_one_seed(self):
        self.assertIsNone(confidence([0.1])['ci95_lower'])
        estimate = confidence([0.1, 0.2, 0.3])
        self.assertAlmostEqual(estimate['mean'], 0.2)
        self.assertLess(estimate['ci95_lower'], estimate['mean'])
        self.assertGreater(estimate['ci95_upper'], estimate['mean'])

    def test_bundle_does_not_include_its_temporary_or_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'record.json').write_text('{"test":true}')
            (output / 'partial.tmp').write_text('incomplete')
            archive = bundle(output)
            bundle(output)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(set(zipped.namelist()), {'record.json', 'manifest.json'})
                manifest = json.loads(zipped.read('manifest.json'))['files']
                self.assertEqual(hashlib.sha256(zipped.read('record.json')).hexdigest(), manifest['record.json']['sha256'])

    def test_complete_smoke_resume_and_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / 'out'
            config = EvidenceConfig(datasets=('shd',), epochs=1)
            first = run(config, output, Path(temporary) / 'data', 'cpu')
            again = run(config, output, Path(temporary) / 'data', 'cpu')
            self.assertEqual(first, again)
            self.assertEqual(len(first['records']), 4)
            self.assertEqual(first['summary']['decision'], 'smoke_only')
            with self.assertRaisesRegex(ValueError, 'Protocol/source/environment changed'):
                run(replace(config, epochs=2), output, Path(temporary) / 'data', 'cpu')

    def test_fits_adapter_does_not_test_during_selection(self):
        spec = importlib.util.spec_from_file_location('fits_adapter_test', ROOT / 'tools/fits_validation_reference.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        calls = []
        module = types.ModuleType('common.training')
        def create_optimizer(model, cfg):
            optimizer = torch.optim.Adam(model.parameters())
            return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 1), None
        def run_epoch(model, loader, criterion, device, optimizer=None, prep=None):
            calls.append(loader)
            if optimizer is not None:
                optimizer.step()
            return 1.0, 0.5
        module.create_optimizer, module.run_epoch = create_optimizer, run_epoch
        cfg = types.SimpleNamespace(epochs=3)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'common.training': module}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                adapter.validation_fit(torch.nn.Linear(2, 2), cfg, temporary, 'cpu', 'train', 'test')
            summary = adapter.validation_fit(torch.nn.Linear(2, 2), cfg, temporary, 'cpu', 'train', 'test', 'valid')
            self.assertEqual(calls, ['train', 'valid'] * 3 + ['test'])
            self.assertEqual(summary['best_epoch'], 1)
            adapter.validation_fit(torch.nn.Linear(2, 2), cfg, temporary, 'cpu', 'train', 'test', 'valid')
            self.assertEqual(calls.count('test'), 1)

    def test_notebook_snapshots_and_syntax(self):
        import ast, base64
        path = ROOT / 'notebooks/evidence1_research_restart.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                tree = ast.parse(source)
                if source.startswith('SNAPSHOTS = '):
                    snapshots = ast.literal_eval(tree.body[0].value)
                    for relative, record in snapshots.items():
                        content = base64.b64decode(record['base64'])
                        self.assertEqual(hashlib.sha256(content).hexdigest(), record['sha256'])
                        self.assertEqual(content, (ROOT.parent / relative).read_bytes())
        self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
