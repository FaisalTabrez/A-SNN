"""Closure tooling must not promote software checks to scientific approval."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import tempfile
import unittest
import zipfile
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ammc_gen5.evidence1_closure import safe_members, verify_return_bundle, estimate, recompute_results
TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class ClosureTests(unittest.TestCase):
    def test_archive_rejects_traversal_duplicates_and_unpinned_data(self):
        for names in [['../escape'], ['x', 'x'], ['C:/escape']]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'bad.zip'
                import warnings
                with warnings.catch_warnings(), zipfile.ZipFile(path, 'w') as archive:
                    warnings.simplefilter('ignore', UserWarning)
                    for name in names: archive.writestr(name, 'test')
                with zipfile.ZipFile(path) as archive, self.assertRaises(ValueError): safe_members(archive)
                with self.assertRaises(ValueError): verify_return_bundle(path)

    def test_raw_zip_backslash_is_rejected_even_when_windows_normalizes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.zip'
            with zipfile.ZipFile(path, 'w') as archive: archive.writestr('a/b', 'test')
            path.write_bytes(path.read_bytes().replace(b'a/b', bytes([97, 92, 98])))
            with zipfile.ZipFile(path) as archive, self.assertRaises(ValueError): safe_members(archive)

    def test_nonfinite_fidelity_is_never_zero_error(self):
        diagnostic = load('evidence1_fidelity_diagnostic')
        finite = torch.tensor([[1., 2.], [3., 4.]])
        row = diagnostic.compare_logits(finite, finite + .01)
        self.assertEqual(row['samples'], 2)
        self.assertGreater(row['maximum_logit_difference'], .009)
        with self.assertRaises(FloatingPointError): diagnostic.compare_logits(finite, finite * math.nan)
        with self.assertRaises(ValueError): diagnostic.compare_logits(finite, finite[:1])

    def test_original_source_package_matches_published_protocol(self):
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / 'gen5/docs/evidence1_release/original_protocol.json').read_text())
        diagnostic = load('evidence1_fidelity_diagnostic')
        sources = diagnostic.verify_source_archive(root / 'gen5/repro/evidence1_original_sources.zip', protocol)
        self.assertEqual(len(sources), 73)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad_sources.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for name, content in sources.items(): archive.writestr(name, content + b'\n')
                archive.writestr('source_receipt.json', json.dumps({'protocol_sha256': protocol['sha256']}))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                diagnostic.verify_source_archive(path, protocol)

    def test_published_results_recompute_without_approval(self):
        root = Path(__file__).resolve().parents[1] / 'docs/evidence1_release'
        for name in ['primary_results.json', 'repeat_results.json']:
            results = json.loads((root / name).read_text())
            computed = recompute_results(results)
            self.assertEqual(computed['decision'], 'causal_audit_only')
            self.assertFalse(computed['execution_independence_certified'])
            self.assertFalse(computed['human_reviewed_lock'])
            self.assertLess(computed['maximum_statistic_error'], 1e-10)
            results['summary']['datasets']['shd']['causal_gate'] = False
            with self.assertRaises(ValueError): recompute_results(results)

    def test_recovery_inspection_does_not_certify_a_filename(self):
        recovery = load('audit_historical_recovery')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gen19_bundle.zip'
            with zipfile.ZipFile(path, 'w') as archive: archive.writestr('fake.json', '{}')
            result = recovery.inspect(path)
            self.assertFalse(result['recovery_complete'])
            self.assertFalse(result['metrics_recomputed'])

    def test_interval_rejects_nonfinite_or_single_observation(self):
        for values in [[1.], [1., float('nan')], [1., float('inf')]]:
            with self.assertRaises(ValueError): estimate(values)


if __name__ == '__main__': unittest.main()
