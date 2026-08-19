from __future__ import annotations

import unittest

from ammc_gen5 import (
    DenseResidualPipeline,
    Gen25Config,
    SparseHybridPipeline,
    decide_gen25,
    sparse_temporal_currents,
)
from ammc_gen5.event_mnist import nn, torch
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.shd_state_placement_diagnostic import ResidualTemporalConvStateClassifier
from ammc_gen5.gen25_event_driven_sparse_audit import (
    ResidualLIFStateHead,
    _optional_sparse_pipeline,
)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class Gen25ContractTest(unittest.TestCase):
    def test_compiled_module_is_not_truth_tested(self):
        class CompiledLike:
            def __len__(self):
                raise TypeError("compiled modules must not be truth-tested")

        compiled = CompiledLike()
        pipeline = _optional_sparse_pipeline(object(), compiled)
        self.assertIs(pipeline.compiled_head, compiled)
        self.assertIsNone(_optional_sparse_pipeline(object(), None))

    def test_sparse_operator_matches_dense_conv1d(self):
        torch.manual_seed(1)
        temporal = nn.Conv1d(4, 3, 3, padding=1)
        events = (torch.rand((2, 5, 4)) > 0.7).to(torch.float32)
        dense = temporal(events.transpose(1, 2)).transpose(1, 2)
        sparse = sparse_temporal_currents(events, temporal)
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-5))

    def test_sparse_and_dense_pipelines_share_exact_state_head(self):
        config = SHDConfig(
            input_neurons=4, classes=3, timesteps=5, hidden_neurons=3,
            max_edges=8, epochs=1, warmup_epochs=0,
        )
        source = ResidualTemporalConvStateClassifier(
            config, channels=3, kernel_size=3, temporal_levels=(1,),
            dynamics="lif", surrogate_slope=10.0,
        ).eval()
        events = (torch.rand((2, 5, 4)) > 0.7).to(torch.float32)
        dense = DenseResidualPipeline(source).eval()(events)
        sparse = SparseHybridPipeline(source.temporal, ResidualLIFStateHead(source).eval())(events)
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-5))

    def test_decision_requires_real_speed_and_equivalence(self):
        config = Gen25Config(seeds=(1, 2, 3), batch_sizes=(32, 256), density_batch_size=32)
        records = []
        for seed in config.seeds:
            records.extend([
                {
                    "seed": seed, "runtime": "event_sparse_hybrid", "workload": "real_ssc",
                    "batch_size": 256, "compile_active": True, "maximum_logit_difference": 1e-6,
                    "prediction_agreement": 1.0, "speed_ratio_vs_dense": 1.2,
                },
                {
                    "seed": seed, "runtime": "event_sparse_hybrid", "workload": "synthetic_0.0050",
                    "batch_size": 32, "event_density": 0.005, "compile_active": True,
                    "maximum_logit_difference": 1e-6, "prediction_agreement": 1.0,
                    "speed_ratio_vs_dense": 1.1,
                },
            ])
        decision = decide_gen25(records, config)
        self.assertEqual(decision["status"], "pass")
        self.assertTrue(decision["low_density_crossover_supported"])
        self.assertFalse(decision["hardware_energy_claim_authorized"])

    def test_slow_sparse_operator_stops(self):
        config = Gen25Config(seeds=(1, 2, 3), batch_sizes=(256,), density_batch_size=256)
        rows = [{
            "seed": seed, "runtime": "event_sparse_hybrid", "workload": "real_ssc",
            "batch_size": 256, "compile_active": True, "maximum_logit_difference": 1e-6,
            "prediction_agreement": 1.0, "speed_ratio_vs_dense": 0.5,
        } for seed in config.seeds]
        self.assertEqual(decide_gen25(rows, config)["status"], "stop")


if __name__ == "__main__":
    unittest.main()
