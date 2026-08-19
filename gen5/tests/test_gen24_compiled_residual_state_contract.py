from __future__ import annotations

import unittest

from ammc_gen5 import (
    Gen24Config,
    available_gen24_models,
    decide_gen24,
    summarize_gen24,
)


class Gen24ContractTest(unittest.TestCase):
    def test_models_are_frozen(self):
        self.assertEqual(available_gen24_models(), ("matched_tcn", "residual_lif"))

    def test_summary_and_decision_pass_exact_fast_compilation(self):
        config = Gen24Config(
            seeds=(1, 2, 3),
            batch_sizes=(256,),
            warmup_iterations=1,
            measurement_iterations=1,
            measurement_repeats=1,
        )
        records = []
        for seed in config.seeds:
            for model, eager_rate, compiled_rate in (
                ("matched_tcn", 1200.0, 1800.0),
                ("residual_lif", 500.0, 1000.0),
            ):
                for runtime, rate in (("eager", eager_rate), ("compiled", compiled_rate)):
                    records.append({
                        "seed": seed,
                        "model": model,
                        "runtime": runtime,
                        "batch_size": 256,
                        "seconds": 1.0,
                        "examples_per_second": rate,
                        "milliseconds_per_batch": 1.0,
                        "compile_seconds": 1.0 if runtime == "compiled" else 0.0,
                        "compile_active": runtime == "compiled",
                        "compile_error": None,
                        "maximum_logit_difference": 1e-6 if runtime == "compiled" else 0.0,
                        "prediction_agreement": 1.0,
                        "cuda_peak_memory_mb": 10.0,
                        "speedup_vs_eager": rate / eager_rate,
                    })
        summary = summarize_gen24(records)
        self.assertEqual(len(summary), 4)
        decision = decide_gen24(records, config)
        self.assertEqual(decision["status"], "pass")
        self.assertTrue(decision["compiled_residual_state_supported"])
        self.assertFalse(decision["software_throughput_parity_vs_tcn_supported"])
        self.assertFalse(decision["hardware_energy_claim_authorized"])

    def test_equivalence_failure_stops(self):
        config = Gen24Config(seeds=(1, 2, 3), batch_sizes=(256,))
        rows = [{
            "seed": seed,
            "model": "residual_lif",
            "runtime": "compiled",
            "batch_size": 256,
            "compile_active": True,
            "maximum_logit_difference": 0.1,
            "prediction_agreement": 0.9,
            "speedup_vs_eager": 2.0,
            "examples_per_second": 1000.0,
        } for seed in config.seeds]
        self.assertEqual(decide_gen24(rows, config)["status"], "stop")


if __name__ == "__main__":
    unittest.main()
