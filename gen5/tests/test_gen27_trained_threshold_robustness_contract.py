from __future__ import annotations
import unittest
from ammc_gen5 import Gen27Config, ResidualLIFStateHead, decide_gen27, head_with_trace
from ammc_gen5.event_mnist import torch
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.shd_state_placement_diagnostic import ResidualTemporalConvStateClassifier

class Gen27ContractTest(unittest.TestCase):
    def test_diagnostic_head_matches_source_model(self):
        if torch is None:
            self.skipTest("PyTorch unavailable")
        config=SHDConfig(input_neurons=4,classes=3,timesteps=5,hidden_neurons=3,max_edges=8,epochs=1,warmup_epochs=0)
        model=ResidualTemporalConvStateClassifier(config,channels=3,kernel_size=3,temporal_levels=(1,),dynamics="lif",surrogate_slope=10.0).eval()
        events=(torch.rand((2,5,4))>0.7).to(torch.float32)
        currents=model.temporal(events.transpose(1,2)).transpose(1,2)
        expected=model(events);actual=head_with_trace(ResidualLIFStateHead(model),currents,1e-3)[0]
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-5))

    def test_stable_trained_behavior_passes(self):
        config=Gen27Config(seeds=(1,2,3))
        rows=[]
        for seed in config.seeds:
            for arm in ("dense_reference","sparse_operator","shuffled_error_control"):
                rows.append({"seed":seed,"arm":arm,"accuracy_change_vs_dense":0.0005 if arm=="sparse_operator" else 0.0,
                    "prediction_agreement_vs_dense":0.9995,"spike_disagreement_rate":0.00005})
        decision=decide_gen27(rows,config);self.assertEqual(decision["status"],"pass");self.assertFalse(decision["hardware_energy_claim_authorized"])

    def test_spike_instability_stops(self):
        config=Gen27Config(seeds=(1,2,3))
        rows=[{"seed":seed,"arm":"sparse_operator","accuracy_change_vs_dense":0.0,
            "prediction_agreement_vs_dense":1.0,"spike_disagreement_rate":0.01} for seed in config.seeds]
        decision=decide_gen27(rows,config);self.assertEqual(decision["status"],"stop");self.assertEqual(decision["next_milestone"],"threshold_margin_training")

if __name__=="__main__":unittest.main()
