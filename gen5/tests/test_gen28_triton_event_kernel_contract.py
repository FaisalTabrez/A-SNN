from __future__ import annotations
import unittest
from ammc_gen5 import Gen28Config,build_event_list,decide_gen28
from ammc_gen5.event_mnist import torch

class Gen28ContractTest(unittest.TestCase):
    def test_event_list_roundtrip(self):
        if torch is None:self.skipTest("PyTorch unavailable")
        events=torch.zeros((2,3,4));events[0,1,2]=1;events[1,2,3]=2
        b,t,i,v=build_event_list(events);rebuilt=torch.zeros_like(events);rebuilt[b.long(),t.long(),i.long()]=v
        self.assertTrue(torch.equal(events,rebuilt))
    def test_decision_requires_real_native_speed(self):
        config=Gen28Config(seeds=(1,2,3),batch_sizes=(256,),density_batch_size=256)
        rows=[]
        for seed in config.seeds:
            rows.append({"seed":seed,"runtime":"triton_event_native","workload":"real_ssc","batch_size":256,
                "maximum_current_difference_vs_coo":1e-4,"prediction_agreement_vs_dense":1.0,"speed_ratio_vs_dense":1.2})
            rows.append({"seed":seed,"runtime":"triton_end_to_end","workload":"real_ssc","batch_size":256,"speed_ratio_vs_dense":0.8})
        decision=decide_gen28(rows,config);self.assertEqual(decision["status"],"pass");self.assertTrue(decision["sensor_native_kernel_supported"]);self.assertFalse(decision["dense_cache_kernel_supported"])
    def test_slow_kernel_closes_path(self):
        config=Gen28Config(seeds=(1,2,3),batch_sizes=(256,),density_batch_size=256)
        rows=[{"seed":seed,"runtime":"triton_event_native","workload":"real_ssc","batch_size":256,
            "maximum_current_difference_vs_coo":1e-4,"prediction_agreement_vs_dense":1.0,"speed_ratio_vs_dense":0.5} for seed in config.seeds]
        self.assertEqual(decide_gen28(rows,config)["next_milestone"],"close_event_sparse_software_path")
if __name__=="__main__":unittest.main()
