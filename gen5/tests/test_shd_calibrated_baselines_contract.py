from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_calibrated_baselines import SHD_CALIBRATED_BASELINE_ARMS, available_shd_calibrated_baseline_arms, matched_temporal_conv_channels, temporal_conv_parameter_count


class SHDCalibratedBaselinesContractTest(unittest.TestCase):
    def test_registered_baselines(self) -> None:
        self.assertEqual(len(SHD_CALIBRATED_BASELINE_ARMS), 4)
        self.assertEqual(set(available_shd_calibrated_baseline_arms()), {"raw_temporal_pyramid", "temporal_conv1d", "gru_temporal", "dense_lif_recurrent"})

    def test_temporal_conv_is_parameter_matched(self) -> None:
        channels, actual = matched_temporal_conv_channels(700, 20, 133631, kernel_size=5, temporal_levels=(1, 2, 4, 8))
        self.assertEqual(actual, temporal_conv_parameter_count(700, channels, 20, kernel_size=5, temporal_levels=(1, 2, 4, 8)))
        self.assertLessEqual(actual, 133631)
        self.assertGreater(actual / 133631, 0.95)


if __name__ == "__main__":
    unittest.main()
