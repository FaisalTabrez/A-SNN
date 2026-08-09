from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_spiking_temporal_conv import SHD_SPIKING_TEMPORAL_CONV_ARMS, available_shd_spiking_temporal_conv_arms, matched_temporal_conv_state_channels, temporal_conv_state_parameter_count


class SHDSpikingTemporalConvContractTest(unittest.TestCase):
    def test_registered_redesign_matrix(self) -> None:
        self.assertEqual(len(SHD_SPIKING_TEMPORAL_CONV_ARMS), 5)
        self.assertIn("temporal_conv_leaky_lif", available_shd_spiking_temporal_conv_arms())
        self.assertIn("temporal_conv_leaky_analog", available_shd_spiking_temporal_conv_arms())

    def test_state_channels_are_parameter_matched(self) -> None:
        channels, actual = matched_temporal_conv_state_channels(700, 20, 133631, kernel_size=5, temporal_levels=(1, 2, 4, 8))
        self.assertEqual(actual, temporal_conv_state_parameter_count(700, channels, 20, kernel_size=5, temporal_levels=(1, 2, 4, 8), spiking=True))
        self.assertLessEqual(actual, 133631)
        self.assertGreater(actual / 133631, 0.95)


if __name__ == "__main__":
    unittest.main()
