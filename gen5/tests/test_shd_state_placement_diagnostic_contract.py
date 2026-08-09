from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_state_placement_diagnostic import (
    SHD_STATE_PLACEMENT_ARMS,
    available_shd_state_placement_arms,
    matched_temporal_conv_residual_channels,
    temporal_conv_residual_parameter_count,
)


class SHDStatePlacementDiagnosticContractTest(unittest.TestCase):
    def test_diagnostic_matrix_contains_state_and_residual_pairs(self) -> None:
        self.assertEqual(len(SHD_STATE_PLACEMENT_ARMS), 5)
        names = available_shd_state_placement_arms()
        self.assertIn("leaky_analog_state_only", names)
        self.assertIn("leaky_analog_residual", names)
        self.assertIn("leaky_lif_state_only", names)
        self.assertIn("leaky_lif_residual", names)

    def test_residual_channels_are_parameter_matched(self) -> None:
        channels, actual = matched_temporal_conv_residual_channels(
            700, 20, 133631, kernel_size=5, temporal_levels=(1, 2, 4, 8)
        )
        self.assertEqual(
            actual,
            temporal_conv_residual_parameter_count(
                700,
                channels,
                20,
                kernel_size=5,
                temporal_levels=(1, 2, 4, 8),
                spiking=True,
            ),
        )
        self.assertLessEqual(actual, 133631)
        self.assertGreater(actual / 133631, 0.95)


if __name__ == "__main__":
    unittest.main()
