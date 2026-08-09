from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.ssc_efficiency_baselines import (
    SSC_EFFICIENCY_BASELINE_ARMS,
    available_ssc_efficiency_baseline_arms,
    matched_temporal_tcn_channels,
    temporal_tcn_parameter_count,
)


class SSCEfficiencyBaselinesContractTest(unittest.TestCase):
    def test_registered_baseline_matrix(self) -> None:
        self.assertEqual(len(SSC_EFFICIENCY_BASELINE_ARMS), 3)
        self.assertEqual(
            available_ssc_efficiency_baseline_arms(),
            ("temporal_conv1d", "dilated_tcn", "residual_lif"),
        )

    def test_tcn_channels_are_parameter_matched(self) -> None:
        channels, actual = matched_temporal_tcn_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=(1, 2, 4, 8),
        )
        self.assertEqual(
            actual,
            temporal_tcn_parameter_count(
                700,
                channels,
                35,
                input_kernel_size=5,
                hidden_kernel_size=3,
                temporal_levels=(1, 2, 4, 8),
            ),
        )
        self.assertLessEqual(actual, 133631)
        self.assertGreater(actual / 133631, 0.95)


if __name__ == "__main__":
    unittest.main()
