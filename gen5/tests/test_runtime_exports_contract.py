from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RuntimeExportsContractTest(unittest.TestCase):
    def test_benchmark_runtime_helpers_are_exported_from_package_root(self) -> None:
        from ammc_gen5 import memory_namespace, resolve_device, sync

        self.assertTrue(callable(memory_namespace))
        self.assertTrue(callable(resolve_device))
        self.assertTrue(callable(sync))

    def test_world_preset_helpers_are_exported_from_package_root(self) -> None:
        from ammc_gen5 import available_world_presets, world_preset_config, world_preset_names

        self.assertTrue(callable(available_world_presets))
        self.assertTrue(callable(world_preset_config))
        self.assertIn("simple", world_preset_names())


if __name__ == "__main__":
    unittest.main()
