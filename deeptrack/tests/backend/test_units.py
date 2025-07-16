# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack.backend import units
from deeptrack import units_registry as u
from deeptrack import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestUnits(unittest.TestCase):


    def test_get_active_voxel_size(self):
        voxel = units.get_active_voxel_size()
        self.assertIsInstance(voxel, tuple)
        self.assertEqual(len(voxel), 3)
        for dim in voxel:
            self.assertIsInstance(dim, float)


    def test_get_active_scale(self):
        scale = units.get_active_scale()
        self.assertIsInstance(scale, tuple)
        self.assertEqual(len(scale), 3)
        for factor in scale:
            self.assertIsInstance(factor, float)


    def test_create_context_conversion(self):
        ctx = units.create_context(
            xpixel=2e-6, ypixel=1e-6, zpixel=1e-6,
            xscale=2, yscale=1, zscale=1
        )
        with u.context(ctx):
            self.assertAlmostEqual(
                (1 * u.simulation_xpixel).to("meter").magnitude,
                1e-6,
            )
            self.assertAlmostEqual(
                (1 * u.simulation_ypixel).to("meter").magnitude,
                1e-6,
            )


    def test_conversion_table_with_scalars(self):
        converter = units.ConversionTable(
            length=(u.meter, u.micrometer),
            time=(u.second, u.millisecond)
        )
        result = converter.convert(length=1.2, time=0.5)
        self.assertAlmostEqual(result["length"].magnitude, 1.2e6)
        self.assertEqual(str(result["length"].units), "micrometer")
        self.assertAlmostEqual(result["time"].magnitude, 500.0)
        self.assertEqual(str(result["time"].units), "millisecond")


    def test_conversion_table_with_numpy_array(self):
        converter = units.ConversionTable(length=(u.meter, u.micrometer))
        arr = np.array([1.0, 2.0])
        result = converter.convert(length=arr)
        np.testing.assert_allclose(result["length"].magnitude, [1e6, 2e6])
        self.assertEqual(str(result["length"].units), "micrometer")


    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_conversion_table_with_torch_tensor(self):
        converter = units.ConversionTable(length=(u.meter, u.micrometer))
        tensor = torch.tensor([1.0, 2.0])
        result = converter.convert(length=tensor)
        np.testing.assert_allclose(result["length"].magnitude, [1e6, 2e6])
        self.assertEqual(str(result["length"].units), "micrometer")


if __name__ == "__main__":
    unittest.main()
