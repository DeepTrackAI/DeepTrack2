# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path

import unittest

import numpy as np

from deeptrack.optics import Fluorescence, Brightfield
from deeptrack import scatterers

from deeptrack.backend import TORCH_AVAILABLE, xp
from deeptrack.tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestScatterers_NumPy(BackendTestBase):
    BACKEND = "numpy"

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        elif self.BACKEND == "torch":
            return torch.Tensor
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")
        
    def test__all__(self):
        from deeptrack import (
            PointParticle,
            Ellipse,
            Sphere,
            Ellipsoid,
            MieSphere,
            MieStratifiedSphere,
            Incoherent,
        )

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def test_PointParticle(self):

        # --- Basic properties ---
        scatterer = scatterers.PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
        )
        output_scatterer = scatterer.resolve()
        self.assertIsInstance(output_scatterer.array, self.array_type)
        self.assertEqual(output_scatterer.shape, (1, 1, 1))
        self.assertTrue(
            np.allclose(
                np.asarray(output_scatterer.properties["position"]),
                np.array([32, 32]),
            )
        )
        self.assertEqual(output_scatterer.properties["intensity"], 100)

    def test_Ellipse(self):

        e1 = scatterers.Ellipse(
            radius=(3e-6, 2e-6),
            rotation=0.0,
            position=(16, 16),
            position_unit="pixel",
            upsample=1,
        )

        e3 = scatterers.Ellipse(
            radius=(3e-6, 2e-6),
            rotation=0.0,
            position=(16, 16),
            position_unit="pixel",
            upsample=3,
        )

        v1 = e1.resolve()
        v3 = e3.resolve()
        
        self.assertIsInstance(v1.array, self.array_type)
        self.assertIsInstance(v3.array, self.array_type)
        self.assertEqual(v1.shape, (3, 5, 1))
        self.assertEqual(v3.shape, (5, 6, 1))
        self.assertTrue(
            np.allclose(
                np.asarray(v1.properties["position"]),
                np.array([16, 16]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(v3.properties["position"]),
                np.array([16, 16]),
            )
        )

        a1 = self.to_numpy(v1.array)
        a3 = self.to_numpy(v3.array)

        self.assertGreater(a1.sum(), 0)
        self.assertGreater(a3.sum(), 0)
        self.assertGreaterEqual(a3.sum(), a1.sum())

        self.assertTrue(np.allclose(a1, np.flip(a1, axis=0)))
        self.assertTrue(np.allclose(a1, np.flip(a1, axis=1)))


    def test_Sphere(self):
        s1 = scatterers.Sphere(
            radius=1.5e-6,
            position=(16, 16),
            position_unit="pixel",
            upsample=1,
        )

        s3 = scatterers.Sphere(
            radius=1.5e-6,
            position=(16, 16),
            position_unit="pixel",
            upsample=3,
        )

        v1 = s1.resolve()
        v3 = s3.resolve()

        self.assertIsInstance(v1.array, self.array_type)
        self.assertIsInstance(v3.array, self.array_type)

        self.assertEqual(v1.shape, (3, 3, 3))
        self.assertEqual(v3.shape, (3, 3, 3))

        self.assertTrue(np.allclose(np.asarray(v1.properties["position"]), np.array([16, 16])))
        self.assertTrue(np.allclose(np.asarray(v3.properties["position"]), np.array([16, 16])))

        a1 = self.to_numpy(v1.array)
        a3 = self.to_numpy(v3.array)

        self.assertGreater(a1.sum(), 0)
        self.assertGreater(a3.sum(), 0)
        self.assertTrue(np.any((a3 > 0) & (a3 <=1.0)))

        self.assertTrue(np.allclose(a1, np.flip(a1, axis=0)))
        self.assertTrue(np.allclose(a1, np.flip(a1, axis=1)))
        self.assertTrue(np.allclose(a1, np.flip(a1, axis=2)))

    def test_Ellipsoid(self):
        e1 = scatterers.Ellipsoid(
            radius=(3e-6, 2e-6, 1e-6),
            rotation=(0.0, 0.0, 0.0),
            position=(16, 16),
            position_unit="pixel",
            upsample=1,
        )

        e3 = scatterers.Ellipsoid(
            radius=(3e-6, 2e-6, 1e-6),
            rotation=(0.0, 0.0, 0.0),
            position=(16, 16),
            position_unit="pixel",
            upsample=3,
        )

        v1 = e1.resolve()
        v3 = e3.resolve()

        self.assertIsInstance(v1.array, self.array_type)
        self.assertIsInstance(v3.array, self.array_type)

        self.assertEqual(v1.shape, (5, 6, 3))
        self.assertEqual(v3.shape, (5, 6, 3))

        self.assertTrue(np.allclose(np.asarray(v1.properties["position"]), np.array([16, 16])))
        self.assertTrue(np.allclose(np.asarray(v3.properties["position"]), np.array([16, 16])))

        a1 = self.to_numpy(v1.array)
        a3 = self.to_numpy(v3.array)

        self.assertGreater(a1.sum(), 0)
        self.assertGreater(a3.sum(), 0)
        self.assertGreaterEqual(a3.sum(), a1.sum())

        self.assertTrue(np.allclose(a1, np.flip(a1, axis=0)))
        self.assertTrue(np.allclose(a1, np.flip(a1, axis=2)))


    # def test_MieStratifiedSphere(self):
    #     optics_1 = Brightfield(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=1,
    #         output_region=(0, 0, 64, 128),
    #         padding=(10, 10, 10, 10),
    #         return_field=True,
    #         upscale=4,
    #     )

    #     scatterer = scatterers.MieStratifiedSphere(
    #         radius=np.array([0.5e-6, 1.5e-6]),
    #         refractive_index=[1.45 + 0.1j, 1.52],
    #         aperature_angle=0.1,
    #     )
    #     imaged_scatterer_1 = optics_1(scatterer)
    #     imaged_scatterer_1.update().resolve()

    #     scatterer = scatterers.MieStratifiedSphere(
    #         radius=[0.5e-6, 1.5e-6, 3e-6],
    #         refractive_index=[1.45 + 0.1j, 1.52, 1.23],
    #         aperature_angle=0.1,
    #     )
    #     imaged_scatterer_1 = optics_1(scatterer)
    #     imaged_scatterer_1.update().resolve()

class TestScatterers_NumPy_Only(BackendTestBase):
    BACKEND = "numpy"

    def test_MieSphere(self):
        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45,
            position=(16, 16),
            position_unit="pixel",
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            padding=(0, 0, 0, 0),
            input_polarization=0.0,
            output_polarization=0.0,
            return_fft=False,
        )

        out = scatterer.resolve()

        self.assertIsInstance(out.array, np.ndarray)
        self.assertEqual(out.shape, (32, 32, 1))
        
        arr = out.array
        self.assertTrue(np.iscomplexobj(arr))
        self.assertTrue(np.isfinite(arr.real).all())
        self.assertTrue(np.isfinite(arr.imag).all())
        self.assertGreater(np.abs(arr).sum(), 0)

        self.assertTrue(
            np.allclose(
                np.asarray(out.properties["position"]),
                np.array([16, 16]),
            )
        )

    def test_MieSphere_rejects_none_polarizations(self):
        with self.assertRaises(ValueError):
            scatterers.MieSphere(
                radius=0.5e-6,
                refractive_index=1.45,
                wavelength=680e-9,
                refractive_index_medium=1.33,
                NA=0.7,
                output_region=(0, 0, 32, 32),
                input_polarization=None,
                output_polarization=0.0,
            ).resolve()

        with self.assertRaises(ValueError):
            scatterers.MieSphere(
                radius=0.5e-6,
                refractive_index=1.45,
                wavelength=680e-9,
                refractive_index_medium=1.33,
                NA=0.7,
                output_region=(0, 0, 32, 32),
                input_polarization=0.0,
                output_polarization=None,
            ).resolve()

    def test_MieSphere_auto_parameters(self):
        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45,
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            input_polarization=0.0,
            output_polarization=0.0,
            L="auto",
            collection_angle="auto",
            offset_z="auto",
        )

        out = scatterer.resolve()

        self.assertIsInstance(out.array, np.ndarray)
        self.assertIsInstance(out.properties["L"], int)
        self.assertGreater(out.properties["L"], 0)
        self.assertTrue(np.isscalar(out.properties["collection_angle"]))
        self.assertGreater(float(out.properties["collection_angle"]), 0)
        self.assertGreater(float(out.properties["offset_z"]), 0)

    def test_MieSphere_modes(self):
        common_kwargs = dict(
            radius=0.5e-6,
            refractive_index=1.45,
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            padding=(0, 0, 0, 0),
            input_polarization=0.0,
            output_polarization=0.0,
            return_fft=False,
        )

        out_geom = scatterers.MieSphere(
            mode="geometric",
            **common_kwargs,
        ).resolve()

        out_hybrid = scatterers.MieSphere(
            mode="hybrid",
            **common_kwargs,
        ).resolve()

        self.assertIsInstance(out_geom.array, np.ndarray)
        self.assertIsInstance(out_hybrid.array, np.ndarray)

        self.assertEqual(out_geom.shape, out_hybrid.shape)

        a_geom = out_geom.array
        a_hybrid = out_hybrid.array

        self.assertTrue(np.iscomplexobj(a_geom))
        self.assertTrue(np.iscomplexobj(a_hybrid))

        self.assertTrue(np.isfinite(a_geom.real).all())
        self.assertTrue(np.isfinite(a_geom.imag).all())
        self.assertTrue(np.isfinite(a_hybrid.real).all())
        self.assertTrue(np.isfinite(a_hybrid.imag).all())

        self.assertGreater(np.abs(a_geom).sum(), 0)
        self.assertGreater(np.abs(a_hybrid).sum(), 0)

        ratio = np.abs(a_geom).sum() / np.abs(a_hybrid).sum()
        self.assertGreater(ratio, 1e-2)
        self.assertLess(ratio, 1e2)

    def test_MieStratifiedSphere(self):
        scatterer = scatterers.MieStratifiedSphere(
            radius=(0.5e-6, 1.0e-6),
            refractive_index=(1.45, 1.52),
            position=(16, 16),
            position_unit="pixel",
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            padding=(0, 0, 0, 0),
            input_polarization=0.0,
            output_polarization=0.0,
            return_fft=False,
        )

        out = scatterer.resolve()

        self.assertIsInstance(out.array, np.ndarray)
        self.assertEqual(out.shape[-1], 1)

        arr = out.array
        self.assertTrue(np.iscomplexobj(arr))
        self.assertTrue(np.isfinite(arr.real).all())
        self.assertTrue(np.isfinite(arr.imag).all())
        self.assertGreater(np.abs(arr).sum(), 0)

        self.assertTrue(
            np.allclose(
                np.asarray(out.properties["position"]),
                np.array([16, 16]),
            )
        )

    def test_MieStratifiedSphere_rejects_none_polarizations(self):
        with self.assertRaises(ValueError):
            scatterers.MieStratifiedSphere(
                radius=(0.5e-6, 1.0e-6),
                refractive_index=(1.45, 1.52),
                wavelength=680e-9,
                refractive_index_medium=1.33,
                NA=0.7,
                output_region=(0, 0, 32, 32),
                input_polarization=None,
                output_polarization=0.0,
            ).resolve()

        with self.assertRaises(ValueError):
            scatterers.MieStratifiedSphere(
                radius=(0.5e-6, 1.0e-6),
                refractive_index=(1.45, 1.52),
                wavelength=680e-9,
                refractive_index_medium=1.33,
                NA=0.7,
                output_region=(0, 0, 32, 32),
                input_polarization=0.0,
                output_polarization=None,
            ).resolve()

    def test_MieStratifiedSphere_auto_parameters(self):
        scatterer = scatterers.MieStratifiedSphere(
            radius=(0.5e-6, 1.0e-6),
            refractive_index=(1.45, 1.52),
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            input_polarization=0.0,
            output_polarization=0.0,
            L="auto",
            collection_angle="auto",
            offset_z="auto",
        )

        out = scatterer.resolve()

        self.assertIsInstance(out.array, np.ndarray)
        self.assertIsInstance(out.properties["L"], int)
        self.assertGreater(out.properties["L"], 0)
        self.assertTrue(np.isscalar(out.properties["collection_angle"]))
        self.assertGreater(float(out.properties["collection_angle"]), 0)
        self.assertGreater(float(out.properties["offset_z"]), 0)

    def test_MieStratifiedSphere_rejects_nonmonotonic_radii(self):
        with self.assertRaises(ValueError):
            scatterers.MieStratifiedSphere(
                radius=(1.0e-6, 0.5e-6),
                refractive_index=(1.45, 1.52),
                wavelength=680e-9,
                refractive_index_medium=1.33,
                NA=0.7,
                output_region=(0, 0, 32, 32),
                input_polarization=0.0,
                output_polarization=0.0,
            ).resolve()

    def test_MieStratifiedSphere_modes(self):
        common_kwargs = dict(
            radius=(0.5e-6, 1.0e-6),
            refractive_index=(1.45, 1.52),
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            padding=(0, 0, 0, 0),
            input_polarization=0.0,
            output_polarization=0.0,
            return_fft=False,
        )

        out_geom = scatterers.MieStratifiedSphere(
            mode="geometric",
            **common_kwargs,
        ).resolve()

        out_hybrid = scatterers.MieStratifiedSphere(
            mode="hybrid",
            **common_kwargs,
        ).resolve()

        self.assertIsInstance(out_geom.array, np.ndarray)
        self.assertIsInstance(out_hybrid.array, np.ndarray)

        self.assertEqual(out_geom.shape, out_hybrid.shape)

        a_geom = out_geom.array
        a_hybrid = out_hybrid.array

        self.assertTrue(np.iscomplexobj(a_geom))
        self.assertTrue(np.iscomplexobj(a_hybrid))

        self.assertTrue(np.isfinite(a_geom.real).all())
        self.assertTrue(np.isfinite(a_geom.imag).all())
        self.assertTrue(np.isfinite(a_hybrid.real).all())
        self.assertTrue(np.isfinite(a_hybrid.imag).all())

        self.assertGreater(np.abs(a_geom).sum(), 0)
        self.assertGreater(np.abs(a_hybrid).sum(), 0)

        ratio = np.abs(a_geom).sum() / np.abs(a_hybrid).sum()
        self.assertGreater(ratio, 1e-2)
        self.assertLess(ratio, 1e2)

    def test_Incoherent_passthrough(self):
        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45,
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            input_polarization=0.0,
            output_polarization=0.0,
        )

        wrapped = scatterers.Incoherent(
            scatterer,
            input_unpolarized=False,
            output_unpolarized=False,
        )

        out_direct = scatterer.resolve()
        out_wrapped = wrapped.resolve()

        np.testing.assert_allclose(out_direct.array, out_wrapped.array)

    def test_Incoherent_unpolarized_input(self):
        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45,
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            input_polarization=0.0,
            output_polarization=0.0,
        )

        wrapped = scatterers.Incoherent(
            scatterer,
            input_unpolarized=True,
            output_unpolarized=False,
        )

        out = wrapped.resolve()
        arr = out

        self.assertEqual(arr.shape, (32, 32, 1))
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.sum(), 0)
        self.assertTrue(np.isrealobj(arr) or np.allclose(arr.imag, 0))

    def test_Incoherent_unpolarized_input_and_output(self):
        scatterer = scatterers.MieSphere(
            radius=0.5e-6,
            refractive_index=1.45,
            wavelength=680e-9,
            refractive_index_medium=1.33,
            NA=0.7,
            output_region=(0, 0, 32, 32),
            input_polarization=0.0,
            output_polarization=0.0,
        )

        wrapped = scatterers.Incoherent(
            scatterer,
            input_unpolarized=True,
            output_unpolarized=True,
        )

        out = wrapped.resolve()
        arr = out

        self.assertEqual(arr.shape, (32, 32, 1))
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreater(arr.sum(), 0)
        self.assertTrue(np.isrealobj(arr) or np.allclose(arr.imag, 0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestScatterers_Torch(TestScatterers_NumPy):
    BACKEND = "torch"


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMath_TorchOnly(BackendTestBase):
    BACKEND = "torch"

    def test_point_particle_intensity_gradient(self):

        # --- PointParticle intensity optimization ---
        optics = Fluorescence(
            NA=0.7,
            wavelength=500e-9,
            resolution=1e-6,
            magnification=4,
            output_region=(0, 0, 32, 32),
        )
        # target
        true_intensity = 2.0
        particle = scatterers.PointParticle(
            position=(16, 16),
            intensity=true_intensity,
        )
        target = optics(particle).update()().detach()
        # learnable parameter
        intensity = torch.tensor(0.5, requires_grad=True)
        # scatterer with learnable intensity
        particle = scatterers.PointParticle(
            position=(16, 16),
            intensity=intensity,
        )
        optimizer = torch.optim.Adam([intensity], lr=0.1)
        pipeline = optics(particle)
        prev_loss = None
        for _ in range(20):
            optimizer.zero_grad()
            image = pipeline.update()()
            loss = ((image - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            self.assertIsNotNone(intensity.grad)
            if prev_loss is not None:
                self.assertNotEqual(loss.item(), prev_loss)
            prev_loss = loss.item()
        self.assertTrue(abs(intensity.item() - true_intensity) < 0.5)

        

    


if __name__ == "__main__":
    unittest.main()
