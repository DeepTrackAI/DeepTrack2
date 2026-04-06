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
        self.assertEqual(v1.shape, (3, 5, 1))
        self.assertEqual(v3.shape, (5, 6, 1))
        self.assertTrue(
            np.allclose(
                np.asarray(v1.properties["position"]),
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


    # def test_Ellipse(self):

    #     def make_optics(upscale=1):
    #         return Fluorescence(
    #             NA=0.7,
    #             wavelength=680e-9,
    #             resolution=1e-6,
    #             magnification=10,
    #             output_region=(0, 0, 64, 64),
    #             upscale=upscale,
    #         )

    #     def make_scatterer(radius=(1e-6, 0.5e-6)):
    #         return scatterers.Ellipse(
    #             intensity=100,
    #             position_unit="pixel",
    #             position=(32, 32),
    #             radius=radius,
    #         )

    #     # --- Imaging test ---
    #     optics = make_optics()
    #     scatterer = make_scatterer()
    #     out = optics(scatterer).resolve()

    #     self.assertIsInstance(out, self.array_type)
    #     self.assertEqual(out.shape, (64, 64, 1))

    #     # --- Upscale consistency ---
    #     s2 = make_scatterer()
    #     make_optics(upscale=2)(s2).resolve()
    #     shape2 = s2().shape
    #     s4 = make_scatterer()
    #     make_optics(upscale=4)(s4).resolve()
    #     shape4 = s4().shape
    #     # Z must exist
    #     self.assertEqual(shape2[-1], 1)
    #     self.assertEqual(shape4[-1], 1)
    #     # upscale increases resolution
    #     self.assertGreater(shape4[0], shape2[0])
    #     self.assertGreater(shape4[1], shape2[1])

    #     # --- Asymmetric upscale ---
    #     s_y = make_scatterer(radius=(1e-6, 1e-6))
    #     make_optics(upscale=(2, 1, 1))(s_y).resolve()
    #     shape_y = s_y().shape
    #     s_x = make_scatterer(radius=(1e-6, 1e-6))
    #     make_optics(upscale=(1, 2, 1))(s_x).resolve()
    #     shape_x = s_x().shape
    #     # anisotropy check
    #     self.assertGreater(shape_y[0], shape_y[1])  # stretched in Y
    #     self.assertGreater(shape_x[1], shape_x[0])  # stretched in X

    #     # --- Translation invariance ---
    #     s1 = scatterers.Ellipse(position=(32, 32))
    #     s2 = scatterers.Ellipse(position=(40, 40))
    #     v1 = s1()
    #     v2 = s2()
    #     self.assertEqual(v1.shape, v2.shape)

    #     # --- Rotation invariance ---
    #     r1 = (1e-6, 0.5e-6)
    #     r2 = (0.5e-6, 1e-6)
    #     s1 = scatterers.Ellipse(
    #         radius=r1,
    #         rotation=0,
    #     )
    #     s2 = scatterers.Ellipse(
    #         radius=r2,
    #         rotation=np.pi / 2,
    #     )
    #     v1 = s1().array.squeeze()
    #     v2 = s2().array.squeeze()
    #     # allow small interpolation differences
    #     np.testing.assert_allclose(v1, v2, atol=1e-6)

    # def test_Sphere(self):

    #     def make_optics(upscale=1):
    #         return Fluorescence(
    #             NA=0.7,
    #             wavelength=680e-9,
    #             resolution=1e-6,
    #             magnification=10,
    #             output_region=(0, 0, 16, 16),
    #             upscale=upscale,
    #         )

    #     def make_scatterer():
    #         return scatterers.Sphere(
    #             intensity=100,
    #             position_unit="pixel",
    #             position=(8, 8),
    #             radius=1e-6,
    #         )

    #     # --- Imaging test ---
    #     optics = make_optics()
    #     scatterer = make_scatterer()
    #     out = optics(scatterer).resolve()
    #     self.assertIsInstance(out, self.array_type)
    #     self.assertEqual(out.shape, (16, 16, 1))

    #     # --- Volume properties ---
    #     v = scatterer()
    #     shape = v.shape
    #     # cube volume
    #     self.assertEqual(len(shape), 3)
    #     self.assertEqual(shape[0], shape[1])
    #     self.assertEqual(shape[1], shape[2])
    #     # must be centered → odd size preferred
    #     self.assertTrue(shape[0] > 0)
    #     # non-empty
    #     self.assertGreater(v.array.sum(), 0)

    #     # --- Upscale consistency ---
    #     s2 = make_scatterer()
    #     make_optics(upscale=2)(s2).resolve()
    #     shape2 = s2().shape
    #     s4 = make_scatterer()
    #     make_optics(upscale=4)(s4).resolve()
    #     shape4 = s4().shape
    #     # upscale increases resolution
    #     self.assertEqual(shape4[0], 2*shape2[0])
    #     # still cubic
    #     self.assertEqual(shape2[0], shape2[1])
    #     self.assertEqual(shape4[1], shape4[2])

    #     # --- Radial symmetry check ---
    #     v = scatterer().array if hasattr(scatterer(), "array") else scatterer()
    #     v = v.squeeze()
    #     center = tuple(s // 2 for s in v.shape)
    #     # sample along axes
    #     r_x = v[center[0], center[1], :]
    #     r_y = v[center[0], :, center[2]]
    #     r_z = v[:, center[1], center[2]]
    #     diff = xp.mean(xp.abs(r_x - r_y))
    #     diff = diff.item() if hasattr(diff, "item") else diff
    #     self.assertLess(diff, 1e-3)
    #     diff = xp.mean(xp.abs(r_y - r_z))
    #     diff = diff.item() if hasattr(diff, "item") else diff
    #     self.assertLess(diff, 1e-3)


    # def test_Ellipsoid(self):
    #     optics = Fluorescence(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=10,
    #         output_region=(0, 0, 64, 64),
    #     )
    #     scatterer = scatterers.Ellipsoid(
    #         intensity=100,
    #         position_unit="pixel",
    #         position=(32, 32),
    #         radius=(1e-6, 0.5e-6, 0.25e-6),
    #         rotation=(np.pi / 4, 0, 0),
    #         upsample=4,
    #     )
    #     imaged_scatterer = optics(scatterer)
    #     output_image = imaged_scatterer.resolve()
    #     self.assertIsInstance(output_image, self.array_type)
    #     self.assertEqual(output_image.shape, (64, 64, 1))

    # def test_EllipsoidUpscale(self):
    #     optics = Fluorescence(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=10,
    #         output_region=(0, 0, 64, 64),
    #         upscale=2,
    #     )
    #     scatterer = scatterers.Ellipsoid(
    #         intensity=100,
    #         position_unit="pixel",
    #         position=(32, 32),
    #         radius=(1e-6, 0.5e-6, 0.25e-6),
    #         # rotation=(np.pi / 4, 0, 0),
    #     )
    #     imaged_scatterer = optics(scatterer)
    #     imaged_scatterer.resolve()
    #     scatterer_volume = scatterer()
    #     self.assertEqual(scatterer_volume.shape, (21, 40, 11))

    # def test_EllipsoidUpscaleAsymmetric(self):
    #     optics = Fluorescence(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=10,
    #         output_region=(0, 0, 64, 64),
    #         upscale=(4, 2, 2),
    #     )
    #     scatterer = scatterers.Ellipsoid(
    #         intensity=100,
    #         position_unit="pixel",
    #         position=(32, 32),
    #         radius=(1e-6, 0.5e-6, 0.25e-6),
    #         # rotation=(np.pi / 4, 0, 0),
    #     )
    #     imaged_scatterer = optics(scatterer)
    #     imaged_scatterer.resolve()
    #     scatterer_volume = scatterer()
    #     self.assertEqual(scatterer_volume.shape, (41, 41, 11))

    #     optics = Fluorescence(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=10,
    #         output_region=(0, 0, 64, 64),
    #         upscale=(2, 4, 2),
    #     )
    #     scatterer = scatterers.Ellipsoid(
    #         intensity=100,
    #         position_unit="pixel",
    #         position=(32, 32),
    #         radius=(1e-6, 0.5e-6, 0.25e-6),
    #         # rotation=(np.pi / 4, 0, 0),
    #     )
    #     imaged_scatterer = optics(scatterer)
    #     imaged_scatterer.resolve()
    #     scatterer_volume = scatterer()
    #     self.assertEqual(scatterer_volume.shape, (21, 80, 11))

    #     optics = Fluorescence(
    #         NA=0.7,
    #         wavelength=680e-9,
    #         resolution=1e-6,
    #         magnification=10,
    #         output_region=(0, 0, 64, 64),
    #         upscale=(2, 2, 4),
    #     )
    #     scatterer = scatterers.Ellipsoid(
    #         intensity=100,
    #         position_unit="pixel",
    #         position=(32, 32),
    #         radius=(1e-6, 0.5e-6, 0.25e-6),
    #         # rotation=(np.pi / 4, 0, 0),
    #     )
    #     imaged_scatterer = optics(scatterer)
    #     imaged_scatterer.resolve()
    #     scatterer_volume = scatterer()
    #     self.assertEqual(scatterer_volume.shape, (21, 41, 21))

    # def test_MieSphere(self):
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

    #     scatterer = scatterers.MieSphere(
    #         radius=0.5e-6, refractive_index=1.45 + 0.1j, aperature_angle=0.1
    #     )

    #     imaged_scatterer_1 = optics_1(scatterer)

    #     imaged_scatterer_1.update().resolve()

    # def test_MieSphere_Coherence_length(self):
    #     optics_1 = Brightfield(
    #         NA=0.15,
    #         wavelength=633e-9,
    #         resolution=2e-6,
    #         magnification=1,
    #         output_region=(0, 0, 256, 256),
    #         return_field=True,
    #     )

    #     scatterer = scatterers.MieSphere(
    #         position=(128, 128),
    #         radius=3e-6,
    #         refractive_index=1.45 + 0.1j,
    #         z=2612 * 1e-6,
    #         coherence_length=5.9e-05,
    #     )

    #     imaged_scatterer_1 = optics_1(scatterer)

    #     imaged_scatterer_1.update().resolve()

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

# TODO: Extending the test and setting the backend to torch
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
