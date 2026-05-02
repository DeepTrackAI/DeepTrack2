import unittest
import numpy as np

from deeptrack.optical import aberrations
from deeptrack.optical.optics import Fluorescence
from deeptrack.scatterers import PointParticle
from deeptrack.backend import TORCH_AVAILABLE
from tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestAberrations_NumPy(BackendTestBase):
    BACKEND = "numpy"

    def setUp(self):
        super().setUp()
        self.particle = PointParticle(
            position=(32, 32),
            position_unit="pixel",
            intensity=1,
        )

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        if self.BACKEND == "torch":
            return torch.Tensor
        raise ValueError(f"Unsupported backend: {self.BACKEND}")

    def _make_optics(self, pupil):
        return Fluorescence(
            NA=0.3,
            resolution=1e-6,
            magnification=10,
            wavelength=530e-9,
            output_region=(0, 0, 64, 48),
            padding=(64, 64, 64, 64),
            pupil=pupil,
        )
    
    def _to_numpy(self, x):
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _render(self, pupil=None, z=0):
        optics = self._make_optics(pupil)
        image = optics(self.particle).resolve(z=z)

        self.assertIsInstance(image, self.array_type)
        self.assertEqual(image.shape, (64, 48, 1))

        return self._to_numpy(image[..., 0])
    
    def _com(self, img):
        y, x = np.indices(img.shape)
        s = img.sum()
        return (y * img).sum() / s, (x * img).sum() / s

    def _second_moments(self, img):
        cy, cx = self._com(img)
        y, x = np.indices(img.shape)
        s = img.sum()
        vy = (img * (y - cy) ** 2).sum() / s
        vx = (img * (x - cx) ** 2).sum() / s
        return vy, vx

    def _radial_second_moment(self, img):
        cy, cx = self._com(img)
        y, x = np.indices(img.shape)
        r2 = (y - cy) ** 2 + (x - cx) ** 2
        return (img * r2).sum() / img.sum()

    def _assert_resolves(self, pupil):
        aberrated_optics = self._make_optics(pupil)
        aberrated_particle = aberrated_optics(self.particle)

        for z in (-100, 0, 100):
            with self.subTest(z=z, pupil=type(pupil).__name__):
                im = aberrated_particle.resolve(z=z)
                self.assertIsInstance(im, self.array_type)
                self.assertEqual(im.shape, (64, 48, 1))

    def test___all__(self):
        from deeptrack import (
            GaussianApodization,
            Zernike,
            Piston,
            VerticalTilt,
            HorizontalTilt,
            ObliqueAstigmatism,
            Defocus,
            Astigmatism,
            ObliqueTrefoil,
            VerticalComa,
            HorizontalComa,
            Trefoil,
            SphericalAberration,
        )

    def testGaussianApodization_resolves(self):
        self._assert_resolves(
            aberrations.GaussianApodization(sigma=0.5)
        )

    def testGaussianApodization_reduces_peak(self):
        for z in (-100, 0, 100):
            with self.subTest(z=z):
                base = self._render(pupil=None, z=z)
                out = self._render(
                    pupil=aberrations.GaussianApodization(sigma=0.5),
                    z=z,
                )
                self.assertLess(out.max(), base.max())

    def testZernike_resolves(self):
        self._assert_resolves(
            aberrations.Zernike(n=[2, 3], m=[0, 1], coefficient=[0.5, 0.3])
        )

    def testPiston_resolves(self):
        self._assert_resolves(aberrations.Piston(coefficient=1))

    def testPiston_image_invariant(self):
        for z in (-100, 0, 100):
            with self.subTest(z=z):
                base = self._render(pupil=None, z=z)
                out = self._render(
                    pupil=aberrations.Piston(coefficient=1),
                    z=z,
                )
                np.testing.assert_allclose(out, base, atol=1e-6, rtol=1e-6)

    def testVerticalTilt_resolves(self):
        self._assert_resolves(aberrations.VerticalTilt(coefficient=1))

    def testVerticalTilt_shifts_y(self):
        base = self._render(pupil=None, z=0)
        out = self._render(pupil=aberrations.VerticalTilt(coefficient=5), z=0)

        cy0, cx0 = self._com(base)
        cy1, cx1 = self._com(out)

        self.assertGreater(abs(cy1 - cy0), 0.05)
        self.assertLess(abs(cx1 - cx0), abs(cy1 - cy0))

    def testHorizontalTilt_resolves(self):
        self._assert_resolves(aberrations.HorizontalTilt(coefficient=1))

    def testHorizontalTilt_shifts_x(self):
        base = self._render(pupil=None, z=0)
        out = self._render(pupil=aberrations.HorizontalTilt(coefficient=5), z=0)

        cy0, cx0 = self._com(base)
        cy1, cx1 = self._com(out)

        self.assertGreater(abs(cx1 - cx0), 0.05)
        self.assertLess(abs(cy1 - cy0), abs(cx1 - cx0))

    def testObliqueAstigmatism_resolves(self):
        self._assert_resolves(aberrations.ObliqueAstigmatism(coefficient=1))

    def testDefocus_resolves(self):
        self._assert_resolves(aberrations.Defocus(coefficient=1))

    def testDefocus_matches_Zernike(self):
        img1 = self._render(pupil=aberrations.Defocus(coefficient=1), z=0)
        img2 = self._render(pupil=aberrations.Zernike(n=2, m=0, coefficient=1), z=0)
        np.testing.assert_allclose(img1, img2, atol=1e-6, rtol=1e-6)

    def testDefocus_broadens_psf(self):
        base = self._render(pupil=None, z=0)
        out = self._render(pupil=aberrations.Defocus(coefficient=1), z=0)

        self.assertLess(out.max(), base.max())
        self.assertGreater(
            self._radial_second_moment(out),
            self._radial_second_moment(base),
        )

    def testAstigmatism_resolves(self):
        self._assert_resolves(aberrations.Astigmatism(coefficient=1))

    def testAstigmatism_breaks_xy_symmetry(self):
        base = self._render(pupil=None, z=0)
        out = self._render(pupil=aberrations.Astigmatism(coefficient=1), z=0)

        vy0, vx0 = self._second_moments(base)
        vy1, vx1 = self._second_moments(out)

        base_anisotropy = abs(vy0 - vx0)
        out_anisotropy = abs(vy1 - vx1)

        self.assertGreater(out_anisotropy, base_anisotropy)

    def testObliqueTrefoil_resolves(self):
        self._assert_resolves(aberrations.ObliqueTrefoil(coefficient=1))

    def testVerticalComa_resolves(self):
        self._assert_resolves(aberrations.VerticalComa(coefficient=1))

    def testHorizontalComa_resolves(self):
        self._assert_resolves(aberrations.HorizontalComa(coefficient=1))

    def testTrefoil_resolves(self):
        self._assert_resolves(aberrations.Trefoil(coefficient=1))

    def testSphericalAberration_resolves(self):
        self._assert_resolves(aberrations.SphericalAberration(coefficient=1))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestAberrations_PyTorch(TestAberrations_NumPy):
    BACKEND = "torch"


if __name__ == "__main__":
    unittest.main()