# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path

import unittest
import warnings

import numpy as np

from deeptrack.backend import TORCH_AVAILABLE
from deeptrack.optical.optics import Brightfield, Fluorescence
from deeptrack.optical import scatterers
from tests import BackendTestBase

from packaging.version import parse as parse_version

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
        return (
            x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        )

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

        self.assertTrue(
            np.allclose(
                np.asarray(v1.properties["position"]), np.array([16, 16])
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(v3.properties["position"]), np.array([16, 16])
            )
        )

        a1 = self.to_numpy(v1.array)
        a3 = self.to_numpy(v3.array)

        self.assertGreater(a1.sum(), 0)
        self.assertGreater(a3.sum(), 0)
        self.assertTrue(np.any((a3 > 0) & (a3 <= 1.0)))

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

        self.assertTrue(
            np.allclose(
                np.asarray(v1.properties["position"]), np.array([16, 16])
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(v3.properties["position"]), np.array([16, 16])
            )
        )

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

    def _torch_mie_sphere(self, mode, radius, refractive_index, **kwargs):
        params = dict(
            radius=radius,
            refractive_index=refractive_index,
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
            L=5,
            collection_angle=0.3,
            offset_z=1e-5,
            mode=mode,
        )
        params.update(kwargs)
        return scatterers.MieSphere(**params)

    def test_mie_sphere_resolves_with_torch_autodiff(self):
        for mode in ("geometric", "hybrid"):
            with self.subTest(mode=mode):
                radius = torch.tensor(
                    0.5e-6,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                refractive_index = torch.tensor(
                    1.45,
                    dtype=torch.float64,
                    requires_grad=True,
                )

                out = self._torch_mie_sphere(
                    mode,
                    radius,
                    refractive_index,
                ).resolve()

                self.assertIsInstance(out.array, torch.Tensor)
                self.assertEqual(out.shape, (32, 32, 1))
                self.assertTrue(torch.is_complex(out.array))
                self.assertTrue(torch.isfinite(out.array.real).all())
                self.assertTrue(torch.isfinite(out.array.imag).all())
                self.assertGreater(
                    float(torch.abs(out.array).sum().detach()),
                    0,
                )
                self.assertTrue(out.array.requires_grad)

                loss = torch.abs(out.array).sum()
                loss.backward()

                self.assertIsNotNone(radius.grad)
                self.assertIsNotNone(refractive_index.grad)
                self.assertTrue(torch.isfinite(radius.grad))
                self.assertTrue(torch.isfinite(refractive_index.grad))
                self.assertGreater(abs(float(radius.grad)), 0)
                self.assertGreater(abs(float(refractive_index.grad)), 0)

    @unittest.skipIf(
        not TORCH_AVAILABLE
        or parse_version(torch.__version__) < parse_version("2.9"),
        "Autograd through Mie scatterer requires torch >= 2.9"
    )
    def test_mie_sphere_brightfield_sums_multiple_torch_fields(self):
        radius_1 = torch.tensor(
            0.45e-6,
            dtype=torch.float64,
            requires_grad=True,
        )
        radius_2 = torch.tensor(
            0.55e-6,
            dtype=torch.float64,
            requires_grad=True,
        )

        common = dict(
            refractive_index=1.45,
            input_polarization=0.0,
            output_polarization=0.0,
            L=5,
            collection_angle=0.3,
            offset_z=1e-5,
            mode="hybrid",
        )
        sample = scatterers.MieSphere(
            radius=radius_1,
            position=(14, 16),
            position_unit="pixel",
            **common,
        ) >> scatterers.MieSphere(
            radius=radius_2,
            position=(18, 16),
            position_unit="pixel",
            **common,
        )
        microscope = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 32, 32),
            padding=(4, 4, 4, 4),
            return_field=True,
        )

        image = microscope(sample).resolve()

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (32, 32, 1))
        self.assertTrue(torch.is_complex(image))
        self.assertTrue(torch.isfinite(image.real).all())
        self.assertTrue(torch.isfinite(image.imag).all())

        loss = torch.abs(image).sum()
        loss.backward()

        self.assertIsNotNone(radius_1.grad)
        self.assertIsNotNone(radius_2.grad)
        self.assertTrue(torch.isfinite(radius_1.grad))
        self.assertTrue(torch.isfinite(radius_2.grad))
        self.assertGreater(abs(float(radius_1.grad)), 0)
        self.assertGreater(abs(float(radius_2.grad)), 0)

    @unittest.skipIf(
        not TORCH_AVAILABLE
        or parse_version(torch.__version__) < parse_version("2.9"),
        "Autograd through Mie scatterer requires torch >= 2.9"
    )
    def test_mie_sphere_brightfield_autodiff_learnable_parameters(self):
        cases = [
            ("x", 14.25, "sample"),
            ("y", 16.75, "sample"),
            ("resolution", 1.0e-6, "optics"),
            ("NA", 0.7, "optics"),
            ("magnification", 10.0, "optics"),
            ("wavelength", 680e-9, "optics"),
            ("refractive_index_medium", 1.33, "optics"),
        ]

        for name, value, owner in cases:
            with self.subTest(parameter=name):
                parameter = torch.tensor(
                    value,
                    dtype=torch.float64,
                    requires_grad=True,
                )

                sample_kwargs = dict(
                    radius=0.5e-6,
                    refractive_index=1.45,
                    position=(14.25, 16.75),
                    position_unit="pixel",
                    input_polarization=0.0,
                    output_polarization=0.0,
                    L=5,
                    collection_angle=0.3,
                    offset_z=1e-5,
                    mode="hybrid",
                )
                optics_kwargs = dict(
                    NA=0.7,
                    wavelength=680e-9,
                    refractive_index_medium=1.33,
                    resolution=1e-6,
                    magnification=10,
                    output_region=(0, 0, 32, 32),
                    padding=(4, 4, 4, 4),
                    return_field=True,
                )

                if owner == "sample" and name == "x":
                    sample_kwargs["position"] = (parameter, 16.75)
                elif owner == "sample" and name == "y":
                    sample_kwargs["position"] = (14.25, parameter)
                else:
                    optics_kwargs[name] = parameter
                    if name == "NA":
                        sample_kwargs.pop("collection_angle")
                        sample_kwargs.pop("offset_z")

                sample = scatterers.MieSphere(**sample_kwargs)
                microscope = Brightfield(**optics_kwargs)

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    image = microscope(sample).resolve()

                tensor_warning = (
                    "Converting a tensor with requires_grad=True to a scalar"
                )
                self.assertFalse(
                    any(tensor_warning in str(w.message) for w in caught)
                )
                self.assertIsInstance(image, torch.Tensor)
                self.assertTrue(image.requires_grad)
                self.assertTrue(torch.isfinite(image.real).all())
                self.assertTrue(torch.isfinite(image.imag).all())

                weights = torch.linspace(
                    0.5,
                    1.5,
                    image.numel(),
                    dtype=image.real.dtype,
                    device=image.device,
                ).reshape(image.shape)
                loss = (torch.abs(image) * weights).sum()
                loss.backward()

                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad))
                self.assertGreater(abs(float(parameter.grad)), 0)

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
