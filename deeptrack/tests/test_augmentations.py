import sys

# sys.path.append(".")  # Adds the module to path

import unittest

import numpy as np

from deeptrack import (
    augmentations,
    config,
    features,
    scatterers,
    sources,
    TORCH_AVAILABLE,
)
# from deeptrack import units_registry as u

if TORCH_AVAILABLE:
    import torch


class TestAugmentations(unittest.TestCase):

    class _ProjectSum(features.Feature):
        """Minimal 'optics-like' projection: sums a list of inputs."""

        __distributed__ = False

        def get(self, inputs, **kwargs):
            if inputs is None:
                return None

            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            arrays = []
            for x in inputs:
                if hasattr(x, "array"):
                    arrays.append(x.array)
                else:
                    arrays.append(x)

            out = arrays[0]
            for a in arrays[1:]:
                out = out + a
            return out
    
    @staticmethod
    def make_ellipse(H, W, cy, cx, ry, rx):
        yy, xx = np.meshgrid(
            np.arange(H),
            np.arange(W),
            indexing="ij"
        )
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        return mask.astype(np.float32)[..., None]

    @staticmethod
    def center_of_mass(img):
        """
        Backend-agnostic center of mass.
        Works for:
            - np.ndarray (H, W) or (H, W, C)
            - torch.Tensor (H, W) or (H, W, C)
        """

        if img.ndim == 3:
            img = img[..., 0]

        if hasattr(img, "detach"):  # torch
            import torch

            H, W = img.shape
            device = img.device
            dtype = img.dtype

            ys = torch.arange(H, dtype=dtype, device=device)
            xs = torch.arange(W, dtype=dtype, device=device)

            Y, X = torch.meshgrid(ys, xs, indexing="ij")

            mass = img.sum()
            cy = (img * Y).sum() / mass
            cx = (img * X).sum() / mass

            return float(cy), float(cx)

        else:  # numpy
            import numpy as np

            H, W = img.shape
            ys = np.arange(H)
            xs = np.arange(W)
            Y, X = np.meshgrid(ys, xs, indexing="ij")

            mass = img.sum()
            cy = (img * Y).sum() / mass
            cx = (img * X).sum() / mass

            return float(cy), float(cx)


    def test_Reuse(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            # Deterministic increasing feature
            counter = {"i": 0}

            class CounterFeature(features.Feature):
                __distributed__ = False

                def get(self, data=None, **kwargs):
                    counter["i"] += 1
                    return counter["i"]

            base_feature = CounterFeature()

            reuse = augmentations.Reuse(base_feature, uses=2, storage=2)

            # First calls must compute because cache fills
            out1 = reuse.update()()
            out2 = reuse.update()()

            self.assertEqual(out1, 1)
            self.assertEqual(out2, 2)

            # Next calls reuse cache
            out3 = reuse.update()()
            out4 = reuse.update()()

            # Should not increment underlying feature yet
            self.assertEqual(out3, 1)
            self.assertEqual(out4, 2)
            self.assertEqual(counter["i"], 2)

            # After uses*storage = 4 calls, recompute
            out5 = reuse.update()()
            self.assertEqual(counter["i"], 3)
            self.assertEqual(out5, 3)

            # Ensure storage trimming works
            self.assertLessEqual(len(reuse.cache), 2)

            # Test update resets evaluation cycle
            out6 = reuse.update()()
            self.assertEqual(counter["i"], 3)
            self.assertEqual(out6, 3)


    def test_FlipLR(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            # Ensure both the augmentation and the test arrays align
            # with the active backend.
            config.set_backend(backend)

            H, W, C = 3, 4, 1
            base_np = np.arange(H * W, dtype=np.float32).reshape(H, W, C)

            base = base_np if backend == "numpy" else torch.tensor(base_np)

            flip = augmentations.FlipLR(augment=True)

           # check array flipping correctness
            out = flip(base) 
            if backend == "numpy":
                expected = base[:, ::-1, :]
                np.testing.assert_array_equal(out, expected)
            else:
                expected = torch.flip(base, dims=[1])
                self.assertTrue(torch.equal(out, expected))

            # check that flipping twice returns the original array
            out2 = flip(out)
            if backend == "numpy":
                np.testing.assert_array_equal(out2, base)
            else:
                self.assertTrue(torch.equal(out2, base))

            # check scatteredVolume correctness + position update
            # Convention: position = [y, x]
            position = np.array([1, 1], dtype=np.float32)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties=[{"position": position.copy()}],
            )

            flipped_volume = flip(volume)

            if backend == "numpy":
                expected_array = base[:, ::-1, :]
                np.testing.assert_array_equal(flipped_volume.array, expected_array)
            else:
                expected_array = torch.flip(base, dims=[1])
                self.assertTrue(torch.equal(flipped_volume.array, expected_array))

            # Position update (x mirrored around width)
            expected_x = (W - 1) - position[1]
            got_pos = flipped_volume.properties[0]["position"]
            self.assertEqual(float(got_pos[0]), float(position[0]))
            self.assertEqual(float(got_pos[1]), float(expected_x))

            # check list behavior (arrays + scattered volumes)
            # Arrays list
            arrays_list = [base, base]
            out_list = flip(arrays_list)
            self.assertIsInstance(out_list, list)
            self.assertEqual(len(out_list), 2)

            # Volumes list
            vol2 = volume.copy()
            vol_list = [volume, vol2]
            flipped_vol_list = flip(vol_list)
            self.assertIsInstance(flipped_vol_list, list)
            self.assertEqual(len(flipped_vol_list), 2)

            # check differentiability / gradient permutation check
            if backend == "torch":
                x = torch.tensor(base_np, requires_grad=True)
                y = flip(x)

                # Use a non-uniform weight field so we detect the permutation.
                w = torch.arange(H * W * C, dtype=x.dtype).reshape(H, W, C)

                loss = (y * w).sum()
                loss.backward()

                # y[i, j] = x[i, W-1-j]  => dloss/dx[i, k] = w[i, W-1-k]
                expected_grad = torch.flip(w, dims=[1])
                self.assertTrue(torch.equal(x.grad, expected_grad))

            # check "optics-like" geometry consistency with list inputs
            # (projection commutes with FlipLR for linear sum)
            project = self._ProjectSum()

            # Project list of volumes into an image
            img_before = project(vol_list)

            # Flip volumes then project
            img_after_1 = project(flip(vol_list))

            # Project then flip image
            img_after_2 = flip(img_before)

            if backend == "numpy":
                np.testing.assert_array_equal(img_after_1, img_after_2)
            else:
                self.assertTrue(torch.equal(img_after_1, img_after_2))


    def test_FlipUD(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, C = 3, 4, 1
            base_np = np.arange(H * W, dtype=np.float32).reshape(H, W, C)

            base = base_np if backend == "numpy" else torch.tensor(base_np)

            flip = augmentations.FlipUD(augment=True)

            # array correctness
            out = flip(base)

            if backend == "numpy":
                expected = base[::-1, :, :]
                np.testing.assert_array_equal(out, expected)
            else:
                expected = torch.flip(base, dims=[0])
                self.assertTrue(torch.equal(out, expected))

            # flip twice = identity
            out2 = flip(out)
            if backend == "numpy":
                np.testing.assert_array_equal(out2, base)
            else:
                self.assertTrue(torch.equal(out2, base))

            # scattered volume + position update
            position = np.array([1, 2], dtype=np.float32)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties=[{"position": position.copy()}],
            )

            flipped_volume = flip(volume)

            if backend == "numpy":
                expected_array = base[::-1, :, :]
                np.testing.assert_array_equal(flipped_volume.array, expected_array)
            else:
                expected_array = torch.flip(base, dims=[0])
                self.assertTrue(torch.equal(flipped_volume.array, expected_array))

            expected_y = (H - 1) - position[0]
            got_pos = flipped_volume.properties[0]["position"]

            self.assertEqual(float(got_pos[0]), float(expected_y))
            self.assertEqual(float(got_pos[1]), float(position[1]))

            # gradient check
            if backend == "torch":
                x = torch.tensor(base_np, requires_grad=True)
                y = flip(x)

                w = torch.arange(H * W * C, dtype=x.dtype).reshape(H, W, C)

                loss = (y * w).sum()
                loss.backward()

                expected_grad = torch.flip(w, dims=[0])
                self.assertTrue(torch.equal(x.grad, expected_grad))

            # projection commutes
            project = self._ProjectSum()

            vol_list = [volume, volume.copy()]

            img_before = project(vol_list)
            img_after_1 = project(flip(vol_list))
            img_after_2 = flip(img_before)

            if backend == "numpy":
                np.testing.assert_array_equal(img_after_1, img_after_2)
            else:
                self.assertTrue(torch.equal(img_after_1, img_after_2))


    def test_FlipDiagonal(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, C = 4, 4, 1   # must be square
            base_np = np.arange(H * W, dtype=np.float32).reshape(H, W, C)

            base = base_np if backend == "numpy" else torch.tensor(base_np)

            flip = augmentations.FlipDiagonal(augment=True)

            # array correctness
            out = flip(base)

            if backend == "numpy":
                expected = np.swapaxes(base, 0, 1)
                np.testing.assert_array_equal(out, expected)
            else:
                expected = base.transpose(0, 1)
                self.assertTrue(torch.equal(out, expected))

            # flip twice = identity
            out2 = flip(out)
            if backend == "numpy":
                np.testing.assert_array_equal(out2, base)
            else:
                self.assertTrue(torch.equal(out2, base))

            # scattered volume + position update
            position = np.array([1, 2], dtype=np.float32)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties=[{"position": position.copy()}],
            )

            flipped_volume = flip(volume)

            if backend == "numpy":
                expected_array = np.swapaxes(base, 0, 1)
                np.testing.assert_array_equal(flipped_volume.array, expected_array)
            else:
                expected_array = base.transpose(0, 1)
                self.assertTrue(torch.equal(flipped_volume.array, expected_array))

            got_pos = flipped_volume.properties[0]["position"]

            self.assertEqual(float(got_pos[0]), float(position[1]))
            self.assertEqual(float(got_pos[1]), float(position[0]))

            # gradient check
            if backend == "torch":
                x = torch.tensor(base_np, requires_grad=True)
                y = flip(x)

                w = torch.arange(H * W * C, dtype=x.dtype).reshape(H, W, C)

                loss = (y * w).sum()
                loss.backward()

                expected_grad = w.transpose(0, 1)
                self.assertTrue(torch.equal(x.grad, expected_grad))

            # projection commutes
            project = self._ProjectSum()

            vol_list = [volume, volume.copy()]

            img_before = project(vol_list)
            img_after_1 = project(flip(vol_list))
            img_after_2 = flip(img_before)

            if backend == "numpy":
                np.testing.assert_array_equal(img_after_1, img_after_2)
            else:
                self.assertTrue(torch.equal(img_after_1, img_after_2))


    def test_Affine(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, C = 32, 40, 1
            base_np = np.arange(H * W, dtype=np.float32).reshape(H, W, C)
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            affine = augmentations.Affine(
                scale=(1.05, 0.95),
                translate=(2.0, -3.0),
                rotate=0.2,
                shear=0.05,
            )

            # Array correctness
            out = affine(base)

            self.assertEqual(out.shape, base.shape)

            if backend == "numpy":
                self.assertFalse(np.isnan(out).any())
            else:
                self.assertFalse(torch.isnan(out).any())

            # ScatteredVolume + position update
            position = np.array([10.0, 15.0], dtype=np.float32)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={"position": position.copy()},
            )

            transformed_volume = affine(volume)

            # Array consistency
            if backend == "numpy":
                np.testing.assert_array_equal(
                    transformed_volume.array,
                    out,
                )
            else:
                self.assertTrue(torch.equal(
                    transformed_volume.array,
                    out,
                ))

            # Position update check
            forward = affine._last_affine["forward"]
            forward_offset = affine._last_affine["forward_offset"]

            if backend == "numpy":
                expected_pos = (forward @ position + forward_offset)
            else:
                pos_t = torch.tensor(position)
                expected_pos = (forward @ pos_t + forward_offset).detach()
            got_pos = transformed_volume.properties["position"]

            self.assertAlmostEqual(float(got_pos[0]), float(expected_pos[0]), places=4)
            self.assertAlmostEqual(float(got_pos[1]), float(expected_pos[1]), places=4)

            # List behavior
            arrays_list = [base, base]
            out_list = affine(arrays_list)
            self.assertIsInstance(out_list, list)
            self.assertEqual(len(out_list), 2)

            vol2 = volume.copy()
            vol_list = [volume, vol2]
            out_vol_list = affine(vol_list)
            self.assertIsInstance(out_vol_list, list)
            self.assertEqual(len(out_vol_list), 2)

            # Torch differentiability
            if backend == "torch":

                x = torch.tensor(base_np, requires_grad=True)
                y = affine(x)

                w = torch.arange(H * W * C, dtype=x.dtype).reshape(H, W, C)

                loss = (y * w).sum()
                loss.backward()

                self.assertIsNotNone(x.grad)
                self.assertFalse(torch.isnan(x.grad).any())

            # Optics-like linearity check
            project = self._ProjectSum()

            vol2 = volume.copy()
            vol_list = [volume, vol2]

            img_before = project(vol_list)
            img_after_1 = project(affine(vol_list))
            img_after_2 = affine(img_before)

            if backend == "numpy":
                np.testing.assert_allclose(img_after_1, img_after_2, atol=1e-5)
            else:
                self.assertTrue(torch.allclose(img_after_1, img_after_2, atol=1e-5))

            # Deterministic: Identity
            identity = augmentations.Affine(
                scale=(1.0, 1.0),
                translate=(0.0, 0.0),
                rotate=0.0,
                shear=0.0,
            )

            out_id = identity(base)

            if backend == "numpy":
                np.testing.assert_array_equal(out_id, base)
            else:
                self.assertTrue(torch.allclose(out_id, base, atol=1e-12))

            # Deterministic: Pure translation
            H = W = 64

            base_np = self.make_ellipse(
                H, W,
                cy=34,
                cx=28,
                ry=6,
                rx=10,
            )
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            shift_y = 7
            shift_x = -5

            translation = augmentations.Affine(
                scale=(1.0, 1.0),
                translate=(shift_x, shift_y),
                rotate=0.0,
                shear=0.0,
                order=0,
            )

            out_trans = translation(base)

            cy0, cx0 = self.center_of_mass(base)
            cy1, cx1 = self.center_of_mass(out_trans)
            self.assertAlmostEqual(cy1, cy0 + shift_y, places=4)
            self.assertAlmostEqual(cx1, cx0 + shift_x, places=4)

            # Deterministic position update (translation)
            position = np.array([5.0, 7.0], dtype=np.float32)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={"position": position.copy()},
            )

            translated_volume = translation(volume)
            got_pos = translated_volume.properties["position"]

            expected_pos = np.array([
                position[0] + shift_y,
                position[1] + shift_x,
            ], dtype=np.float32)

            self.assertAlmostEqual(float(got_pos[0]), float(expected_pos[0]), places=5)
            self.assertAlmostEqual(float(got_pos[1]), float(expected_pos[1]), places=5)

            # Deterministic: 90-degree rotation
            rot = augmentations.Affine(
                scale=1.0,
                translate=(0, 0),
                rotate=np.pi / 2,
                shear=0.0,
                order=0,
            )

            out_rot = rot(base)

            if backend == "numpy":
                expected = np.rot90(base, k=1, axes=(0, 1))
                self.assertTrue(np.array_equal(out_rot, expected))
            else:
                expected = torch.rot90(base, k=1, dims=(0, 1))
                self.assertTrue(torch.equal(out_rot, expected))

            # Deterministic: scaling
            scale = (0.75, 3.5)

            scaling = augmentations.Affine(
                scale=scale,
                translate=(0.0, 0.0),
                rotate=0.0,
                shear=0.0,
                order=1,
            )

            out_scaled = scaling(base)

            cy0, cx0 = self.center_of_mass(base)
            cy1, cx1 = self.center_of_mass(out_scaled)
            center_y = (H - 1) / 2
            center_x = (W - 1) / 2
            expected_cy = center_y + scale[1] * (cy0 - center_y)
            expected_cx = center_x + scale[0] * (cx0 - center_x)

            self.assertAlmostEqual(cy1, expected_cy, places=1)
            self.assertAlmostEqual(cx1, expected_cx, places=1)


    def test_ElasticTransformation(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, C = 64, 64, 3

            # Deterministic seed
            np.random.seed(0)
            if backend == "torch":
                torch.manual_seed(0)

            # Simple structured image (ellipse)
            base_np = self.make_ellipse(
                H, W,
                cy=32,
                cx=28,
                ry=12,
                rx=20,
            )

            base_np = np.repeat(base_np, C, axis=-1)

            base = base_np if backend == "numpy" else torch.tensor(base_np)

            # Identity test (alpha=0 should return identical image)
            elastic_identity = augmentations.ElasticTransformation(
                alpha=0.0,
                sigma=3,
                ignore_last_dim=True,
                order=1,
            )

            out_id = elastic_identity(base)

            if backend == "numpy":
                np.testing.assert_allclose(out_id, base, atol=1e-6)
            else:
                self.assertTrue(torch.allclose(out_id, base, atol=1e-6))

            # Deterministic reproducibility
            elastic = augmentations.ElasticTransformation(
                alpha=15,
                sigma=3,
                ignore_last_dim=True,
                order=1,
            )

            np.random.seed(42)
            if backend == "torch":
                torch.manual_seed(42)

            out_a = elastic(base)

            np.random.seed(42)
            if backend == "torch":
                torch.manual_seed(42)

            out_b = elastic(base)

            if backend == "numpy":
                np.testing.assert_allclose(out_a, out_b, atol=1e-6)
            else:
                self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6))

            # Basic sanity checks on output
            out = elastic(base)

            # Mean intensity should be approximately preserved
            if backend == "numpy":
                self.assertAlmostEqual(
                    float(out.mean()),
                    float(base.mean()),
                    places=2,
                )
            else:
                self.assertAlmostEqual(
                    float(out.mean().item()),
                    float(base.mean().item()),
                    places=2,
                )

            # Shape preserved
            self.assertEqual(out.shape, base.shape)

            # No NaNs or inf
            if backend == "numpy":
                self.assertFalse(np.isnan(out).any())
                self.assertFalse(np.isinf(out).any())
            else:
                self.assertFalse(torch.isnan(out).any())
                self.assertFalse(torch.isinf(out).any())

            # Non-trivial deformation
            if backend == "numpy":
                diff = np.mean(np.abs(out - base))
                self.assertGreater(diff, 1e-3)
            else:
                diff = torch.mean(torch.abs(out - base))
                self.assertGreater(diff.item(), 1e-3)

            # Channel consistency (ignore_last_dim=True)
            if backend == "numpy":
                self.assertTrue(np.allclose(out[..., 0], out[..., 1]))
                self.assertTrue(np.allclose(out[..., 1], out[..., 2]))
            else:
                self.assertTrue(torch.allclose(out[..., 0], out[..., 1]))
                self.assertTrue(torch.allclose(out[..., 1], out[..., 2]))

            # Differentiability (torch only)
            if backend == "torch":
                x = torch.tensor(base_np, requires_grad=True)
                y = elastic(x)

                loss = y.mean()
                loss.backward()

                self.assertIsNotNone(x.grad)
                self.assertFalse(torch.isnan(x.grad).any())

            # Test that ignore_last_dim=False produces different warps per channel
            base2_np = np.zeros((H, W, 2), dtype=np.float32)
            base2_np[..., 0] = self.make_ellipse(H, W, cy=32, cx=28, ry=12, rx=20)[..., 0]
            base2_np[..., 1] = self.make_ellipse(H, W, cy=20, cx=40, ry=8,  rx=10)[..., 0]
            base2 = base2_np if backend == "numpy" else torch.tensor(base2_np)

            # Same seed for both runs so randomness is comparable
            np.random.seed(123)
            if backend == "torch":
                torch.manual_seed(123)

            elastic_shared = augmentations.ElasticTransformation(
                alpha=15, sigma=3, ignore_last_dim=True, order=1
            )
            out_shared = elastic_shared(base2)

            np.random.seed(123)
            if backend == "torch":
                torch.manual_seed(123)

            elastic_indep = augmentations.ElasticTransformation(
                alpha=15, sigma=3, ignore_last_dim=False, order=1
            )
            out_indep = elastic_indep(base2)

            # The per-channel difference should change more with independent warps
            if backend == "numpy":
                d_shared = out_shared[..., 0] - out_shared[..., 1]
                d_indep = out_indep[..., 0] - out_indep[..., 1]
                self.assertGreater(np.mean(np.abs(d_indep - d_shared)), 1e-3)
            else:
                d_shared = out_shared[..., 0] - out_shared[..., 1]
                d_indep = out_indep[..., 0] - out_indep[..., 1]
                self.assertGreater(torch.mean(torch.abs(d_indep - d_shared)).item(), 1e-3)


    def test_Crop(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            # Pure array behaviour
            image_np = np.ones((10, 10, 10), dtype=np.float32)
            image = image_np if backend == "numpy" else torch.tensor(image_np)

            cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="remove")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (7, 8, 9))

            cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="retain")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (3, 2, 1))

            cropper = augmentations.Crop(crop=2, crop_mode="remove")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (8, 8, 8))

            cropper = augmentations.Crop(crop=2, crop_mode="retain")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (2, 2, 2))

            cropper = augmentations.Crop(crop=12, crop_mode="remove")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (1, 1, 1))

            cropper = augmentations.Crop(crop=0, crop_mode="retain")
            out = cropper(image)
            self.assertSequenceEqual(tuple(out.shape), (1, 1, 1))

            # ScatteredVolume geometry + metadata
            H, W, C = 20, 30, 1

            base_np = np.arange(H * W, dtype=np.float32).reshape(H, W, 1)
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            # Known geometry
            position = np.array([10, 15], dtype=float)  # (y, x)
            output_region = (0, 0, H, W)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={
                    "position": position.copy(),
                    "output_region": output_region,
                },
            )

            # Deterministic crop
            crop = augmentations.Crop(
                crop=(10, 12, None),   # retain shape in last dim
                crop_mode="retain",
                corner=(3, 5, 0),
            )

            cropped = crop(volume)

            # Array correctness
            expected = base_np[3:13, 5:17, :]

            if backend == "numpy":
                np.testing.assert_array_equal(cropped.array, expected)
            else:
                self.assertTrue(torch.equal(cropped.array, torch.tensor(expected)))

            # Position update (y, x)
            expected_pos = np.array([
                position[0] - 3,
                position[1] - 5
            ])

            got_pos = cropped.properties["position"]

            self.assertAlmostEqual(got_pos[0], expected_pos[0])
            self.assertAlmostEqual(got_pos[1], expected_pos[1])

            # output_region update
            # Convention: (ymin, xmin, ymax, xmax)
            ymin, xmin, ymax, xmax = output_region

            expected_region = (
                ymin + 3,
                xmin + 5,
                ymin + 3 + 10,
                xmin + 5 + 12,
            )

            self.assertEqual(
                cropped.properties["output_region"],
                expected_region,
            )

    def test_Crop_time_consistent_bind(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            # Deterministic base image
            img = np.arange(64 * 64).reshape(64, 64).astype(np.float32)

            if backend == "torch":
                img = torch.tensor(img)

            f1 = features.Value(img)

            stacked = f1 & f1
            crop = stacked >> augmentations.Crop(
                crop=32,
                crop_mode="retain",
                corner="random",
                time_consistent=True,
            )

            out1, out2 = crop.resolve()

            # Must be identical
            if backend == "torch":
                self.assertTrue(torch.equal(out1, out2))
            else:
                self.assertTrue(np.array_equal(out1, out2))

            # with source time_consistent is automatic because source is shared
            source = sources.Source(
                a=img
            )

            source = source.product(crop=[True])

            # Create features from the source:
            f1 = features.Value(source.a)

            stacked = f1 & f1
            crop = stacked >> augmentations.Crop(
                source.crop,
                crop=32,
                crop_mode="retain",
                corner="random",
            )

            out1, out2 = crop.resolve()

            # Must be identical
            if backend == "torch":
                self.assertTrue(torch.equal(out1, out2))
            else:
                self.assertTrue(np.array_equal(out1, out2))


    def test_CropToMultiplesOf(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, D = 11, 11, 11

            base_np = np.arange(H * W * D, dtype=np.float32).reshape(H, W, D)
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            position = np.array([5, 6], dtype=float)  # (y, x)
            output_region = (0, 0, H, W)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={
                    "position": position.copy(),
                    "output_region": output_region,
                },
            )

            # multiple = 2
            cropper = augmentations.CropToMultiplesOf(multiple=2, corner=(0, 0, 0))
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (10, 10, 10))

            # position unchanged if corner=(0,0,0)
            self.assertAlmostEqual(cropped.properties["position"][0], position[0])
            self.assertAlmostEqual(cropped.properties["position"][1], position[1])

            self.assertEqual(
                cropped.properties["output_region"],
                (0, 0, 10, 10),
            )

            # multiple = -1 (no crop)
            cropper = augmentations.CropToMultiplesOf(multiple=-1, corner=(0, 0, 0))
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (11, 11, 11))
            self.assertEqual(
                cropped.properties["output_region"],
                (0, 0, 11, 11),
            )

            # multiple per axis
            cropper = augmentations.CropToMultiplesOf(
                multiple=(2, 3, 5),
                corner=(0, 0, 0),
            )
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (10, 9, 10))
            self.assertEqual(
                cropped.properties["output_region"],
                (0, 0, 10, 9),
            )

            # skip one axis with -1
            cropper = augmentations.CropToMultiplesOf(
                multiple=(2, -1, 7),
                corner=(0, 0, 0),
            )
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (10, 11, 7))
            self.assertEqual(
                cropped.properties["output_region"],
                (0, 0, 10, 11),
            )

            # skip with None
            cropper = augmentations.CropToMultiplesOf(
                multiple=(2, 3, None),
                corner=(0, 0, 0),
            )
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (10, 9, 11))
            self.assertEqual(
                cropped.properties["output_region"],
                (0, 0, 10, 9),
            )

            # Corner shift test
            cropper = augmentations.CropToMultiplesOf(
                multiple=2,
                corner=(1, 2, 0),
            )
            cropped = cropper(volume)

            self.assertSequenceEqual(cropped.array.shape, (10, 10, 10))

            # Position must shift by corner
            effective_corner = (1 % 2, 2 % 2, 0 % 2)

            expected_pos = np.array([
                position[0] - effective_corner[0],
                position[1] - effective_corner[1],
            ])

            got_pos = cropped.properties["position"]

            self.assertAlmostEqual(got_pos[0], expected_pos[0])
            self.assertAlmostEqual(got_pos[1], expected_pos[1])

            self.assertEqual(
                cropped.properties["output_region"],
                (1, 0, 11, 10),
            )

    def test_CropTight(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, D = 20, 30, 10

            base_np = np.zeros((H, W, D), dtype=np.float32)

            # Insert solid block
            y0, y1 = 5, 15
            x0, x1 = 8, 22
            z0, z1 = 2, 7

            base_np[y0:y1, x0:x1, z0:z1] = 1.0

            base = base_np if backend == "numpy" else torch.tensor(base_np)

            position = np.array([10.0, 15.0])  # inside block
            output_region = (0, 0, H, W)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={
                    "position": position.copy(),
                    "output_region": output_region,
                },
            )

            crop = augmentations.CropTight(eps=1e-6)
            cropped = crop(volume)

            # Shape correctness
            expected_shape = (y1 - y0, x1 - x0, z1 - z0)

            self.assertSequenceEqual(
                cropped.array.shape,
                expected_shape,
            )

            # Array correctness
            expected_array = base_np[y0:y1, x0:x1, z0:z1]

            if backend == "numpy":
                np.testing.assert_array_equal(cropped.array, expected_array)
            else:
                self.assertTrue(
                    torch.equal(
                        cropped.array,
                        torch.tensor(expected_array),
                    )
                )

            # Position update
            expected_pos = np.array([
                position[0] - y0,
                position[1] - x0,
            ])

            got_pos = cropped.properties["position"]

            self.assertAlmostEqual(got_pos[0], expected_pos[0])
            self.assertAlmostEqual(got_pos[1], expected_pos[1])

            # Output region update
            # Convention: (ymin, xmin, ymax, xmax)
            expected_region = (
                output_region[0] + y0,
                output_region[1] + x0,
                output_region[0] + y1,
                output_region[1] + x1,
            )

            self.assertEqual(
                cropped.properties["output_region"],
                expected_region,
            )

            # No-op case (already tight)
            tight = crop(cropped)

            self.assertSequenceEqual(
                tight.array.shape,
                expected_shape,
            )

            # Torch differentiability
            if backend == "torch":
                x = torch.tensor(base_np, requires_grad=True)
                out = crop(x)

                loss = out.sum()
                loss.backward()

                self.assertIsNotNone(x.grad)
                self.assertFalse(torch.isnan(x.grad).any())


    def test_Pad(self):
        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            H, W, D = 10, 10, 10

            base_np = np.ones((H, W, D), dtype=np.float32)
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            # Shape correctness
            padder = augmentations.Pad(px=(2, 0, 2, 0, 0, 0), mode="constant")
            out = padder.update().resolve(base)
            self.assertSequenceEqual(out.shape, (12, 12, 10))

            padder = augmentations.Pad(px=(2, 2, 2, 0, 0, 0), mode="constant")
            out = padder.update().resolve(base)
            self.assertSequenceEqual(out.shape, (14, 12, 10))

            padder = augmentations.Pad(px=(2, 2, 2, 2, 0, 0), mode="constant")
            out = padder.update().resolve(base)
            self.assertSequenceEqual(out.shape, (14, 14, 10))

            padder = augmentations.Pad(px=(2, 2, 2, 2, 2, 0), mode="constant")
            out = padder.update().resolve(base)
            self.assertSequenceEqual(out.shape, (14, 14, 12))

            padder = augmentations.Pad(px=(2, 2, 2, 2, 2, 2), mode="constant")
            out = padder.update().resolve(base)
            self.assertSequenceEqual(out.shape, (14, 14, 14))

            # Interior must remain unchanged
            if backend == "numpy":
                interior = out[2:-2, 2:-2, 2:-2]
                np.testing.assert_array_equal(interior, base_np)
            else:
                interior = out[2:-2, 2:-2, 2:-2]
                self.assertTrue(torch.equal(interior, base))

            # Padding must contain cval
            if backend == "numpy":
                border_sum = np.sum(out) - np.sum(interior)
                self.assertEqual(border_sum, 0.0)
            else:
                border_sum = torch.sum(out) - torch.sum(interior)
                self.assertEqual(border_sum.item(), 0.0)

            # Non-symmetric padding
            padder = augmentations.Pad(px=(1, 3, 2, 4, 0, 0), mode="constant", cval=5)
            out = padder.update().resolve(base)

            self.assertSequenceEqual(out.shape, (H + 1 + 3, W + 2 + 4, D))

            # Check one known padded corner
            if backend == "numpy":
                self.assertEqual(out[0, 0, 0], 5)
            else:
                self.assertEqual(out[0, 0, 0].item(), 5)

            # Scatterer metadata update
            position = np.array([4.0, 5.0])
            output_region = (0, 0, H, W)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={
                    "position": position.copy(),
                    "output_region": output_region,
                },
            )

            padder = augmentations.Pad(px=(2, 0, 3, 0, 0, 0), mode="constant")

            padded = padder(volume)

            # Shape
            self.assertSequenceEqual(padded.array.shape, (H + 2 + 0, W + 3 + 0, D))

            # Position shifts with top/left padding
            expected_pos = np.array([
                position[0] + 2,
                position[1] + 3,
            ])

            got_pos = padded.properties["position"]

            self.assertAlmostEqual(got_pos[0], expected_pos[0])
            self.assertAlmostEqual(got_pos[1], expected_pos[1])

            # output_region must expand accordingly
            ymin, xmin, ymax, xmax = output_region

            expected_region = (
                ymin - 2,
                xmin - 3,
                ymax + 0,
                xmax + 0,
            )

            self.assertEqual(
                padded.properties["output_region"],
                expected_region,
            )

        
    def test_PadToMultiplesOf(self):
        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            # Simple array test
            image_np = np.ones((11, 13, 17), dtype=np.float32)
            image = image_np if backend == "numpy" else torch.tensor(image_np)

            padder = augmentations.PadToMultiplesOf(multiple=4, mode="constant")
            out = padder.update().resolve(image)

            # 11 → 12
            # 13 → 16
            # 17 → 20
            self.assertSequenceEqual(out.shape, (12, 16, 20))

            # Axis skipping
            padder = augmentations.PadToMultiplesOf(
                multiple=(4, -1, None),
                mode="constant",
            )
            out = padder.update().resolve(image)

            # only axis 0 padded
            self.assertSequenceEqual(out.shape, (12, 13, 17))

            # Scatterer test
            H, W = 11, 13
            base_np = np.zeros((H, W, 1), dtype=np.float32)
            base = base_np if backend == "numpy" else torch.tensor(base_np)

            position = np.array([5.0, 6.0])
            output_region = (0, 0, H, W)

            volume = scatterers.ScatteredVolume(
                array=base,
                properties={
                    "position": position.copy(),
                    "output_region": output_region,
                },
            )

            padder = augmentations.PadToMultiplesOf(
                multiple=4,
                mode="constant",
            )

            padded = padder(volume)

            # Shape check
            self.assertSequenceEqual(padded.array.shape, (12, 16, 4))

            # Compute expected padding (centered padding logic)
            pad_y = (-H) % 4
            pad_x = (-W) % 4

            pad_top = pad_y // 2
            pad_left = pad_x // 2

            # Position shift
            expected_pos = np.array([
                position[0] + pad_top,
                position[1] + pad_left,
            ])

            got_pos = padded.properties["position"]
            self.assertAlmostEqual(got_pos[0], expected_pos[0])
            self.assertAlmostEqual(got_pos[1], expected_pos[1])

            # output_region update
            ymin, xmin, ymax, xmax = output_region

            expected_region = (
                ymin - pad_top,
                xmin - pad_left,
                ymax + (pad_y - pad_top),
                xmax + (pad_x - pad_left),
            )

            self.assertEqual(
                padded.properties["output_region"],
                expected_region,
            )

if __name__ == "__main__":
    unittest.main()


