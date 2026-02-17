import sys

# sys.path.append(".")  # Adds the module to path

import unittest

# raise unittest.SkipTest("Temporarily skipped")

import numpy as np

from deeptrack import (
    augmentations,
    config,
    features,
    scatterers,
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

            # check "pptics-like" geometry consistency with list inputs
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
                properties=[{"position": position.copy()}],
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

            # Position update check (explicit forward equation)
            mapping = affine._last_affine["mapping"]
            offset = affine._last_affine["offset"]

            if backend == "numpy":
                inv_mapping = np.linalg.inv(mapping)
                expected_pos = (inv_mapping @ (position - offset).T).T
            else:
                inv_mapping = torch.linalg.inv(mapping)
                pos_t = torch.tensor(position)
                expected_pos = (inv_mapping @ (pos_t - offset)).detach()

            got_pos = transformed_volume.properties[0]["position"]

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
                self.assertLess(torch.max(torch.abs(out_id - base)), 5e-5)




if __name__ == "__main__":
    unittest.main()







    # def test_Affine(self):
    #     opt = optics.Fluorescence(magnification=10)
    #     particle = scatterers.PointParticle(
    #         position=lambda image_size: np.random.rand(2) * image_size[-2:],
    #         image_size=opt.output_region,
    #     )

    #     augmentation = augmentations.Affine(
    #         scale=lambda: 0.25 + np.random.rand(2) * 0.25,
    #         rotation=lambda: np.random.rand() * np.pi * 2,
    #         shear=lambda: np.random.rand() * np.pi / 2 - np.pi / 4,
    #         translate=lambda: np.random.rand(2) * 20 - 10,
    #         mode="constant",
    #     )

    #     pipe = opt(particle) >> augmentation
    #     pipe.store_properties(True)

    #     for _ in range(10):
    #         image = pipe.update().resolve()
    #         pmax = np.unravel_index(
    #             np.argmax(image[:, :, 0], axis=None),
    #             shape=image[:, :, 0].shape
    #         )

    #         dist = np.sum(
    #             np.abs(np.array(image.get_property("position"))- pmax)
    #         )

    #         self.assertLess(dist, 3)

    # def test_ElasticTransformation(self):
    #     np.random.seed(1000)
    #     import random
    #     random.seed(1000)
    #     # 3D input
        
    #     im = np.zeros((10, 8, 2))
    #     transformer = augmentations.ElasticTransformation(
    #         alpha=20,
    #         sigma=2,
    #         ignore_last_dim=True,
    #         order=1,
    #         mode="reflect",
    #     )

    #     im[:, :, 0] = 1

    #     out_1 = transformer.update().resolve(im)
    #     self.assertIsNone(np.testing.assert_allclose(out_1, im))

    #     im[:, :, :] = 0
    #     im[0, :, :] = 1
    #     out_2 = transformer.update().resolve(im)
    #     self.assertIsNone(
    #         np.testing.assert_allclose(out_2[:, :, 0], out_2[:, :, 1])
    #     )

    #     transformer.ignore_last_dim.set_value(False)
    #     out_3 = transformer.resolve(im)
    #     self.assertRaises(
    #         AssertionError,
    #         lambda: np.testing.assert_allclose(out_3[:, :, 0], out_3[:, :, 1]),
    #     )

    #     # 2D input
    #     im = np.zeros((10, 8))
    #     transformer = augmentations.ElasticTransformation(
    #         alpha=20,
    #         sigma=2,
    #         ignore_last_dim=False,
    #         order=1,
    #         mode="reflect",
    #     )

    #     out_1 = transformer.update().resolve(im)

    # def test_Crop(self):
    #     image = np.ones((10, 10, 10))

    #     cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="remove")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (7, 8, 9))

    #     cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="retain")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (3, 2, 1))

    #     cropper = augmentations.Crop(crop=2, crop_mode="remove")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (8, 8, 8))

    #     cropper = augmentations.Crop(crop=2, crop_mode="retain")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (2, 2, 2))

    #     cropper = augmentations.Crop(crop=12, crop_mode="remove")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (1, 1, 1))

    #     cropper = augmentations.Crop(crop=0, crop_mode="retain")
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (1, 1, 1))

    # def test_CropToMultiple(self):
    #     image = np.ones((11, 11, 11))

    #     cropper = augmentations.CropToMultiplesOf(multiple=2)
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (10, 10, 10))

    #     cropper = augmentations.CropToMultiplesOf(multiple=-1)
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (11, 11, 11))

    #     cropper = augmentations.CropToMultiplesOf(multiple=(2, 3, 5))
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (10, 9, 10))

    #     cropper = augmentations.CropToMultiplesOf(multiple=(2, -1, 7))
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (10, 11, 7))

    #     cropper = augmentations.CropToMultiplesOf(multiple=(2, 3, None))
    #     out = cropper.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (10, 9, 11))
    
    # def test_Pad(self):
    #     image = np.ones((10, 10, 10))

    #     padder = augmentations.Pad(px=(2, 0, 2, 0, 0, 0), mode="constant")
    #     out = padder.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (12, 12, 10))

    #     padder = augmentations.Pad(px=(2, 2, 2, 0, 0, 0), mode="constant")
    #     out = padder.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (14, 12, 10))

    #     padder = augmentations.Pad(px=(2, 2, 2, 2, 0, 0), mode="constant")
    #     out = padder.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (14, 14, 10))

    #     padder = augmentations.Pad(px=(2, 2, 2, 2, 2, 0), mode="constant")
    #     out = padder.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (14, 14, 12))

    #     padder = augmentations.Pad(px=(2, 2, 2, 2, 2, 2), mode="constant")
    #     out = padder.update().resolve(image)
    #     self.assertSequenceEqual(out.shape, (14, 14, 14))

# if __name__ == "__main__":
#     unittest.main()