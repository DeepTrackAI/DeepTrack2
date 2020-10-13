import sys

sys.path.append(".")  # Adds the module to path

import unittest
import deeptrack as dt
import deeptrack.augmentations as augmentations

from deeptrack.features import Feature
import numpy as np


class TestAugmentations(unittest.TestCase):
    class DummyFeature(Feature):
        __distributed__ = False

        def get(self, image, **kwargs):
            output = np.array([[[1], [2]], [[0], [0]]])
            return output

    def test_FlipLR(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipLR(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[2], [1]], [[0], [0]]])))

    def test_FlipUD(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipUD(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[0], [0]], [[1], [2]]])))

    def test_FlipDiagonal(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipDiagonal(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[1], [0]], [[2], [0]]])))

    def test_Affine(self):
        optics = dt.Fluorescence(magnification=10)
        particle = dt.PointParticle(
            position=lambda image_size: np.random.rand(2) * image_size[-2:],
            image_size=optics.output_region,
        )

        augmentation = augmentations.Affine(
            scale=lambda: 0.25 + np.random.rand(2) * 0.25,
            rotation=lambda: np.random.rand() * np.pi * 2,
            shear=lambda: np.random.rand() * np.pi / 2 - np.pi / 4,
            translate=lambda: np.random.rand(2) * 20 - 10,
            mode="constant",
        )

        pipe = optics(particle) + augmentation

        for _ in range(10):
            image = pipe.update().resolve()
            pmax = np.unravel_index(
                np.argmax(image[:, :, 0], axis=None), shape=image[:, :, 0].shape
            )

            dist = np.sum(np.abs(np.array(image.get_property("position")) - pmax))

            self.assertLess(dist, 3)

    def test_ElasticTransformation(self):
        # 3D input
        im = np.zeros((10, 8, 2))
        transformer = augmentations.ElasticTransformation(
            alpha=20,
            sigma=2,
            ignore_last_dim=True,
            order=1,
            mode="reflect",
        )

        im[:, :, 0] = 1

        out_1 = transformer.update().resolve(im)
        self.assertIsNone(np.testing.assert_allclose(out_1, im))

        im[:, :, :] = 0
        im[0, :, :] = 1
        out_2 = transformer.update().resolve(im)
        self.assertIsNone(np.testing.assert_allclose(out_2[:, :, 0], out_2[:, :, 1]))

        out_3 = transformer.update(ignore_last_dim=False).resolve(im)
        self.assertRaises(
            AssertionError,
            lambda: np.testing.assert_allclose(out_3[:, :, 0], out_3[:, :, 1]),
        )

        # 2D input
        im = np.zeros((10, 8))
        transformer = augmentations.ElasticTransformation(
            alpha=20,
            sigma=2,
            ignore_last_dim=False,
            order=1,
            mode="reflect",
        )

        out_1 = transformer.update().resolve(im)

    def test_Crop(self):
        image = np.ones((10, 10, 10))

        cropper = dt.Crop(crop=(3, 2, 1), crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (7, 8, 9))

        cropper = dt.Crop(crop=(3, 2, 1), crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (3, 2, 1))

        cropper = dt.Crop(crop=2, crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (8, 8, 8))

        cropper = dt.Crop(crop=2, crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (2, 2, 2))

        cropper = dt.Crop(crop=12, crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (1, 1, 1))

        cropper = dt.Crop(crop=0, crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (1, 1, 1))

    def test_CropToMultiple(self):
        image = np.ones((11, 11, 11))

        cropper = dt.CropToMultiplesOf(multiple=2)
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 10, 10))

        cropper = dt.CropToMultiplesOf(multiple=-1)
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (11, 11, 11))

        cropper = dt.CropToMultiplesOf(multiple=(2, 3, 5))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 9, 10))

        cropper = dt.CropToMultiplesOf(multiple=(2, -1, 7))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 11, 7))

        cropper = dt.CropToMultiplesOf(multiple=(2, 3, None))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 9, 11))


if __name__ == "__main__":
    unittest.main()