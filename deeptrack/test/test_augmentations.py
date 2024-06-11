import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import augmentations, optics, scatterers

from ..features import Feature
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
        opt = optics.Fluorescence(magnification=10)
        particle = scatterers.PointParticle(
            position=lambda image_size: np.random.rand(2) * image_size[-2:],
            image_size=opt.output_region,
        )

        augmentation = augmentations.Affine(
            scale=lambda: 0.25 + np.random.rand(2) * 0.25,
            rotation=lambda: np.random.rand() * np.pi * 2,
            shear=lambda: np.random.rand() * np.pi / 2 - np.pi / 4,
            translate=lambda: np.random.rand(2) * 20 - 10,
            mode="constant",
        )

        pipe = opt(particle) >> augmentation
        pipe.store_properties(True)

        for _ in range(10):
            image = pipe.update().resolve()
            pmax = np.unravel_index(
                np.argmax(image[:, :, 0], axis=None), shape=image[:, :, 0].shape
            )

            dist = np.sum(np.abs(np.array(image.get_property("position")) - pmax))

            self.assertLess(dist, 3)

    def test_ElasticTransformation(self):
        np.random.seed(1000)
        import random
        random.seed(1000)
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

        transformer.ignore_last_dim.set_value(False)
        out_3 = transformer.resolve(im)
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

        cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (7, 8, 9))

        cropper = augmentations.Crop(crop=(3, 2, 1), crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (3, 2, 1))

        cropper = augmentations.Crop(crop=2, crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (8, 8, 8))

        cropper = augmentations.Crop(crop=2, crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (2, 2, 2))

        cropper = augmentations.Crop(crop=12, crop_mode="remove")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (1, 1, 1))

        cropper = augmentations.Crop(crop=0, crop_mode="retain")
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (1, 1, 1))

    def test_CropToMultiple(self):
        image = np.ones((11, 11, 11))

        cropper = augmentations.CropToMultiplesOf(multiple=2)
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 10, 10))

        cropper = augmentations.CropToMultiplesOf(multiple=-1)
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (11, 11, 11))

        cropper = augmentations.CropToMultiplesOf(multiple=(2, 3, 5))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 9, 10))

        cropper = augmentations.CropToMultiplesOf(multiple=(2, -1, 7))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 11, 7))

        cropper = augmentations.CropToMultiplesOf(multiple=(2, 3, None))
        out = cropper.update().resolve(image)
        self.assertSequenceEqual(out.shape, (10, 9, 11))


if __name__ == "__main__":
    unittest.main()