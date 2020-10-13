import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.image as image

from deeptrack.features import Feature
import numpy as np


class TestImage(unittest.TestCase):
    class Particle(Feature):
        def get(self, image, position=None, **kwargs):
            # Code for simulating a particle not included
            return image

    def test_Image(self):
        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        self.assertIsInstance(output_image, image.Image)

    def test_Image_properties(self):
        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        properties = output_image.properties
        self.assertIsInstance(properties, list)
        self.assertIsInstance(properties[0], dict)
        self.assertEqual(properties[0]["position"], (128, 128))
        self.assertEqual(properties[0]["name"], "Particle")

    def test_Image_get_property(self):
        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)

        property_position = output_image.get_property("position")
        self.assertEqual(property_position, (128, 128))

        property_name = output_image.get_property("name")
        self.assertEqual(property_name, "Particle")

        property_hash_key = output_image.get_property("hash_key")
        self.assertIsInstance(property_hash_key, list)

    def test_Image_append(self):
        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        properties = output_image.properties

        property_dict = {"key1": 1, "key2": 2}
        output_image.append(property_dict)
        self.assertEqual(properties[0]["position"], (128, 128))
        self.assertEqual(properties[0]["name"], "Particle")
        self.assertEqual(properties[1]["key1"], 1)
        self.assertEqual(output_image.get_property("key1"), 1)
        self.assertEqual(properties[1]["key2"], 2)
        self.assertEqual(output_image.get_property("key2"), 2)

        property_dict2 = {"key1": 11, "key2": 22}
        output_image.append(property_dict2)
        self.assertEqual(output_image.get_property("key1"), 1)
        self.assertEqual(output_image.get_property("key1", get_one=False), [1, 11])

    def test_Image_merge_properties_from(self):
        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image1 = particle.resolve(input_image)
        output_image2 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image2)
        self.assertEqual(len(output_image1.properties), 1)

        particle.update()
        output_image3 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image3)
        self.assertEqual(len(output_image1.properties), 2)

    def test_pad_image_to_fft(self):
        input_image = image.Image(np.zeros((7, 25)))
        padded_image = image.pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (8, 27))

        input_image = image.Image(np.zeros((30, 27)))
        padded_image = image.pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (32, 27))

        input_image = image.Image(np.zeros((300, 400)))
        padded_image = image.pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (324, 432))


if __name__ == "__main__":
    unittest.main()