# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import itertools
import operator
import unittest

import numpy as np

from deeptrack import features, image


class TestImage(unittest.TestCase):

    class Particle(features.Feature):
        def get(self, image, position=None, **kwargs):
            # Code for simulating a particle not included
            return image

    _test_cases = [
        np.zeros((3, 1)),
        np.ones((3, 1)),
        np.random.randn(3, 1),
        [1, 2, 3],
        -1,
        0,
        1,
        1 / 2,
        -0.5,
        True,
        False,
        1j,
        1 + 1j,
    ]

    def _test_binary_method(self, op):

        for a, b in itertools.product(self._test_cases, self._test_cases):
            a = np.array(a)
            b = np.array(b)
            try:
                try:
                    op(a, b)
                except (TypeError, ValueError):
                    continue
                A = image.Image(a)
                A.append({"name": "a"})
                B = image.Image(b)
                B.append({"name": "b"})

                true_out = op(a, b)

                out = op(A, b)
                self.assertIsInstance(out, (image.Image, tuple))
                np.testing.assert_array_almost_equal(np.array(out),
                                                     np.array(true_out))
                if isinstance(out, image.Image):
                    self.assertIn(A.properties[0], out.properties)
                    self.assertNotIn(B.properties[0], out.properties)

                out = op(A, B)
                self.assertIsInstance(out, (image.Image, tuple))
                np.testing.assert_array_almost_equal(np.array(out),
                                                     np.array(true_out))
                if isinstance(out, image.Image):
                    self.assertIn(A.properties[0], out.properties)
                    self.assertIn(B.properties[0], out.properties)
            except AssertionError:
                raise AssertionError(
                    f"Received the obove error when evaluating {op.__name__} "
                    f"between {a} and {b}"
                )

    def _test_reflected_method(self, op):

        for a, b in itertools.product(self._test_cases, self._test_cases):
            a = np.array(a)
            b = np.array(b)

            try:
                op(a, b)
            except (TypeError, ValueError):
                continue

            A = image.Image(a)
            A.append({"name": "a"})
            B = image.Image(b)
            B.append({"name": "b"})

            true_out = op(a, b)

            out = op(a, B)
            self.assertIsInstance(out, (image.Image, tuple))
            np.testing.assert_array_almost_equal(np.array(out),
                                                 np.array(true_out))
            if isinstance(out, image.Image):
                self.assertNotIn(A.properties[0], out.properties)
                self.assertIn(B.properties[0], out.properties)

    def _test_inplace_method(self, op):

        for a, b in itertools.product(self._test_cases, self._test_cases):
            a = np.array(a)
            b = np.array(b)

            try:
                op(a, b)
            except (TypeError, ValueError):
                continue
            A = image.Image(a)
            A.append({"name": "a"})
            B = image.Image(b)
            B.append({"name": "b"})

            op(a, b)

            self.assertIsNot(a, A._value)
            self.assertIsNot(b, B._value)

            op(A, B)
            self.assertIsInstance(A, (image.Image, tuple))
            np.testing.assert_array_almost_equal(np.array(A), np.array(a))

            self.assertIn(A.properties[0], A.properties)
            self.assertNotIn(B.properties[0], A.properties)


    def test_Image(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        self.assertIsInstance(output_image, image.Image)


    def test_Image_properties(self):
        # Check the property attribute.

        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        properties = output_image.properties
        self.assertIsInstance(properties, list)
        self.assertIsInstance(properties[0], dict)
        self.assertEqual(properties[0]["position"], (128, 128))
        self.assertEqual(properties[0]["name"], "Particle")


    def test_Image_not_store(self):
        # Check that without particle.store_properties(),
        # it returns a numoy array.

        particle = self.Particle(position=(128, 128))
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)


    def test_Image__lt__(self):
        self._test_binary_method(operator.lt)


    def test_Image__le__(self):
        self._test_binary_method(operator.gt)


    def test_Image__eq__(self):
        self._test_binary_method(operator.eq)


    def test_Image__ne__(self):
        self._test_binary_method(operator.ne)


    def test_Image__gt__(self):
        self._test_binary_method(operator.gt)


    def test_Image__ge__(self):
        self._test_binary_method(operator.ge)


    def test_Image__add__(self):
        self._test_binary_method(operator.add)
        self._test_reflected_method(operator.add)
        self._test_inplace_method(operator.add)


    def test_Image__sub__(self):
        self._test_binary_method(operator.sub)
        self._test_reflected_method(operator.sub)
        self._test_inplace_method(operator.sub)


    def test_Image__mul__(self):
        self._test_binary_method(operator.mul)
        self._test_reflected_method(operator.mul)
        self._test_inplace_method(operator.mul)


    def test_Image__matmul__(self):
        self._test_binary_method(operator.matmul)
        self._test_reflected_method(operator.matmul)
        self._test_inplace_method(operator.matmul)


    def test_Image__truediv__(self):
        self._test_binary_method(operator.truediv)
        self._test_reflected_method(operator.truediv)
        self._test_inplace_method(operator.truediv)


    def test_Image__floordiv__(self):
        self._test_binary_method(operator.floordiv)
        self._test_reflected_method(operator.floordiv)
        self._test_inplace_method(operator.floordiv)


    def test_Image__mod__(self):
        self._test_binary_method(operator.mod)
        self._test_reflected_method(operator.mod)
        self._test_inplace_method(operator.mod)


    def test_Image__divmod__(self):
        self._test_binary_method(divmod)
        self._test_reflected_method(divmod)


    def test_Image__pow__(self):
        self._test_binary_method(operator.pow)
        self._test_reflected_method(operator.pow)
        self._test_inplace_method(operator.pow)


    def test_lshift(self):
        self._test_binary_method(operator.lshift)
        self._test_reflected_method(operator.lshift)
        self._test_inplace_method(operator.lshift)


    def test_Image__rshift__(self):
        self._test_binary_method(operator.rshift)
        self._test_reflected_method(operator.rshift)
        self._test_inplace_method(operator.rshift)


    def test_Image___array___from_constant(self):
        a = image.Image(1)
        self.assertIsInstance(a, image.Image)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)


    def test_Image___array___from_list_of_constants(self):
        a = [image.Image(1), image.Image(2)]

        self.assertIsInstance(image.Image(a)._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.shape, (2,))


    def test_Image___array___from_array(self):
        a = image.Image(np.zeros((2, 2)))

        self.assertIsInstance(a._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.shape, (2, 2))


    def test_Image___array___from_list_of_array(self):
        a = [image.Image(np.zeros((2, 2))), image.Image(np.ones((2, 2)))]

        self.assertIsInstance(image.Image(a)._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 3)
        self.assertEqual(a.shape, (2, 2, 2))


    def test_Image_append(self):

        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        properties = output_image.properties
        self.assertEqual(properties[0]["position"], (128, 128))
        self.assertEqual(properties[0]["name"], "Particle")

        property_dict = {"key1": 1, "key2": 2}
        output_image.append(property_dict)
        properties = output_image.properties
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


    def test_Image_get_property(self):

        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)

        property_position = output_image.get_property("position")
        self.assertEqual(property_position, (128, 128))

        property_name = output_image.get_property("name")
        self.assertEqual(property_name, "Particle")


    def test_Image_merge_properties_from(self):

        # With `other` containing an Image.
        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image1 = particle.resolve(input_image)
        output_image2 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image2)
        self.assertEqual(len(output_image1.properties), 1)

        particle.update()
        output_image3 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image3)
        self.assertEqual(len(output_image1.properties), 2)

        # With `other` containing a numpy array.
        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        output_image.merge_properties_from(np.zeros((10, 10)))
        self.assertEqual(len(output_image.properties), 1)

        # With `other` containing a list.
        particle = self.Particle(position=(128, 128))
        particle.store_properties()  # To return an Image and not an array.
        input_image = image.Image(np.zeros((256, 256)))
        output_image1 = particle.resolve(input_image)
        output_image2 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image2)
        self.assertEqual(len(output_image1.properties), 1)

        particle.update()
        output_image3 = particle.resolve(input_image)
        particle.update()
        output_image4 = particle.resolve(input_image)
        output_image1.merge_properties_from(
            [
                np.zeros((10, 10)), output_image3, np.zeros((10, 10)),
                output_image1, np.zeros((10, 10)), output_image4,
                np.zeros((10, 10)), output_image2, np.zeros((10, 10)),
            ]
        )
        self.assertEqual(len(output_image1.properties), 3)


    def test_Image__view(self):

        for value in self._test_cases:
            im = image.Image(value)
            np.testing.assert_array_equal(im._view(value),
                                          np.array(value))

            im_nested = image.Image(im)
            np.testing.assert_array_equal(im_nested._view(value),
                                          np.array(value))


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