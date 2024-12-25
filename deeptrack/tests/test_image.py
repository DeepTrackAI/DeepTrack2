import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import Image, pad_image_to_fft

from ..features import Feature
import numpy as np
import itertools
import operator as ops

TF_BINDINGS_AVAILABLE = True
try:
    from ..backend import tensorflow_bindings
    tensorflow_bindings.implements_tf
except ImportError:
    TF_BINDINGS_AVAILABLE = False


class TestImage(unittest.TestCase):
    class Particle(Feature):
        def get(self, image, position=None, **kwargs):
            # Code for simulating a particle not included
            return image

    _test_cases = [
        -1,
        0,
        1,
        1 / 2,
        -0.5,
        np.zeros((3, 1)),
        np.ones((3, 1)),
        np.random.randn(3, 1),
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
                A = Image(a)
                A.append({"name": "a"})
                B = Image(b)
                B.append({"name": "b"})

                true_out = op(a, b)

                out = op(A, b)
                self.assertIsInstance(out, (Image, tuple))
                np.testing.assert_array_almost_equal(np.array(out), np.array(true_out))
                if isinstance(out, Image):
                    self.assertIn(A.properties[0], out.properties)
                    self.assertNotIn(B.properties[0], out.properties)

                out = op(A, B)
                self.assertIsInstance(out, (Image, tuple))
                np.testing.assert_array_almost_equal(np.array(out), np.array(true_out))
                if isinstance(out, Image):
                    self.assertIn(A.properties[0], out.properties)
                    self.assertIn(B.properties[0], out.properties)
            except AssertionError as e:

                raise AssertionError(
                    "Received the obove error when evaluating {} between {} and {}".format(
                        op.__name__, a, b
                    )
                )

    def _test_reflected_method(self, op):

        for a, b in itertools.product(self._test_cases, self._test_cases):
            a = np.array(a)
            b = np.array(b)

            try:
                op(a, b)
            except (TypeError, ValueError):
                continue

            A = Image(a)
            A.append({"name": "a"})
            B = Image(b)
            B.append({"name": "b"})

            true_out = op(a, b)

            out = op(a, B)
            self.assertIsInstance(out, (Image, tuple))
            np.testing.assert_array_almost_equal(np.array(out), np.array(true_out))
            if isinstance(out, Image):
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
            A = Image(a)
            A.append({"name": "a"})
            B = Image(b)
            B.append({"name": "b"})

            op(a, b)

            self.assertIsNot(a, A._value)
            self.assertIsNot(b, B._value)

            op(A, B)
            self.assertIsInstance(A, (Image, tuple))
            np.testing.assert_array_almost_equal(np.array(A), np.array(a))

            self.assertIn(A.properties[0], A.properties)
            self.assertNotIn(B.properties[0], A.properties)

    def test_lt(self):
        self._test_binary_method(ops.lt)

    def test_le(self):
        self._test_binary_method(ops.gt)

    def test_eq(self):
        self._test_binary_method(ops.eq)

    def test_gt(self):
        self._test_binary_method(ops.gt)

    def test_ge(self):
        self._test_binary_method(ops.ge)

    def test_add(self):
        self._test_binary_method(ops.add)
        self._test_reflected_method(ops.add)
        self._test_inplace_method(ops.add)

    def test_sub(self):
        self._test_binary_method(ops.sub)
        self._test_reflected_method(ops.sub)
        self._test_inplace_method(ops.sub)

    def test_mul(self):
        self._test_binary_method(ops.mul)
        self._test_reflected_method(ops.mul)
        self._test_inplace_method(ops.mul)

    def test_matmul(self):
        self._test_binary_method(ops.matmul)
        self._test_reflected_method(ops.matmul)
        self._test_inplace_method(ops.matmul)

    def test_truediv(self):
        self._test_binary_method(ops.truediv)
        self._test_reflected_method(ops.truediv)
        self._test_inplace_method(ops.truediv)

    def test_floordiv(self):
        self._test_binary_method(ops.floordiv)
        self._test_reflected_method(ops.floordiv)
        self._test_inplace_method(ops.floordiv)

    def test_truediv(self):
        self._test_binary_method(ops.mod)
        self._test_reflected_method(ops.mod)
        self._test_inplace_method(ops.mod)

    def test_truediv(self):
        self._test_binary_method(divmod)
        self._test_reflected_method(divmod)

    def test_pow(self):
        self._test_binary_method(ops.pow)
        self._test_reflected_method(ops.pow)
        self._test_inplace_method(ops.pow)

    def test_rshift(self):
        self._test_binary_method(ops.rshift)
        self._test_reflected_method(ops.rshift)
        self._test_inplace_method(ops.rshift)

    def test_lshift(self):
        self._test_binary_method(ops.lshift)
        self._test_reflected_method(ops.lshift)
        self._test_inplace_method(ops.lshift)

    def test_array_from_constant(self):
        a = Image(1)
        self.assertIsInstance(a, Image)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a, 1)

    def test_array_from_list_of_constants(self):
        a = [Image(1), Image(2)]

        self.assertIsInstance(Image(a)._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.shape, (2,))

    def test_array_from_array(self):
        a = Image(np.zeros((2, 2)))

        self.assertIsInstance(a._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.shape, (2, 2))

    def test_array_from_list_of_array(self):
        a = [Image(np.zeros((2, 2))), Image(np.ones((2, 2)))]

        self.assertIsInstance(Image(a)._value, np.ndarray)
        a = np.array(a)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(a.ndim, 3)
        self.assertEqual(a.shape, (2, 2, 2))

    def test_square_Tensor(self):

        if not TF_BINDINGS_AVAILABLE:
            return
        import tensorflow as tf
        a = tf.constant([-1, 0, 2])
        A = Image(a)
        A.append({"name": a})
        self.assertIsInstance(A._value, tf.Tensor)

        B = np.square(A)

        self.assertIsInstance(B, Image)
        self.assertIsInstance(B._value, tf.Tensor)
        np.testing.assert_array_almost_equal(B.numpy(), np.array([1, 0, 4]))

    def test_reducer_Tensor(self):
        if not TF_BINDINGS_AVAILABLE:
            return
        import tensorflow as tf
        a = tf.constant([[-1, 1], [4, 4]])
        A = Image(a)
        A.append({"name": a})
        self.assertIsInstance(A._value, tf.Tensor)

        B = np.sum(A, axis=0, keepdims=True)

        self.assertIsInstance(B, Image)
        self.assertIsInstance(B._value, tf.Tensor)
        np.testing.assert_array_almost_equal(B.numpy(), np.array([[3, 5]]))

    def test_Image(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        self.assertIsInstance(output_image, Image)

    def test_Image_not_store(self):
        particle = self.Particle(position=(128, 128))
        input_image = Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)

    def test_Image_properties(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)
        properties = output_image.properties
        self.assertIsInstance(properties, list)
        self.assertIsInstance(properties[0], dict)
        self.assertEqual(properties[0]["position"], (128, 128))
        self.assertEqual(properties[0]["name"], "Particle")

    def test_Image_get_property(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)

        property_position = output_image.get_property("position")
        self.assertEqual(property_position, (128, 128))

        property_name = output_image.get_property("name")
        self.assertEqual(property_name, "Particle")

    def test_Image_append(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = Image(np.zeros((256, 256)))
        output_image = particle.resolve(input_image)

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

    def test_Image_merge_properties_from(self):
        particle = self.Particle(position=(128, 128))
        particle.store_properties()
        input_image = Image(np.zeros((256, 256)))
        output_image1 = particle.resolve(input_image)
        output_image2 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image2)
        self.assertEqual(len(output_image1.properties), 1)

        particle.update()
        output_image3 = particle.resolve(input_image)
        output_image1.merge_properties_from(output_image3)
        self.assertEqual(len(output_image1.properties), 2)

    def test_pad_image_to_fft(self):
        input_image = Image(np.zeros((7, 25)))
        padded_image = pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (8, 27))

        input_image = Image(np.zeros((30, 27)))
        padded_image = pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (32, 27))

        input_image = Image(np.zeros((300, 400)))
        padded_image = pad_image_to_fft(input_image)
        self.assertEqual(padded_image.shape, (324, 432))


if __name__ == "__main__":
    unittest.main()