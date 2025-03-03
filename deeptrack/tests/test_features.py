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

from deeptrack import features, properties, scatterers, units, optics
from deeptrack.image import Image
from deeptrack.noises import Gaussian

def grid_test_features(
    tester,
    feature_a,
    feature_b,
    feature_a_inputs,
    feature_b_inputs,
    expected_result_function,
    merge_operator=operator.rshift,
):

    assert callable(feature_a), "First feature constructor needs to be callable"
    assert callable(feature_b), "Second feature constructor needs to be callable"
    assert (
        len(feature_a_inputs) > 0 and len(feature_b_inputs) > 0
    ), "Feature input-lists cannot be empty"
    assert callable(expected_result_function), "Result function needs to be callable"

    for f_a_input, f_b_input in itertools.product(feature_a_inputs, feature_b_inputs):

        f_a = feature_a(**f_a_input)
        f_b = feature_b(**f_b_input)
        f = merge_operator(f_a, f_b)
        f.store_properties()

        tester.assertIsInstance(f, features.Feature)

        try:
            output = f()
        except Exception as e:
            tester.assertRaises(
                type(e),
                lambda: expected_result_function(f_a.properties(), f_b.properties()),
            )
            continue

        expected_result = expected_result_function(
            f_a.properties(),
            f_b.properties(),
        )

        if isinstance(output, list) and isinstance(expected_result, list):
            [np.testing.assert_almost_equal(np.array(a), np.array(b))
             for a, b in zip(output, expected_result)]

        else:
            is_equal = np.array_equal(
                np.array(output), np.array(expected_result), equal_nan=True
            )

            tester.assertFalse(
                not is_equal,
                "Feature output {} is not equal to expect result {}.\n Using arguments \n\tFeature_1: {}, \n\t Feature_2: {}".format(
                    output, expected_result, f_a_input, f_b_input
                ),
            )
        if not isinstance(output, list):
            tester.assertFalse(
                not any(p == f_a.properties() for p in output.properties),
                "Feature_a properties {} not in output Image, with properties {}".format(
                    f_a.properties(), output.properties
                ),
            )


def test_operator(self, operator, emulated_operator=None):
    if emulated_operator is None:
        emulated_operator = operator

    value = features.Value(value=2)
    f = operator(value, 3)
    f.store_properties()
    self.assertEqual(f(), operator(2, 3))
    self.assertListEqual(f().get_property("value", get_one=False), [2, 3])

    f = operator(3, value)
    f.store_properties()
    self.assertEqual(f(), operator(3, 2))

    f = operator(value, lambda: 3)
    f.store_properties()
    self.assertEqual(f(), operator(2, 3))
    self.assertListEqual(f().get_property("value", get_one=False), [2, 3])

    grid_test_features(
        self,
        features.Value,
        features.Value,
        [
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        [
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        lambda a, b: emulated_operator(a["value"], b["value"]),
        operator,
    )


class TestFeatures(unittest.TestCase):

    def test_Feature_basics(self):

        F = features.DummyFeature()
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(F.properties(), {'name': 'DummyFeature'})

        F = features.DummyFeature(a=1, b=2)
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(F.properties(),
                         {'a': 1, 'b': 2, 'name': 'DummyFeature'})

        F = features.DummyFeature(prop_int=1, prop_bool=True, prop_str='a')
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(
            F.properties(),
            {'prop_int': 1, 'prop_bool': True, 'prop_str': 'a', 
             'name': 'DummyFeature'},
        )
        self.assertIsInstance(F.properties['prop_int'](), int)
        self.assertEqual(F.properties['prop_int'](), 1)
        self.assertIsInstance(F.properties['prop_bool'](), bool)
        self.assertEqual(F.properties['prop_bool'](), True)
        self.assertIsInstance(F.properties['prop_str'](), str)
        self.assertEqual(F.properties['prop_str'](), 'a')


    def test_Feature_properties_update(self):

        feature = features.DummyFeature(
            prop_a=lambda: np.random.rand(),
            prop_b="b",
            prop_c=iter(range(10)),
        )

        prop_dict = feature.properties()

        self.assertIsInstance(prop_dict["prop_a"], float)
        self.assertIsInstance(prop_dict["prop_b"], str)
        self.assertIsInstance(prop_dict["prop_c"], int)

        prop_dict_without_update = feature.properties()
        self.assertDictEqual(prop_dict, prop_dict_without_update)

        feature.update()
        prop_dict_with_update = feature.properties()
        self.assertNotEqual(prop_dict, prop_dict_with_update)


    def test_Feature_memorized(self):

        list_of_inputs = []

        class ConcreteFeature(features.Feature):
            __distributed__ = False

            def get(self, input, **kwargs):
                list_of_inputs.append(input)
                return input

        feature = ConcreteFeature(prop_a=1)
        self.assertEqual(len(list_of_inputs), 0)

        feature()
        self.assertEqual(len(list_of_inputs), 1)

        feature.update()
        self.assertEqual(len(list_of_inputs), 1)
        feature()
        self.assertEqual(len(list_of_inputs), 2)

        feature.prop_a.set_value(1)
        feature()
        self.assertEqual(len(list_of_inputs), 2)

        feature.prop_a.set_value(2)
        feature()
        self.assertEqual(len(list_of_inputs), 3)

        feature([])
        self.assertEqual(len(list_of_inputs), 3)

        feature([1])
        self.assertEqual(len(list_of_inputs), 4)


    def test_Feature_dependence(self):

        A = features.Value(lambda: np.random.rand())
        B = features.Value(value=A.value)
        C = features.Value(value=B.value + 1)
        D = features.Value(value=C.value + B.value)
        E = features.Value(value=D + C.value)

        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        A.update()
        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        B.update()
        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        C.update()
        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        D.update()
        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        E.update()
        self.assertEqual(B(), A())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())


    def test_Feature_validation(self):

        class ConcreteFeature(features.Feature):
            __distributed__ = False
            def get(self, input, **kwargs):
                return input

        feature = ConcreteFeature(prop=1)

        self.assertFalse(feature.is_valid())

        feature()
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(1)  # Does not change value.
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(2)  # Changes value.
        self.assertFalse(feature.is_valid())


    def test_Feature_store_properties_in_image(self):

        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature = FeatureAddValue(value_to_add=1)
        feature.store_properties()  # Return an Image containing properties.
        feature.update()
        input_image = np.zeros((1, 1))

        output_image = feature.resolve(input_image)
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image, 1)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1]
        )

        output_image = feature.resolve(output_image)
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image, 2)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1, 1]
        )


    def test_Feature_with_dummy_property(self):

        class FeatureConcreteClass(features.Feature):
            __distributed__ = False
            def get(self, *args, **kwargs):
                image = np.ones((2, 3))
                return image

        feature = FeatureConcreteClass(dummy_property="foo")
        feature.store_properties()  # Return an Image containing properties.
        feature.update()
        output_image = feature.resolve()
        self.assertListEqual(
            output_image.get_property("dummy_property", get_one=False), ["foo"]
        )


    def test_Feature_plus_1(self):

        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureAddValue(value_to_add=2)
        feature = feature1 >> feature2
        feature.store_properties()  # Return an Image containing properties.
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 3)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1, 2]
        )
        self.assertEqual(
            output_image.get_property("value_to_add", get_one=True), 1
        )


    def test_Feature_plus_2(self):

        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        class FeatureMultiplyByValue(features.Feature):
            def get(self, image, value_to_multiply=0, **kwargs):
                image = image * value_to_multiply
                return image

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureMultiplyByValue(value_to_multiply=10)
        input_image = np.zeros((1, 1))

        feature12 = feature1 >> feature2
        feature12.update()
        output_image12 = feature12.resolve(input_image)
        self.assertEqual(output_image12, 10)

        feature21 = feature2 >> feature1
        feature12.update()
        output_image21 = feature21.resolve(input_image)
        self.assertEqual(output_image21, 1)


    def test_Feature_plus_3(self):

        class FeatureAppendImageOfShape(features.Feature):
            __distributed__ = False
            __list_merge_strategy__ = features.MERGE_STRATEGY_APPEND
            def get(self, *args, shape, **kwargs):
                image = np.zeros(shape)
                return image

        feature1 = FeatureAppendImageOfShape(shape=(1, 1))
        feature2 = FeatureAppendImageOfShape(shape=(2, 2))
        feature12 = feature1 >> feature2
        feature12.update()
        output_image = feature12.resolve()
        self.assertIsInstance(output_image, list)
        self.assertIsInstance(output_image[0], np.ndarray)
        self.assertIsInstance(output_image[1], np.ndarray)
        self.assertEqual(output_image[0].shape, (1, 1))
        self.assertEqual(output_image[1].shape, (2, 2))


    def test_Feature_arithmetic(self):

        inp = features.DummyFeature()

        pipeline = inp - inp * 2

        input_1 = 10
        self.assertEqual(pipeline(input_1), -input_1)

        input_2 = [10, 20]
        self.assertListEqual(pipeline(input_2), [-input_2[0], -input_2[1]])


    def test_Features_chain_lambda(self):

        value = features.Value(value=1)
        func = lambda x: x + 1

        feature = value >> func
        feature.store_properties()  # Return an Image containing properties.

        feature.update()
        output_image = feature()
        self.assertEqual(output_image, 2)


    def test_Feature_repeat(self):

        feature = features.Value(value=0) \
            >> (features.Add(1) ^ iter(range(10)))

        for n in range(10):
            feature.update()
            output_image = feature()
            self.assertEqual(np.array(output_image), np.array(n))


    def test_Feature_repeat_random(self):

        feature = features.Value(value=0) >> (
            features.Add(value=lambda: np.random.randint(100)) ^ 100
        )
        feature.store_properties()  # Return an Image containing properties.
        feature.update()
        output_image = feature()
        values = output_image.get_property("value", get_one=False)[1:]

        num_dups = values.count(values[0])
        self.assertNotEqual(num_dups, len(values))
        self.assertEqual(output_image, sum(values))


    def test_Feature_repeat_nested(self):

        value = features.Value(0)
        add = features.Add(5)
        sub = features.Subtract(1)

        feature = value >> (((add ^ 2) >> (sub ^ 5)) ^ 3)

        self.assertEqual(feature(), 15)


    def test_Feature_repeat_nested_random_times(self):

        value = features.Value(0)
        add = features.Add(5)
        sub = features.Subtract(1)

        feature = value >> (
            ((add ^ 2) >> (sub ^ 5)) ^ (lambda: np.random.randint(2, 5))
        )

        for _ in range(5):
            feature.update()
            self.assertEqual(feature(), feature.feature_2.N() * 5)


    def test_Feature_repeat_nested_random_addition(self):

        value = features.Value(0)
        add = features.Add(lambda: np.random.rand())
        sub = features.Subtract(1)

        feature = value >> (((add ^ 2) >> (sub ^ 3)) ^ 4)
        feature.store_properties()  # Return an Image containing properties.

        feature.update()

        for _ in range(4):

            feature.update()

            added_values = list(
                map(
                    lambda f: f["value"],
                    filter(lambda f: f["name"] == "Add", feature().properties),
                )
            )
            self.assertEqual(len(added_values), 8)
            np.testing.assert_almost_equal(
                sum(added_values) - 3 * 4, feature()
            )


    def test_Feature_nested_Duplicate(self):

        A = features.DummyFeature(
            a=lambda: np.random.randint(100) * 1000,
        )
        B = features.DummyFeature(
            a2=A.a,
            b=lambda a2: a2 + np.random.randint(10) * 100,
        )
        C = features.DummyFeature(
            b2=B.b,
            c=lambda b2: b2 + np.random.randint(10) * 10,
        )
        D = features.DummyFeature(
            c2=C.c,
            d=lambda c2: c2 + np.random.randint(10) * 1,
        )

        for _ in range(5):

            AB = A >> (B >> (C >> D ^ 2) ^ 3) ^ 4
            AB.store_properties()

            output = AB.update().resolve(0)
            al = output.get_property("a", get_one=False)
            bl = output.get_property("b", get_one=False)
            cl = output.get_property("c", get_one=False)
            dl = output.get_property("d", get_one=False)

            self.assertFalse(all(a == al[0] for a in al))
            self.assertFalse(all(b == bl[0] for b in bl))
            self.assertFalse(all(c == cl[0] for c in cl))
            self.assertFalse(all(d == dl[0] for d in dl))
            for ai, a in enumerate(al):
                for bi, b in list(enumerate(bl))[ai * 3 : (ai + 1) * 3]:
                    self.assertIn(b - a, range(0, 1000))
                    for ci, c in list(enumerate(cl))[bi * 2 : (bi + 1) * 2]:
                        self.assertIn(c - b, range(0, 100))
                        self.assertIn(dl[ci] - c, range(0, 10))


    def test_Feature_outside_dependence(self):

        A = features.DummyFeature(
            a=lambda: np.random.randint(100) * 1000,
        )

        B = features.DummyFeature(
            a2=A.a,
            b=lambda a2: a2 + np.random.randint(10) * 100,
        )

        AB = A >> (B ^ 5)
        AB.store_properties()

        for _ in range(5):
            AB.update()
            output = AB(0)
            self.assertEqual(len(output.get_property("a", get_one=False)), 1)
            self.assertEqual(len(output.get_property("b", get_one=False)), 5)

            a = output.get_property("a")
            for b in output.get_property("b", get_one=False):
                self.assertLess(b - a, 1000)
                self.assertGreaterEqual(b - a, 0)


    def test_Chain(self):

        class Addition(features.Feature):
            """Simple feature that adds a constant."""
            def get(self, image, **kwargs):
                # 'addend' is a property set via self.properties (default: 0).
                return image + self.properties.get("addend", 0)()

        class Multiplication(features.Feature):
            """Simple feature that multiplies by a constant."""
            def get(self, image, **kwargs):
                # 'multiplier' is a property set via self.properties (default: 1).
                return image * self.properties.get("multiplier", 1)()

        A = Addition(addend=10)
        M = Multiplication(multiplier=0.5)

        input_image = np.ones((2, 3))

        chain_AM = features.Chain(A, M)
        self.assertTrue(np.array_equal(
            chain_AM(input_image),
            (np.ones((2, 3)) + A.properties["addend"]())
            * M.properties["multiplier"](),
            )
        )

        chain_MA = features.Chain(M, A)
        self.assertTrue(np.array_equal(
            chain_MA(input_image),
            (np.ones((2, 3)) * M.properties["multiplier"]()
            + A.properties["addend"]()),
            )
        )
    

    def test_DummyFeature(self):
        """Test that the DummyFeature correctly returns the value of its properties."""

        feature = features.DummyFeature(a=1, b=2, c=3)

        self.assertEqual(feature.a(), 1)
        self.assertEqual(feature.b(), 2)
        self.assertEqual(feature.c(), 3)

        feature.a.set_value(4)
        self.assertEqual(feature.a(), 4)

        feature.b.set_value(5)
        self.assertEqual(feature.b(), 5)

        feature.c.set_value(6)
        self.assertEqual(feature.c(), 6)


    def test_Value(self):

        value = features.Value(value=1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)
        self.assertEqual(value(value=2), 2)
        self.assertEqual(value.value(), 2)

        value = features.Value(value=lambda: 1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)
        self.assertNotEqual(value(value=lambda: 2), 2)
        self.assertNotEqual(value.value(), 2)


    def test_ArithmeticOperationFeature(self):

        addition_feature = \
            features.ArithmeticOperationFeature(operator.add, value=10)
        input_values = [1, 2, 3, 4]
        expected_output = [11, 12, 13, 14]
        output = addition_feature(input_values)
        self.assertEqual(output, expected_output)    


    def test_Add(self):
        test_operator(self, operator.add)


    def test_Subtract(self):
        test_operator(self, operator.sub)


    def test_Multiply(self):
        test_operator(self, operator.add)


    def test_Divide(self):
        test_operator(self, operator.truediv)


    def test_FloorDivide(self):
        test_operator(self, operator.floordiv)


    def test_Power(self):
        test_operator(self, operator.pow)

    def test_LessThan(self):
        test_operator(self, operator.lt)


    def test_LessThanOrEquals(self):
        test_operator(self, operator.le)


    def test_GreaterThan(self):
        test_operator(self, operator.gt)


    def test_GreaterThanOrEquals(self):
        test_operator(self, operator.ge)


    def test_Equals(self):
        """
        Notes
        -----
        - Unlike other arithmetic operators, `Equals` does not define `__eq__` 
          (`==`) and `__req__` (`==`) in `DeepTrackNode` and `Feature`, as this 
          would affect Python’s built-in identity comparison.
        - This means that the standard `==` operator is overloaded only for 
          expressions involving `Feature` instances but not for comparisons 
          involving regular Python objects.
        - Always use `>>` to apply `Equals` correctly in a feature chain.

        """
        
        equals_feature = features.Equals(value=2)
        input_values = np.array([1, 2, 3])
        output_values = equals_feature(input_values)
        self.assertTrue(np.array_equal(output_values, [False, True, False]))


    def test_Stack(self):
        value = features.Value(value=2)
        f = value & 3
        self.assertEqual(f(), [2, 3])

        f = 3 & value
        self.assertEqual(f(), [3, 2])

        f = value & (lambda: 3)
        self.assertEqual(f(), [2, 3])

        grid_test_features(
            self,
            features.Value,
            features.Value,
            [
                {"value": 1},
                {"value": [1, 2]},
                {"value": np.nan},
                {"value": np.inf},
                {"value": np.random.rand(10, 10)},
            ],
            [
                {"value": 1},
                {"value": [1, 2]},
                {"value": np.nan},
                {"value": np.inf},
                {"value": np.random.rand(10, 10)},
            ],
            lambda a, b: [
                *(a["value"] if isinstance(a["value"], list) else [a["value"]]),
                *(b["value"] if isinstance(b["value"], list) else [b["value"]]),
            ],
            operator.__and__,
        )

    def test_Arguments_feature_passing(self):
        """Tests that arguments are correctly passed and updated in a feature pipeline."""

        # Define Arguments with static and dynamic values
        arguments = features.Arguments(
            a="foo",
            b="bar",
            c=lambda a, b: a + b,  # "foobar"
            d=np.random.rand,  # Random float in [0, 1]
        )

        # First feature with dependencies on arguments
        f1 = features.DummyFeature(
            p1=arguments.a,  # "foo"
            p2=lambda p1: p1 + "baz"  # "foobaz"
        )

        # Second feature dependent on the first
        f2 = features.DummyFeature(
            p1=f1.p2,  # Should be "foobaz"
            p2=arguments.d,  # Random value
        )

        # Assertions
        self.assertEqual(f1.properties['p1'](), "foo")  # Check that p1 is set correctly
        self.assertEqual(f1.properties['p2'](), "foobaz")  # Check lambda evaluation

        self.assertEqual(f2.properties['p1'](), "foobaz")  # Check dependency resolution

        # Ensure p2 in f2 is a valid float between 0 and 1
        self.assertTrue(0 <= f2.properties['p2']() <= 1)

        # Ensure `c` was computed correctly
        self.assertEqual(arguments.c(), "foobar")  # Should concatenate "foo" + "bar"

        # Test that d is dynamic (generates new values)
        first_d = arguments.d.update()()
        second_d = arguments.d.update()()
        self.assertNotEqual(first_d, second_d)  # Check that values change


    def test_Arguments(self):
        from tempfile import NamedTemporaryFile
        from PIL import Image as PIL_Image
        import os 

        """Creates a temporary test image."""
        test_image_array = (np.ones((50, 50)) * 128).astype(np.uint8)
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
            PIL_Image.fromarray(test_image_array).save(temp_png.name)

        try: 
            """Tests pipeline behavior when toggling `is_label`."""
            arguments = features.Arguments(is_label=False)
            image_pipeline = (
                features.LoadImage(path=temp_png.name) >>
                Gaussian(sigma=(1 - arguments.is_label) * 5)
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise expected

            """Tests pipeline behavior with dynamically computed sigma."""
            arguments = features.Arguments(is_label=False)
            image_pipeline = (
                features.LoadImage(path=temp_png.name) >>
                Gaussian(
                    is_label=arguments.is_label,
                    sigma=lambda is_label: 0 if is_label else 5
                )
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise expected

            """Tests property storage and modification in the pipeline."""
            arguments = features.Arguments(noise_max_sigma=5)
            image_pipeline = (
                features.LoadImage(path=temp_png.name) >>
                Gaussian(
                    noise_max_sigma=arguments.noise_max_sigma,
                    sigma=lambda noise_max_sigma: np.random.rand() * noise_max_sigma
                )
            )
            image_pipeline.bind_arguments(arguments)
            image_pipeline.store_properties()

            # Check if sigma is within expected range
            image = image_pipeline()
            sigma_value = image.get_property("sigma")
            self.assertTrue(0 <= sigma_value <= 5)

            # Override sigma by setting noise_max_sigma=0
            image = image_pipeline(noise_max_sigma=0)
            self.assertEqual(image.get_property("sigma"), 0.0)

            """Tests passing arguments dynamically using `**arguments.properties`."""
            arguments = features.Arguments(is_label=False, noise_sigma=5)
            image_pipeline = (
                features.LoadImage(path=temp_png.name) >>
                Gaussian(
                    sigma=lambda is_label, noise_sigma: 0 if is_label else noise_sigma,
                    **arguments.properties
                )
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise expected
        
        finally:
            if os.path.exists(temp_png.name):
                os.remove(temp_png.name)


    def test_Probability(self):
        np.random.seed(42)  # Set seed for reproducibility

        add_feature = features.Add(value=2)
        probabilistic_feature = features.Probability(
            feature = add_feature, 
            probability=0.7
        )
        
        input_image = np.ones((5, 5))

        applied_count = 0
        total_runs = 300

        for _ in range(total_runs):
            output_image = probabilistic_feature.update().resolve(input_image)

            if not np.array_equal(output_image, input_image): 
                applied_count += 1
                self.assertTrue(np.array_equal(output_image, input_image + 2))

        observed_probability = applied_count / total_runs
        self.assertTrue(0.65 <= observed_probability <= 0.75, f"Observed probability: {observed_probability}")


    def test_Repeat(self):
        add_ten = features.Add(value=10)

        pipeline = features.Repeat(add_ten, N=3)

        input_data = [1, 2, 3]
        expected_output = [31, 32, 33]

        output_data = pipeline.resolve(input_data)

        self.assertTrue(np.array_equal(output_data, expected_output), \
            f"Expected {expected_output}, got {output_data}")

        pipeline_shorthand = features.Add(value=10) ^ 3
        output_data_shorthand = pipeline_shorthand.resolve(input_data)

        self.assertTrue(np.array_equal(output_data_shorthand, expected_output), \
            f"Shorthand failed. Expected {expected_output}, \
                got {output_data_shorthand}")


    def test_Combine(self):

        noise_feature = Gaussian(mu=0, sigma=2)
        add_feature = features.Add(value=10)
        combined_feature = features.Combine([noise_feature, add_feature])

        input_image = np.ones((10, 10))
        output_list = combined_feature.resolve(input_image)

        self.assertTrue(isinstance(output_list, list), "Output should be a list")
        self.assertTrue(len(output_list) == 2, "Output list should contain results of both features")

        for output in output_list:
            self.assertTrue(output.shape == input_image.shape, "Output shape mismatch")

        noisy_image = output_list[0]
        added_image = output_list[1]

        self.assertFalse(np.all(noisy_image == 1), "Gaussian noise was not applied")
        self.assertTrue(np.allclose(added_image, input_image + 10), "Add operation failed")


    def test_Slice_constant(self):
        input = np.arange(9).reshape((3, 3))

        A = features.DummyFeature()
        A0 = A[0]
        A1 = A[1]
        A22 = A[2, 2]
        A12 = A[1, lambda: -1]

        a0 = A0.resolve(input)
        a1 = A1.resolve(input)
        a22 = A22.resolve(input)
        a12 = A12.resolve(input)

        self.assertEqual(a0.tolist(), input[0].tolist())
        self.assertEqual(a1.tolist(), input[1].tolist())
        self.assertEqual(a22, input[2, 2])
        self.assertEqual(a12, input[1, -1])


    def test_Slice_colon(self):

        input = np.arange(16).reshape((4, 4))

        A = features.DummyFeature()

        A0 = A[0, :1]
        A1 = A[1, lambda: 0 : lambda: 4 : lambda: 2]
        A2 = A[lambda: slice(0, 4, 1), 2]
        A3 = A[lambda: 0 : lambda: 2, :]

        a0 = A0.resolve(input)
        a1 = A1.resolve(input)
        a2 = A2.resolve(input)
        a3 = A3.resolve(input)

        self.assertEqual(a0.tolist(), input[0, :1].tolist())
        self.assertEqual(a1.tolist(), input[1, 0:4:2].tolist())
        self.assertEqual(a2.tolist(), input[:, 2].tolist())
        self.assertEqual(a3.tolist(), input[0:2, :].tolist())


    def test_Slice_ellipse(self):

        input = np.arange(16).reshape((4, 4))

        A = features.DummyFeature()

        A0 = A[..., :1]
        A1 = A[..., lambda: 0 : lambda: 4 : lambda: 2]
        A2 = A[lambda: slice(0, 4, 1), ...]
        A3 = A[lambda: 0 : lambda: 2, lambda: ...]

        a0 = A0.resolve(input)
        a1 = A1.resolve(input)
        a2 = A2.resolve(input)
        a3 = A3.resolve(input)

        self.assertEqual(a0.tolist(), input[..., :1].tolist())
        self.assertEqual(a1.tolist(), input[..., 0:4:2].tolist())
        self.assertEqual(a2.tolist(), input[:, ...].tolist())
        self.assertEqual(a3.tolist(), input[0:2, ...].tolist())


    def test_Slice_static_dynamic(self):
        image = np.arange(27).reshape((3, 3, 3))
        expected_output = image[:, 1:2, ::-2]

        feature = features.DummyFeature()

        static_slicing = feature[:, 1:2, ::-2]
        static_output = static_slicing.resolve(image)
        self.assertTrue(np.array_equal(static_output, expected_output))

        dynamic_slicing = feature >> features.Slice(
            slices=(slice(None), slice(1, 2), slice(None, None, -2))
        )
        dinamic_output = dynamic_slicing.resolve(image)
        self.assertTrue(np.array_equal(dinamic_output, expected_output))


    def test_Bind(self):

        value = features.Value(
            value=lambda input_value: input_value,
            input_value=10,
        )
        value = features.Value(
            value=lambda input_value: input_value,
            input_value=10,
        )
        pipeline = (value + 10) / value

        pipeline_with_small_input = features.Bind(pipeline, input_value=1)

        res = pipeline.update().resolve()
        self.assertEqual(res, 2)

        res = pipeline_with_small_input.update().resolve()
        self.assertEqual(res, 11)

        res = pipeline_with_small_input.update(input_value=10).resolve()
        self.assertEqual(res, 11)


    def test_Bind_gaussian_noise(self):
        # Define the Gaussian noise feature and bind its properties
        gaussian_noise = Gaussian()
        bound_feature = features.Bind(gaussian_noise, mu=-5, sigma=2)

        # Create the input image
        input_image = np.zeros((512, 512))

        # Resolve the feature to get the output image
        output_image = bound_feature.resolve(input_image)

        # Calculate the mean and standard deviation of the output
        output_mean = np.mean(output_image)
        output_std = np.std(output_image)

        # Assert that the mean and standard deviation are close to the bound values
        self.assertAlmostEqual(output_mean, -5, delta=0.2, \
            msg="Mean is not within the expected range")
        self.assertAlmostEqual(output_std, 2, delta=0.2, \
            msg="Standard deviation is not within the expected range")


    def test_BindResolve(self):

        value = features.Value(
            value=lambda input_value: input_value,
            input_value=10,
        )
        value = features.Value(
            value=lambda input_value: input_value,
            input_value=10,
        )
        pipeline = (value + 10) / value

        pipeline_with_small_input = features.BindResolve(
            pipeline,
            input_value=1
        )
        pipeline_with_small_input = features.BindResolve(
            pipeline,
            input_value=1
        )

        res = pipeline.update().resolve()
        self.assertEqual(res, 2)

        res = pipeline_with_small_input.update().resolve()
        self.assertEqual(res, 11)

        res = pipeline_with_small_input.update(input_value=10).resolve()
        self.assertEqual(res, 11)


    def test_BindUpdate(self):

        value = features.Value(
            value=lambda input_value: input_value, 
            input_value=10,
            )
        value = features.Value(
            value=lambda input_value: input_value, 
            input_value=10,
            )
        pipeline = (value + 10) / value

        pipeline_with_small_input = features.BindUpdate(
            pipeline, 
            input_value=1,
        )
        pipeline_with_small_input = features.BindUpdate(
            pipeline, 
            input_value=1,
        )

        res = pipeline.update().resolve()
        self.assertEqual(res, 2)

        res = pipeline_with_small_input.update().resolve()
        self.assertEqual(res, 11)

        res = pipeline_with_small_input.update(input_value=10).resolve()
        self.assertEqual(res, 11)
    
    
    def test_BindUpdate_gaussian_noise(self):
        # Define the Gaussian noise feature and bind its properties
        gaussian_noise = Gaussian()
        bound_feature = features.BindUpdate(gaussian_noise, mu=5, sigma=3)

        # Create the input image
        input_image = np.zeros((512, 512))

        # Resolve the feature to get the output image
        output_image = bound_feature.resolve(input_image)

        # Calculate the mean and standard deviation of the output
        output_mean = np.mean(output_image)
        output_std = np.std(output_image)

        # Assert that the mean and standard deviation are close to the bound values
        self.assertAlmostEqual(output_mean, 5, \
            delta=0.2, msg="Mean is not within the expected range")
        self.assertAlmostEqual(output_std, 3, \
            delta=0.2, msg="Standard deviation is not within the expected range")


    def test_ConditionalSetProperty(self):
        """Test that ConditionalSetProperty correctly modifies properties based on condition."""

        """Set up a Gaussian feature and a test image before each test."""
        gaussian_noise = Gaussian(sigma=0)
        image = np.ones((128, 128))

        """Test that sigma is correctly applied when condition is a boolean."""
        conditional_feature = features.ConditionalSetProperty(
            gaussian_noise, sigma=5,
        )

        # Test with condition met (should apply sigma=5)
        noisy_image = conditional_feature(image, condition=True)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Test without condition met (should apply sigma=0)
        clean_image = conditional_feature.update()(image, condition=False)
        self.assertEqual(clean_image.std(), 0)

        """Test that sigma is correctly applied when condition is a string property."""
        conditional_feature = features.ConditionalSetProperty(
            gaussian_noise, sigma=5, condition="is_noisy"
        )

        # Test with condition met (should apply sigma=5)
        noisy_image = conditional_feature(image, is_noisy=True)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Test without condition met (should apply sigma=0)
        clean_image = conditional_feature.update()(image, is_noisy=False)
        self.assertEqual(clean_image.std(), 0)


    def test_ConditionalSetFeature(self):

        """Set up Gaussian noise features and test image before each test."""
        true_feature = Gaussian(sigma=0)    # Clean image (no noise)
        false_feature = Gaussian(sigma=5)   # Noisy image (sigma=5)
        image = np.ones((512, 512))

        """Test using a direct boolean condition."""
        conditional_feature = features.ConditionalSetFeature(
            on_true=true_feature,
            on_false=false_feature
        )

        # Default condition is True (no noise)
        clean_image = conditional_feature(image)
        self.assertEqual(clean_image.std(), 0)

        # Condition is False (sigma=5)
        noisy_image = conditional_feature(image, condition=False)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Condition is True (sigma=0)
        clean_image = conditional_feature(image, condition=True)
        self.assertEqual(clean_image.std(), 0)

        """Test using a string-based condition."""
        conditional_feature = features.ConditionalSetFeature(
            on_true=true_feature,
            on_false=false_feature,
            condition="is_noisy"
        )

        # Condition is False (sigma=5)
        noisy_image = conditional_feature(image, is_noisy=False)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Condition is True (sigma=0)
        clean_image = conditional_feature(image, is_noisy=True)
        self.assertEqual(clean_image.std(), 0)


    def test_Lambda_dependence(self):
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a" 
                             else (A.b() if key == "b" else A.c()),
        )

        B.update()
        self.assertEqual(B.prop(), 1)
        B.key.set_value("a")
        self.assertEqual(B.prop(), 1)
        B.key.set_value("b")
        self.assertEqual(B.prop(), 2)
        B.key.set_value("c")
        self.assertEqual(B.prop(), 3)


    def test_Lambda_dependence_twice(self):
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a" 
                             else (A.b() if key == "b" else A.c()),
            prop2=lambda prop: prop * 2,
        )

        B.update()
        self.assertEqual(B.prop2(), 2)
        B.key.set_value("a")
        self.assertEqual(B.prop2(), 2)
        B.key.set_value("b")
        self.assertEqual(B.prop2(), 4)
        B.key.set_value("c")
        self.assertEqual(B.prop2(), 6)


    def test_Lambda_dependence_other_feature(self):

        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a" 
                             else (A.b() if key == "b" else A.c()),
            prop2=lambda prop: prop * 2,
        )

        C = features.DummyFeature(B_prop=B.prop2, 
                                  prop=lambda B_prop: B_prop * 2)

        C.update()
        self.assertEqual(C.prop(), 4)
        B.key.set_value("a")
        self.assertEqual(C.prop(), 4)
        B.key.set_value("b")
        self.assertEqual(C.prop(), 8)
        B.key.set_value("c")
        self.assertEqual(C.prop(), 12)


    def test_Lambda_scaling(self):
        def scale_function_factory(scale=2):
            def scale_function(image):
                return image * scale
            return scale_function

        lambda_feature = features.Lambda(function=scale_function_factory, scale=5)
        input_image = np.ones((5, 5))

        output_image = lambda_feature.resolve(input_image)

        expected_output = np.ones((5, 5)) * 5
        self.assertTrue(np.array_equal(output_image, expected_output), "Arrays are not equal")

        lambda_feature = features.Lambda(function=scale_function_factory, scale=3)
        output_image = lambda_feature.resolve(input_image)

        expected_output = np.ones((5, 5)) * 3
        self.assertTrue(np.array_equal(output_image, expected_output), "Arrays are not equal")


    def test_Merge(self):

        def merge_function_factory():
            def merge_function(images):
                return np.mean(np.stack(images), axis=0)
            return merge_function

        merge_feature = features.Merge(function=merge_function_factory)

        image_1 = np.ones((5, 5)) * 2
        image_2 = np.ones((5, 5)) * 4
        expected_output = np.ones((5, 5)) * 3
        output_image = merge_feature.resolve([image_1, image_2])
        self.assertIsNone(np.testing.assert_array_almost_equal(output_image, expected_output))

        image_1 = np.ones((5, 5)) * 2
        image_2 = np.ones((3, 3)) * 4 
        with self.assertRaises(ValueError):
            merge_feature.resolve([image_1, image_2])

        image_1 = np.ones((5, 5)) * 2
        output_image = merge_feature.resolve([image_1])
        self.assertIsNone(np.testing.assert_array_almost_equal(output_image, image_1))


    def test_OneOf(self):
        """Set up the features and input image for testing."""
        feature_1 = features.Add(value=10)
        feature_2 = features.Multiply(value=2)
        input_image = np.array([1, 2, 3])

        """Test that OneOf applies one of the features randomly."""
        one_of_feature = features.OneOf([feature_1, feature_2])
        output_image = one_of_feature.resolve(input_image)
        
        # The output should either be:
        # - self.input_image + 10 (if feature_1 is chosen)
        # - self.input_image * 2  (if feature_2 is chosen)
        expected_outputs = [
            input_image + 10,
            input_image * 2
        ]
        self.assertTrue(
            any(np.array_equal(output_image, expected) for expected in expected_outputs),
            f"Output {output_image} did not match any expected transformations."
        )

        """Test that OneOf applies the selected feature when `key` is provided."""
        controlled_feature = features.OneOf([feature_1, feature_2], key=0)
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image + 10
        self.assertTrue(np.array_equal(output_image, expected_output))

        controlled_feature = features.OneOf([feature_1, feature_2], key=1)
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image * 2
        self.assertTrue(np.array_equal(output_image, expected_output))

    def test_OneOf_list(self):

        values = features.OneOf(
            [features.Value(1), features.Value(2), features.Value(3)]
        )

        has_been_one = False
        has_been_two = False
        has_been_three = False

        for _ in range(50):
            val = values.update().resolve()
            self.assertIn(val, [1, 2, 3])
            if val == 1:
                has_been_one = True
            elif val == 2:
                has_been_two = True
            else:
                has_been_three = True
        self.assertTrue(has_been_one)
        self.assertTrue(has_been_two)
        self.assertTrue(has_been_three)

        self.assertEqual(values.update().resolve(key=0), 1)

        self.assertEqual(values.update().resolve(key=1), 2)

        self.assertEqual(values.update().resolve(key=2), 3)

        self.assertRaises(IndexError, lambda: values.update().resolve(key=3))


    def test_OneOf_tuple(self):

        values = features.OneOf(
            (features.Value(1), features.Value(2), features.Value(3))
        )

        has_been_one = False
        has_been_two = False
        has_been_three = False

        for _ in range(50):
            val = values.update().resolve()
            self.assertIn(val, [1, 2, 3])
            if val == 1:
                has_been_one = True
            elif val == 2:
                has_been_two = True
            else:
                has_been_three = True
        self.assertTrue(has_been_one)
        self.assertTrue(has_been_two)
        self.assertTrue(has_been_three)

        self.assertEqual(values.update().resolve(key=0), 1)

        self.assertEqual(values.update().resolve(key=1), 2)

        self.assertEqual(values.update().resolve(key=2), 3)

        self.assertRaises(IndexError, lambda: values.update().resolve(key=3))


    def test_OneOf_set(self):

        values = features.OneOf(
            set([features.Value(1), features.Value(2), features.Value(3)])
        )

        has_been_one = False
        has_been_two = False
        has_been_three = False

        for _ in range(50):
            val = values.update().resolve()
            self.assertIn(val, [1, 2, 3])
            if val == 1:
                has_been_one = True
            elif val == 2:
                has_been_two = True
            else:
                has_been_three = True
        self.assertTrue(has_been_one)
        self.assertTrue(has_been_two)
        self.assertTrue(has_been_three)

        self.assertRaises(IndexError, lambda: values.update().resolve(key=3))


    def test_OneOfDict_basic(self):

        values = features.OneOfDict(
            {"1": features.Value(1), "2": features.Value(2), "3": features.Value(3)}
        )

        has_been_one = False
        has_been_two = False
        has_been_three = False

        for _ in range(50):
            val = values.update().resolve()
            self.assertIn(val, [1, 2, 3])
            if val == 1:
                has_been_one = True
            elif val == 2:
                has_been_two = True
            else:
                has_been_three = True
        self.assertTrue(has_been_one)
        self.assertTrue(has_been_two)
        self.assertTrue(has_been_three)

        self.assertEqual(values.update().resolve(key="1"), 1)

        self.assertEqual(values.update().resolve(key="2"), 2)

        self.assertEqual(values.update().resolve(key="3"), 3)

        self.assertRaises(KeyError, lambda: values.update().resolve(key="4"))


    def test_OneOfDict(self):
        features_dict = {
            "add": features.Add(value=10),
            "multiply": features.Multiply(value=2),
        }
        one_of_dict_feature = features.OneOfDict(features_dict)

        input_image = np.array([1, 2, 3])

        """Test that OneOfDict selects a feature randomly and applies it correctly."""
        output_image = one_of_dict_feature.resolve(input_image)
        expected_outputs = [
            input_image + 10,  # "add"
            input_image * 2,   # "multiply"
        ]
        self.assertTrue(
            any(np.array_equal(output_image, expected) for expected in expected_outputs),
            f"Output {output_image} did not match any expected transformations."
        )

        """Test that OneOfDict selects the correct feature when a key is specified."""
        controlled_feature = features.OneOfDict(features_dict, key="add")
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image + 10  # The "add" feature should be applied
        self.assertTrue(np.array_equal(output_image, expected_output))

        controlled_feature = features.OneOfDict(features_dict, key="multiply")
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image * 2  # The "multiply" feature should be applied
        self.assertTrue(np.array_equal(output_image, expected_output))
    

    def test_LoadImage(self):
        from tempfile import NamedTemporaryFile
        from PIL import Image as PIL_Image
        import os

        """Create temporary image files in multiple formats for testing."""
        test_image_array = (np.random.rand(50, 50) * 255).astype(np.uint8)

        try:
            with NamedTemporaryFile(suffix=".npy", delete=False) as temp_npy:
                np.save(temp_npy.name, test_image_array)
                # npy_filename = temp_npy.name

            with NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
                PIL_Image.fromarray(test_image_array).save(temp_png.name)
                # png_filename = temp_png.name

            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_jpg:
                PIL_Image.fromarray(test_image_array).convert("RGB").save(temp_jpg.name)
                # jpg_filename = temp_jpg.name


            """Test loading a .npy file."""
            load_feature = features.LoadImage(path=temp_npy.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2], test_image_array.shape[:2])

            """Test loading a .png file."""
            load_feature = features.LoadImage(path=temp_png.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2], test_image_array.shape[:2])

            """Test loading a .jpg file."""
            load_feature = features.LoadImage(path=temp_jpg.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2], test_image_array.shape[:2])
            
            """Test loading an image and converting it to grayscale."""
            load_feature = features.LoadImage(path=temp_png.name, to_grayscale=True)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[-1], 1) 

            """Test ensuring a minimum number of dimensions."""
            load_feature = features.LoadImage(path=temp_png.name, ndim=4)
            loaded_image = load_feature.resolve()
            self.assertGreaterEqual(len(loaded_image.shape), 4)  

        finally:
            for file in [temp_npy.name, temp_png.name, temp_jpg.name]:
                os.remove(file)


    def test_SampleToMasks(self):
        # Parameters
        n_particles = 12
        tolerance = 1  # Allowable pixelation offset

        # Define the optics and particle
        microscope = optics.Fluorescence(output_region=(0, 0, 64, 64))
        particle = scatterers.PointParticle(
            position=lambda: np.random.uniform(5, 55, size=2)
        )
        particles = particle ^ n_particles

        # Define pipelines
        sim_im_pip = microscope(particles)
        sim_mask_pip = particles >> features.SampleToMasks(
            lambda: lambda particles: particles > 0,
            output_region=microscope.output_region,
            merge_method="or",
        )
        pipeline = sim_im_pip & sim_mask_pip
        pipeline.store_properties()
        
        # Generate image and mask
        image, mask = pipeline.update()()

        # Assertions
        self.assertEqual(image.shape, (64, 64, 1), "Image shape is incorrect")
        self.assertEqual(mask.shape, (64, 64, 1), "Mask shape is incorrect")

        # Ensure mask is binary
        self.assertTrue(np.all(np.logical_or(mask == 0, mask == 1)), "Mask is not binary")

        # Ensure the number of particles matches the sum of the mask
        self.assertEqual(np.sum(mask), n_particles, "Number of particles in mask is incorrect")

        # Compare particle positions and mask positions
        positions = np.array(image.get_property("position", get_one=False))
        mask_positions = np.argwhere(mask.squeeze() == 1)

        # Ensure each particle position has a mask pixel nearby within tolerance
        for pos in positions:
            self.assertTrue(
                any(np.linalg.norm(pos - mask_pos) <= tolerance for mask_pos in mask_positions),
                f"Particle at position {pos} not found within tolerance in mask"
            )


    def test_AsType(self):

        input_image = np.array([1.5, 2.5, 3.5])

        data_types = ["float64", "int32", "uint16", "int16", "uint8", "int8"]
        for dtype in data_types:
            astype_feature = features.AsType(dtype=dtype)
            output_image = astype_feature.get(input_image, dtype=dtype)
            self.assertTrue(output_image.dtype == np.dtype(dtype))

            # Additional check for specific behavior of integers.
            if np.issubdtype(np.dtype(dtype), np.integer):
                # Verify that fractional parts are truncated
                self.assertTrue(
                    np.all(output_image == np.array([1, 2, 3], dtype=dtype))
                )


    def test_ChannelFirst2d(self):

        channel_first_feature = features.ChannelFirst2d()

        input_image_2d = np.random.rand(10, 20)
        output_image = channel_first_feature.get(input_image_2d, axis=-1)
        self.assertEqual(output_image.shape, (1, 10, 20))

        input_image_3d = np.random.rand(10, 20, 3)
        output_image = channel_first_feature.get(input_image_3d, axis=-1)
        self.assertEqual(output_image.shape, (3, 10, 20))


    def test_Upscale(self):
        microscope = optics.Fluorescence(output_region=(0, 0, 32, 32))
        particle = scatterers.PointParticle(position=(16, 16))
        simple_pipeline = microscope(particle)
        upscaled_pipeline = features.Upscale(simple_pipeline, factor=4)

        image = simple_pipeline.update()()
        upscaled_image = upscaled_pipeline.update()()

        self.assertEqual(image.shape, upscaled_image.shape,
                         "Upscaled image shape should match original image shape")

        # Allow slight differences due to upscaling and downscaling
        difference = np.abs(image - upscaled_image)
        mean_difference = np.mean(difference)

        self.assertLess(mean_difference, 1E-4,
                        "The upscaled image should be similar to the original within a tolerance")



    def test_NonOverlapping_resample_volume_position(self):

        nonOverlapping = features.NonOverlapping(
            features.Value(value=1),
        )

        positions_no_unit = [1, 2]
        positions_with_unit = [1 * units.px, 2 * units.px]

        positions_no_unit_iter = iter(positions_no_unit)
        positions_with_unit_iter = iter(positions_with_unit)

        volume_1 = scatterers.PointParticle(
            position=lambda: next(positions_no_unit_iter)
        )()
        volume_2 = scatterers.PointParticle(
            position=lambda: next(positions_with_unit_iter)
        )()

        # Test.
        self.assertEqual(volume_1.get_property("position"), positions_no_unit[0])
        self.assertEqual(
            volume_2.get_property("position"),
            positions_with_unit[0].to("px").magnitude,
        )

        nonOverlapping._resample_volume_position(volume_1)
        nonOverlapping._resample_volume_position(volume_2)

        self.assertEqual(volume_1.get_property("position"), positions_no_unit[1])
        self.assertEqual(
            volume_2.get_property("position"),
            positions_with_unit[1].to("px").magnitude,
        )

    def test_NonOverlapping_check_volumes_non_overlapping(self):
        nonOverlapping = features.NonOverlapping(
            features.Value(value=1),
        )

        volume_test0_a = np.zeros((5, 5, 5))
        volume_test0_b = np.zeros((5, 5, 5))

        volume_test1_a = np.zeros((5, 5, 5))
        volume_test1_b = np.zeros((5, 5, 5))
        volume_test1_a[0, 0, 0] = 1
        volume_test1_b[0, 0, 0] = 1

        volume_test2_a = np.zeros((5, 5, 5))
        volume_test2_b = np.zeros((5, 5, 5))
        volume_test2_a[0, 0, 0] = 1
        volume_test2_b[0, 0, 1] = 1

        volume_test3_a = np.zeros((5, 5, 5))
        volume_test3_b = np.zeros((5, 5, 5))
        volume_test3_a[0, 0, 0] = 1
        volume_test3_b[0, 1, 0] = 1

        volume_test4_a = np.zeros((5, 5, 5))
        volume_test4_b = np.zeros((5, 5, 5))
        volume_test4_a[0, 0, 0] = 1
        volume_test4_b[1, 0, 0] = 1

        volume_test5_a = np.zeros((5, 5, 5))
        volume_test5_b = np.zeros((5, 5, 5))
        volume_test5_a[0, 0, 0] = 1
        volume_test5_b[0, 1, 1] = 1

        volume_test6_a = np.zeros((5, 5, 5))
        volume_test6_b = np.zeros((5, 5, 5))
        volume_test6_a[1:3, 1:3, 1:3] = 1
        volume_test6_b[0:2, 0:2, 0:2] = 1

        volume_test7_a = np.zeros((5, 5, 5))
        volume_test7_b = np.zeros((5, 5, 5))
        volume_test7_a[2:4, 2:4, 2:4] = 1
        volume_test7_b[0:2, 0:2, 0:2] = 1

        volume_test8_a = np.zeros((5, 5, 5))
        volume_test8_b = np.zeros((5, 5, 5))
        volume_test8_a[3:, 3:, 3:] = 1
        volume_test8_b[:2, :2, :2] = 1

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test0_a,
                volume_test0_b,
                min_distance=0,
            ),
        )

        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test1_a,
                volume_test1_b,
                min_distance=0,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test2_a,
                volume_test2_b,
                min_distance=0,
            )
        )
        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test2_a,
                volume_test2_b,
                min_distance=1,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test3_a,
                volume_test3_b,
                min_distance=0,
            )
        )
        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test3_a,
                volume_test3_b,
                min_distance=1,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test4_a,
                volume_test4_b,
                min_distance=0,
            )
        )
        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test4_a,
                volume_test4_b,
                min_distance=1,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test5_a,
                volume_test5_b,
                min_distance=0,
            )
        )
        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test5_a,
                volume_test5_b,
                min_distance=1,
            )
        )

        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test6_a,
                volume_test6_b,
                min_distance=0,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test7_a,
                volume_test7_b,
                min_distance=0,
            )
        )
        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test7_a,
                volume_test7_b,
                min_distance=1,
            )
        )

        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test8_a,
                volume_test8_b,
                min_distance=0,
            )
        )
        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test8_a,
                volume_test8_b,
                min_distance=1,
            )
        )
        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test8_a,
                volume_test8_b,
                min_distance=2,
            )
        )
        self.assertTrue(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test8_a,
                volume_test8_b,
                min_distance=3,
            )
        )
        self.assertFalse(
            nonOverlapping._check_volumes_non_overlapping(
                volume_test8_a,
                volume_test8_b,
                min_distance=4,
            )
        )


    def test_NonOverlapping_check_non_overlapping(self):

        # Setup.
        nonOverlapping = features.NonOverlapping(
            features.Value(value=1),
            min_distance=1,
        )

        # Two spheres at the same position.
        volume_test0_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test0_b = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()

        # Two spheres of the same size, one under the other.
        volume_test1_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test1_b = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 10) * units.px
        )()

        # Two spheres of the same size, one under the other, but with a
        # spacing of 1.
        volume_test2_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test2_b = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 11) * units.px
        )()

        # Two spheres of the same size, one under the other, but with a
        # spacing of -1.
        volume_test3_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test3_b = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 9) * units.px
        )()

        # Two spheres of the same size, diagonally next to each other.
        volume_test4_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test4_b = scatterers.Sphere(
            radius=5 * units.px, position=(6, 6, 6) * units.px
        )()

        # Two spheres of the same size, diagonally next to each other, but
        # with a spacing of 1.
        volume_test5_a = scatterers.Sphere(
            radius=5 * units.px, position=(0, 0, 0) * units.px
        )()
        volume_test5_b = scatterers.Sphere(
            radius=5 * units.px, position=(7, 7, 7) * units.px
        )()

        # Run tests.
        self.assertFalse(
            nonOverlapping._check_non_overlapping(
                [volume_test0_a, volume_test0_b],
            )
        )

        self.assertFalse(
            nonOverlapping._check_non_overlapping(
                [volume_test1_a, volume_test1_b],
            )
        )

        self.assertTrue(
            nonOverlapping._check_non_overlapping(
                [volume_test2_a, volume_test2_b],
            )
        )

        self.assertFalse(
            nonOverlapping._check_non_overlapping(
                [volume_test3_a, volume_test3_b],
            )
        )

        self.assertFalse(
            nonOverlapping._check_non_overlapping(
                [volume_test4_a, volume_test4_b],
            )
        )

        self.assertTrue(
            nonOverlapping._check_non_overlapping(
                [volume_test5_a, volume_test5_b],
            )
        )

    def test_NonOverlapping_ellipses(self):
        """Set up common test objects before each test."""
        min_distance = 7  # Minimum distance in pixels
        radius = 10
        scatterer = scatterers.Ellipse(
            radius=radius * units.pixels,
            position=lambda: np.random.uniform(5, 115, size=2) * units.pixels,
        )
        random_scatterers = scatterer ^ 6
        fluo_optics = optics.Fluorescence()

        def calculate_min_distance(positions):
            """Calculate the minimum pairwise distance between objects."""
            distances = [
                np.linalg.norm(positions[i] - positions[j])
                for i in range(len(positions))
                for j in range(i + 1, len(positions))
            ]
            return min(distances)

        # Generate image with possible non-overlapping objects
        image_with_overlap = fluo_optics(random_scatterers)
        image_with_overlap.store_properties()
        im_with_overlap_resolved = image_with_overlap()
        pos_with_overlap = np.array(
            im_with_overlap_resolved.get_property(
                "position", 
                get_one=False
            )
        )

        # Generate image with enforced non-overlapping objects
        non_overlapping_scatterers = features.NonOverlapping(
            random_scatterers, 
            min_distance=min_distance
        )
        image_without_overlap = fluo_optics(non_overlapping_scatterers)
        image_without_overlap.store_properties()
        im_without_overlap_resolved = image_without_overlap()
        pos_without_overlap = np.array(
            im_without_overlap_resolved.get_property(
                "position",
                get_one=False
            )
        )

        # Compute minimum distances
        min_distance_before = calculate_min_distance(pos_with_overlap)
        min_distance_after = calculate_min_distance(pos_without_overlap)

        # print(f"Min distance before: {min_distance_before}, \
        #     should be smaller than {2*radius + min_distance}")
        # print(f"Min distance after: {min_distance_after}, should be larger \
        #     than {2*radius + min_distance} with some tolerance")

        # Assert that the non-overlapping case respects min_distance (with 
        # slight rounding tolerance)
        self.assertLess(min_distance_before, 2*radius + min_distance)  
        self.assertGreaterEqual(min_distance_after,2*radius + min_distance - 2)  


    def test_Store(self):
        value_feature = features.Value(lambda: np.random.rand())

        store_feature = features.Store(feature=value_feature, key="example")

        output = store_feature(None, key="example", replace=False)

        value_feature.update()
        cached_output = store_feature(None, key="example", replace=False)
        self.assertEqual(cached_output, output)

        value_feature.update()
        cached_output = store_feature(None, key="example", replace=True)
        self.assertNotEqual(cached_output, output)


    def test_Squeeze(self):

        input_image = np.array([[[[3], [2], [1]]],[[[1], [2], [3]]]])

        squeeze_feature = features.Squeeze(axis=1)
        output_image = squeeze_feature(input_image)
        self.assertEqual(output_image.shape, (2, 3, 1))

        squeeze_feature = features.Squeeze()
        output_image = squeeze_feature(input_image)
        self.assertEqual(output_image.shape, (2,3))


    def test_Unsqueeze(self):

        input_image = np.array([1, 2, 3])  # shape (3,)

        unsqueeze_feature = features.Unsqueeze(axis=0)
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (1, 3))

        unsqueeze_feature = features.Unsqueeze()
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (3, 1))


    def test_MoveAxis(self):

        input_image = np.random.rand(2, 3, 4)

        move_axis_feature = features.MoveAxis(source=0, destination=2)
        output_image = move_axis_feature(input_image)
        self.assertEqual(output_image.shape, (3, 4, 2))


    def test_Transpose(self):

        input_image = np.random.rand(2, 3, 4)

        transpose_feature = features.Transpose(axes=(1, 2, 0))
        output_image = transpose_feature(input_image)
        self.assertEqual(output_image.shape, (3, 4, 2))

        transpose_feature = features.Transpose()
        output_image = transpose_feature(input_image)
        self.assertEqual(output_image.shape, (4, 3, 2))


    def test_OneHot(self):

        input_image = np.array([0, 1, 2])

        one_hot_feature = features.OneHot(num_classes=3)
        output_image = one_hot_feature.get(input_image, num_classes=3)
        expected_output = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.assertTrue(np.array_equal(output_image, expected_output))


    def test_TakeProperties(self):
        # with custom feature
        class ExampleFeature(features.Feature):
            def __init__(self, my_property, **kwargs):
                super().__init__(my_property=my_property, **kwargs)

        feature = ExampleFeature(my_property=properties.Property(42))

        take_properties = features.TakeProperties(feature)
        take_properties = features.TakeProperties(feature)
        output = take_properties.get(image=None, names=["my_property"])
        self.assertEqual(output, [42])

        # with `Gaussian` feature 
        noise_feature = Gaussian(mu=7, sigma=12)
        
        take_properties = features.TakeProperties(noise_feature)
        output = take_properties.get(image=None, names=["mu"])
        self.assertEqual(output, [7])
        output = take_properties.get(image=None, names=["sigma"])
        self.assertEqual(output, [12])

        # with `Gaussian` feature 
        noise_feature = Gaussian(mu=7, sigma=12)
        
        take_properties = features.TakeProperties(noise_feature)
        output = take_properties.get(image=None, names=["mu"])
        self.assertEqual(output, [7])
        output = take_properties.get(image=None, names=["sigma"])
        self.assertEqual(output, [12])


if __name__ == "__main__":
    unittest.main()
