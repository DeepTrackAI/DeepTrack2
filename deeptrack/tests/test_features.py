# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import itertools
import operator
import unittest
import warnings

import numpy as np

from deeptrack import (
    config, features, Gaussian, properties, TORCH_AVAILABLE, xp,
)


if TORCH_AVAILABLE:
    import torch


def grid_test_features(
    tester,
    feature_a,
    feature_b,
    feature_a_inputs,
    feature_b_inputs,
    expected_result_function,
    assessed_operator,
):
    assert callable(feature_a), "First feature constructor must be callable"
    assert callable(feature_b), "Second feature constructor must be callable"
    assert (
        len(feature_a_inputs) > 0 and len(feature_b_inputs) > 0
    ), "Feature input lists cannot be empty"
    assert (
        callable(expected_result_function)
    ), "Result function must be callable"

    for f_a_input, f_b_input in itertools.product(
        feature_a_inputs, feature_b_inputs
    ):

        f_a = feature_a(**f_a_input)
        f_b = feature_b(**f_b_input)

        f = assessed_operator(f_a, f_b)
        tester.assertIsInstance(f, features.Chain)

        try:
            output = f()
        except Exception as e:
            tester.assertRaises(
                type(e),
                lambda: expected_result_function(
                    f_a.properties(), f_b.properties()
                ),
            )
            continue

        expected_output = expected_result_function(
            f_a.properties(), f_b.properties()
        )

        if isinstance(output, list) and isinstance(expected_output, list):
            for a, b in zip(output, expected_output):
                np.testing.assert_almost_equal(np.array(a), np.array(b))
        else:
            tester.assertTrue(
                np.array_equal(
                    np.array(output), np.array(expected_output), equal_nan=True
                ),
                "Output {output} different from expected {expected_result}.\n "
                "Using arguments \n"
                "\tFeature_1: {f_a_input}\n"
                "\t Feature_2: {f_b_input}"
            )


def test_operator(self, operator, emulated_operator=None):
    if emulated_operator is None:
        emulated_operator = operator

    value = features.Value(value=2)

    f = operator(value, 3)
    self.assertEqual(f(), operator(2, 3))

    f = operator(3, value)
    self.assertEqual(f(), operator(3, 2))

    f = operator(value, lambda: 3)
    self.assertEqual(f(), operator(2, 3))

    grid_test_features(
        self,
        feature_a=features.Value,
        feature_b=features.Value,
        feature_a_inputs=[
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        feature_b_inputs=[
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        expected_result_function= \
            lambda a, b: emulated_operator(a["value"], b["value"]),
        assessed_operator=operator,
    )

    if TORCH_AVAILABLE:
        grid_test_features(
            self,
            feature_a=features.Value,
            feature_b=features.Value,
            feature_a_inputs=[
                {"value": torch.tensor(1.0)},
                {"value": torch.tensor(0.5)},
                {"value": torch.tensor(float("nan"))},
                {"value": torch.tensor(float("inf"))},
                {"value": torch.rand(10, 10)},
            ],
            feature_b_inputs=[
                {"value": torch.tensor(1.0)},
                {"value": torch.tensor(0.5)},
                {"value": torch.tensor(float("nan"))},
                {"value": torch.tensor(float("inf"))},
                {"value": torch.rand(10, 10)},
            ],
            expected_result_function= \
                lambda a, b: emulated_operator(a["value"], b["value"]),
            assessed_operator=operator,
        )


class TestFeatures(unittest.TestCase):

    def test___all__(self):
        from deeptrack import (
            Feature,
            StructuralFeature,
            Chain,
            Branch,
            DummyFeature,
            Value,
            ArithmeticOperationFeature,
            Add,
            Subtract,
            Multiply,
            Divide,
            FloorDivide,
            Power,
            LessThan,
            LessThanOrEquals,
            LessThanOrEqual,
            GreaterThan,
            GreaterThanOrEquals,
            GreaterThanOrEqual,
            Equals,
            Equal,
            Stack,
            Arguments,
            Probability,
            Repeat,
            Combine,
            Slice,
            Bind,
            BindResolve,
            BindUpdate,
            ConditionalSetProperty,
            ConditionalSetFeature,
            Lambda,
            Merge,
            OneOf,
            OneOfDict,
            LoadImage,
            AsType,
            ChannelFirst2d,
            Store,
            Squeeze,
            Unsqueeze,
            ExpandDims,
            MoveAxis,
            Transpose,
            Permute,
            OneHot,
            TakeProperties,
        )


    def test_Feature_init(self):
        # Default init
        f1 = features.Feature()
        self.assertIsNone(f1.arguments)
        self.assertEqual(f1._backend, config.get_backend())

        self.assertEqual(f1.node_name, "Feature")
        self.assertIsInstance(f1.properties, properties.PropertyDict)
        self.assertIn("name", f1.properties)
        self.assertEqual(f1.properties["name"](), "Feature")

        self.assertIsInstance(f1._input, properties.DeepTrackNode)
        self.assertIsInstance(f1._random_seed, properties.DeepTrackNode)

        # `_input=None` should become a new empty list
        self.assertEqual(f1._input(), [])

        # Not shared mutable default across instances
        f2 = features.Feature()
        self.assertEqual(f2._input(), [])

        x1 = f1._input()
        x1.append(123)
        self.assertEqual(f1._input(), [123])
        self.assertEqual(f2._input(), [])

        # Custom name override
        f3 = features.Feature(name="CustomName")
        self.assertEqual(f3.node_name, "CustomName")
        self.assertEqual(f3.properties["name"](), "CustomName")

    def test_Feature___call__(self):  # TODO
        pass

    def test_Feature__to_sequential(self):  # TODO
        pass

    def test_Feature__action(self):  # TODO
        pass

    def test_Feature_update(self):  # TODO
        pass

    def test_Feature_add_feature(self):  # TODO
        pass

    def test_Feature_seed(self):  # TODO
        pass

    def test_Feature_bind_arguments(self):  # TODO
        pass

    def test_Feature_plot(self):  # TODO
        pass

    def test_Feature__normalize(self):  # TODO
        pass

    def test_Feature__process_properties(self):  # TODO
        pass

    def test_Feature__activate_sources(self):  # TODO
        pass

    def test_Feature_torch_numpy_get_backend_dtype_to(self):
        feature = features.DummyFeature()

        # numpy() + get_backend() + to() warning normalization
        feature.numpy()
        self.assertEqual(feature.get_backend(), "numpy")
        self.assertEqual(feature.device, "cpu")

        # Requesting a non-CPU device under NumPy should warn and normalize.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            feature.to("cuda")
            self.assertTrue(
                any(issubclass(x.category, UserWarning) for x in w)
            )
            self.assertEqual(feature.device, "cpu")

        if TORCH_AVAILABLE:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                feature.to(torch.device("cuda"))
                self.assertTrue(
                    any(issubclass(x.category, UserWarning) for x in w)
                )
                self.assertEqual(feature.device, "cpu")

        # After the above, ensure NumPy device is CPU as expected.
        self.assertEqual(feature.get_backend(), "numpy")
        self.assertEqual(feature.device, "cpu")

        # dtype() under NumPy
        feature.dtype(
            float="float32",
            int="int16",
            complex="complex64",
            bool="bool",
        )
        self.assertEqual(feature.float_dtype, np.dtype("float32"))
        self.assertEqual(feature.int_dtype, np.dtype("int16"))
        self.assertEqual(feature.complex_dtype, np.dtype("complex64"))
        self.assertEqual(feature.bool_dtype, np.dtype("bool"))

        # torch() + get_backend() + dtype() + to()
        if TORCH_AVAILABLE:
            feature.torch(device=torch.device("cpu"))
            self.assertEqual(feature.get_backend(), "torch")
            self.assertIsInstance(feature.device, torch.device)
            self.assertEqual(feature.device.type, "cpu")

            # dtype resolution should now be torch dtypes
            feature.dtype(float="float64")
            self.assertEqual(feature.float_dtype.name, "float64")

            # Calling to(torch.device("cpu")) under torch should not warn.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                feature.to(torch.device("cpu"))
                self.assertFalse(
                    any(issubclass(x.category, UserWarning) for x in w)
                )
                self.assertEqual(feature.device.type, "cpu")

            # -----------------------------------------------------------------
            # Extra coverage 1: recursive backend switching in a small pipeline
            pipeline = features.Add(b=1) >> features.Add(b=2)

            pipeline.numpy(recursive=True)
            self.assertEqual(pipeline.get_backend(), "numpy")
            self.assertEqual(pipeline.device, "cpu")

            # Ensure dependent features are also converted when recursive=True.
            for dependency in pipeline.recurse_dependencies():
                if isinstance(dependency, features.Feature):
                    self.assertEqual(dependency.get_backend(), "numpy")
                    self.assertEqual(dependency.device, "cpu")

            if TORCH_AVAILABLE:
                pipeline.torch(device=torch.device("cuda"), recursive=True)
                self.assertEqual(pipeline.get_backend(), "torch")
                self.assertIsInstance(pipeline.device, torch.device)
                self.assertEqual(pipeline.device.type, "cuda")

                for dependency in pipeline.recurse_dependencies():
                    if isinstance(dependency, features.Feature):
                        self.assertEqual(dependency.get_backend(), "torch")
                        self.assertIsInstance(dependency.device, torch.device)
                        self.assertEqual(dependency.device.type, "cuda")

            # -----------------------------------------------------------------
            # Extra coverage 2: numpy() resets device to CPU even after non-CPU
            if TORCH_AVAILABLE:
                feature.torch(device=torch.device("cuda"))
                self.assertEqual(feature.get_backend(), "torch")
                self.assertIsInstance(feature.device, torch.device)
                self.assertEqual(feature.device.type, "cuda")

                feature.numpy()
                self.assertEqual(feature.get_backend(), "numpy")
                self.assertEqual(feature.device, "cpu")

            # -----------------------------------------------------------------
            # Extra coverage 3: to("cpu") under NumPy should not warn.
            feature.numpy()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                feature.to("cpu")
                self.assertFalse(
                    any(issubclass(x.category, UserWarning) for x in w)
                )
                self.assertEqual(feature.device, "cpu")

            if TORCH_AVAILABLE:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    feature.to(torch.device("cpu"))
                    self.assertFalse(
                        any(issubclass(x.category, UserWarning) for x in w)
                    )
                    self.assertEqual(feature.device.type, "cpu")

    def test_Feature_batch(self):
        # Single-output case
        feature = features.Value(value=lambda: xp.arange(3))

        # NumPy backend
        feature.numpy()
        batch = feature.batch(batch_size=4)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].shape, (4, 3))
        self.assertEqual(batch[0].dtype, feature.int_dtype)

        # Torch backend
        if TORCH_AVAILABLE:
            feature.torch(device=torch.device("cpu"))
            batch = feature.batch(batch_size=4)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 1)
            self.assertEqual(tuple(batch[0].shape), (4, 3))
            self.assertEqual(str(batch[0].dtype), str(feature.int_dtype))

        # Multi-output case
        multi = features.Value(
            value=lambda: (xp.arange(3), xp.arange(3) + 1),
        )

        # NumPy backend
        multi.numpy()
        batch = multi.batch(batch_size=4)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].shape, (4, 3))
        self.assertEqual(batch[1].shape, (4, 3))
        self.assertEqual(batch[0].dtype, multi.int_dtype)
        self.assertEqual(batch[1].dtype, multi.int_dtype)

        # Torch backend
        if TORCH_AVAILABLE:
            multi.torch(device=torch.device("cpu"))
            batch = multi.batch(batch_size=4)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 2)
            self.assertEqual(tuple(batch[0].shape), (4, 3))
            self.assertEqual(tuple(batch[1].shape), (4, 3))
            self.assertEqual(str(batch[0].dtype), str(multi.int_dtype))
            self.assertEqual(str(batch[1].dtype), str(multi.int_dtype))

        # Scalar-output case
        scalar = features.Value(value=lambda: 1)

        # NumPy backend
        scalar.numpy()
        batch = scalar.batch(batch_size=4)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].shape, (4,))
        self.assertTrue(xp.all(batch[0] == 1))

        # Torch backend
        if TORCH_AVAILABLE:
            scalar.torch(device=torch.device("cpu"))
            batch = scalar.batch(batch_size=4)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 1)
            self.assertEqual(tuple(batch[0].shape), (4,))
            self.assertTrue(bool(xp.all(batch[0] == 1)))

    def test_Feature___getattr__(self):
        feature = features.DummyFeature(value=42, prop="a")

        self.assertIs(feature.value, feature.properties["value"])
        self.assertIs(feature.prop, feature.properties["prop"])

        self.assertEqual(feature.value(), feature.properties["value"]())
        self.assertEqual(feature.prop(), feature.properties["prop"]())

        with self.assertRaises(AttributeError):
            _ = feature.nonexistent

    def test_Feature___iter__and__next__(self):
        # Deterministic value source
        values = iter([0, 1, 2, 3])
        feature = features.Value(value=lambda: next(values))

        # __iter__ should return self
        self.assertIs(iter(feature), feature)

        # __next__ should return successive values
        self.assertEqual(next(feature), 0)
        self.assertEqual(next(feature), 1)

        # Finite iteration using islice (as documented)
        samples = list(itertools.islice(feature, 2))
        self.assertEqual(samples, [2, 3])

    def test_Feature___rshift__and__rrshift__(self):
        # __rshift__: Feature >> Feature
        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Add(b=1)

        pipeline = feature1 >> feature2
        self.assertIsInstance(pipeline, features.Chain)
        self.assertEqual(pipeline(), [2, 3, 4])

        # __rshift__: Feature >> callable
        import numpy as np

        feature = features.Value(value=np.array([1, 2, 3]))
        pipeline = feature >> np.mean
        self.assertIsInstance(pipeline, features.Chain)
        self.assertEqual(pipeline(), 2.0)

        # Python (Feature.__rshift__ returns NotImplemented).
        with self.assertRaises(TypeError):
            _ = feature1 >> "invalid"

    def test_Feature_operators(self):
        # __add__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature + 5
        self.assertEqual(pipeline(), [6, 7, 8])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 + feature2
        self.assertEqual(pipeline(), [4, 4, 4])

        # __radd__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 4 + feature
        self.assertEqual(pipeline(), [5, 6, 7])

        # __sub__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature - 5
        self.assertEqual(pipeline(), [-4, -3, -2])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 - feature2
        self.assertEqual(pipeline(), [-2, 0, 2])

        # __rsub__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 4 - feature
        self.assertEqual(pipeline(), [3, 2, 1])

        # __mul__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature * 5
        self.assertEqual(pipeline(), [5, 10, 15])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 * feature2
        self.assertEqual(pipeline(), [3, 4, 3])

        # __rmul__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 4 * feature
        self.assertEqual(pipeline(), [4, 8, 12])

        # __truediv__
        feature = features.Value(value=[10, 20, 30])
        pipeline = feature / 5
        self.assertEqual(pipeline(), [2.0, 4.0, 6.0])

        feature1 = features.Value(value=[10, 20, 30])
        feature2 = features.Value(value=[5, 4, 3])
        pipeline = feature1 / feature2
        self.assertEqual(pipeline(), [2.0, 5.0, 10.0])

        # __rtruediv__
        feature = features.Value(value=[2, 4, 5])
        pipeline = 10 / feature
        self.assertEqual(pipeline(), [5.0, 2.5, 2.0])

        # __floordiv__
        feature = features.Value(value=[12, 24, 36])
        pipeline = feature // 5
        self.assertEqual(pipeline(), [2, 4, 7])

        feature1 = features.Value(value=[12, 22, 32])
        feature2 = features.Value(value=[5, 4, 3])
        pipeline = feature1 // feature2
        self.assertEqual(pipeline(), [2, 5, 10])

        # __rfloordiv__
        feature = features.Value(value=[3, 6, 7])
        pipeline = 10 // feature
        self.assertEqual(pipeline(), [3, 1, 1])

        # __pow__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature ** 3 
        self.assertEqual(pipeline(), [1, 8, 27])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 ** feature2
        self.assertEqual(pipeline(), [1, 4, 3])

        # __rpow__
        feature = features.Value(value=[2, 3, 4])
        pipeline = 10 ** feature
        self.assertEqual(pipeline(), [100, 1_000, 10_000])

        # __gt__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature > 2 
        self.assertEqual(pipeline(), [False, False, True])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 > feature2
        self.assertEqual(pipeline(), [False, False, True])

        # __rgt__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 2 > feature
        self.assertEqual(pipeline(), [True, False, False])

        # __lt__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature < 2 
        self.assertEqual(pipeline(), [True, False, False])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 < feature2
        self.assertEqual(pipeline(), [True, False, False])

        # __rlt__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 2 < feature
        self.assertEqual(pipeline(), [False, False, True])

        # __le__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature <= 2 
        self.assertEqual(pipeline(), [True, True, False])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 <= feature2
        self.assertEqual(pipeline(), [True, True, False])

        # __rle__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 2 <= feature
        self.assertEqual(pipeline(), [False, True, True])

        # __ge__
        feature = features.Value(value=[1, 2, 3])
        pipeline = feature >= 2 
        self.assertEqual(pipeline(), [False, True, True])

        feature1 = features.Value(value=[1, 2, 3])
        feature2 = features.Value(value=[3, 2, 1])
        pipeline = feature1 >= feature2
        self.assertEqual(pipeline(), [False, True, True])

        # __rge__
        feature = features.Value(value=[1, 2, 3])
        pipeline = 2 >= feature
        self.assertEqual(pipeline(), [True, True, False])

    def test_Feature___xor__(self):
        add_one = features.Add(b=1)

        pipeline = features.Value(value=0) >> (add_one ^ 3)
        self.assertEqual(pipeline.resolve(), 3)

        # Defensive: non-integer repetition should fail.
        with self.assertRaises(ValueError):
            pipeline = add_one ^ 2.5
            pipeline()

    def test_Feature___and__and__rand__(self):
        base = features.Value(value=[1, 2, 3])
        other = features.Value(value=[4, 5])

        # Feature & Feature
        pipeline = base & other
        self.assertEqual(pipeline.resolve(), [1, 2, 3, 4, 5])

        # Feature & value
        pipeline = base & [4, 5]
        self.assertEqual(pipeline.resolve(), [1, 2, 3, 4, 5])

        # Value & Feature (__rand__)
        pipeline = [4, 5] & base
        self.assertEqual(pipeline.resolve(), [4, 5, 1, 2, 3])

        # Chaining still works
        pipeline = (base & [4]) >> features.Stack(value=[6])
        self.assertEqual(pipeline.resolve(), [1, 2, 3, 4, 6])

    def test_Feature___getitem__(self):
        base_feature = features.Value(value=np.array([10, 20, 30]))

        # Constant index
        indexed_feature = base_feature[1]
        self.assertEqual(indexed_feature.resolve(), 20)

        # Negative index
        indexed_feature = base_feature[-1]
        self.assertEqual(indexed_feature.resolve(), 30)

        # Full slice (identity)
        sliced_feature = base_feature[:]
        np.testing.assert_array_equal(
            sliced_feature.resolve(),
            np.array([10, 20, 30]),
        )

        # Tail slice
        sliced_feature = base_feature[1:]
        np.testing.assert_array_equal(
            sliced_feature.resolve(),
            np.array([20, 30]),
        )

        # All-but-last slice
        sliced_feature = base_feature[:-1]
        np.testing.assert_array_equal(
            sliced_feature.resolve(),
            np.array([10, 20]),
        )

        # Strided slice
        sliced_feature = base_feature[::2]
        np.testing.assert_array_equal(
            sliced_feature.resolve(),
            np.array([10, 30]),
        )

        # Check that chaining still works
        pipeline = base_feature[2] >> features.Add(b=5)
        self.assertEqual(pipeline.resolve(), 35)

        # 2D indexing and slicing
        matrix_feature = features.Value(value=np.array([[1, 2, 3], [4, 5, 6]]))

        # 2D index
        indexed_feature = matrix_feature[0, 2]
        self.assertEqual(indexed_feature.resolve(), 3)

        # 2D slice
        sliced_feature = matrix_feature[:, 1:]
        np.testing.assert_array_equal(
            sliced_feature.resolve(),
            np.array([[2, 3], [5, 6]]),
        )

    def test_Feature_basics(self):

        F = features.DummyFeature()
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(F.properties(), {'name': 'DummyFeature'})

        F = features.DummyFeature(a=1, b=2)
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(
            F.properties(),
            {'a': 1, 'b': 2, 'name': 'DummyFeature'},
        )

        F = features.DummyFeature(prop_int=1, prop_bool=True, prop_str="a")
        self.assertIsInstance(F, features.Feature)
        self.assertIsInstance(F.properties, properties.PropertyDict)
        self.assertEqual(
            F.properties(),
            {'prop_int': 1, 'prop_bool': True, 'prop_str': 'a',
             'name': 'DummyFeature'},
        )
        self.assertIsInstance(F.properties["prop_int"](), int)
        self.assertEqual(F.properties["prop_int"](), 1)
        self.assertIsInstance(F.properties["prop_bool"](), bool)
        self.assertEqual(F.properties["prop_bool"](), True)
        self.assertIsInstance(F.properties["prop_str"](), str)
        self.assertEqual(F.properties["prop_str"](), 'a')

    def test_Feature_properties_update_new(self):

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

        prop_dict_with_new = feature.properties.new()
        self.assertNotEqual(prop_dict, prop_dict_with_new)

    def test_Feature_memorized(self):

        list_of_inputs = []

        class ConcreteFeature(features.Feature):
            __distributed__ = False
            def get(self, data, **kwargs):
                list_of_inputs.append(data)
                return data

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

        feature.new()
        self.assertEqual(len(list_of_inputs), 5)

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
            def get(self, data, **kwargs):
                return data

        feature = ConcreteFeature(prop=1)

        self.assertFalse(feature.is_valid())

        feature()
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(1)  # Does not change value.
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(2)  # Changes value.
        self.assertFalse(feature.is_valid())

    def test_Feature_plus_1(self):

        class FeatureAddValue(features.Feature):
            def get(self, data, value_to_add=0, **kwargs):
                data = data + value_to_add
                return data

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureAddValue(value_to_add=2)
        feature = feature1 >> feature2
        feature.update()
        input_data = np.zeros((1, 1))
        output_data = feature.resolve(input_data)
        self.assertEqual(output_data, 3)

    def test_Feature_plus_2(self):

        class FeatureAddValue(features.Feature):
            def get(self, data, value_to_add=0, **kwargs):
                data = data + value_to_add
                return data

        class FeatureMultiplyByValue(features.Feature):
            def get(self, data, value_to_multiply=0, **kwargs):
                data = data * value_to_multiply
                return data

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureMultiplyByValue(value_to_multiply=10)
        input_data = np.zeros((1, 1))

        feature12 = feature1 >> feature2
        feature12.update()
        output_data12 = feature12.resolve(input_data)
        self.assertEqual(output_data12, 10)

        feature21 = feature2 >> feature1
        feature12.update()
        output_data21 = feature21.resolve(input_data)
        self.assertEqual(output_data21, 1)

    def test_Feature_plus_3(self):

        class FeatureAppendImageOfShape(features.Feature):
            __distributed__ = False
            __list_merge_strategy__ = features.MERGE_STRATEGY_APPEND
            def get(self, *args, shape, **kwargs):
                data = np.zeros(shape)
                return data

        feature1 = FeatureAppendImageOfShape(shape=(1, 1))
        feature2 = FeatureAppendImageOfShape(shape=(2, 2))
        feature12 = feature1 >> feature2
        feature12.update()
        output_data = feature12.resolve()
        self.assertIsInstance(output_data, list)
        self.assertIsInstance(output_data[0], np.ndarray)
        self.assertIsInstance(output_data[1], np.ndarray)
        self.assertEqual(output_data[0].shape, (1, 1))
        self.assertEqual(output_data[1].shape, (2, 2))

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

        output = feature()
        self.assertEqual(output, 2)

        feature.update()
        output = feature()
        self.assertEqual(output, 2)

        output = feature.new()
        self.assertEqual(output, 2)

    def test_Feature_repeat(self):

        feature = features.Value(0) >> (features.Add(1) ^ iter(range(10)))

        for n in range(11):
            output = feature.new()
            self.assertEqual(output, np.min([n, 9]))

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

    def test_Feature_nested_Duplicate(self):

        A = features.DummyFeature(
            r=lambda: np.random.randint(10) * 1000,
            total=lambda r: r,
        )
        B = features.DummyFeature(
            a=A.total,
            r=lambda: np.random.randint(10) * 100,
            total=lambda a, r: a + r,
        )
        C = features.DummyFeature(
            b=B.total,
            r=lambda: np.random.randint(10) * 10,
            total=lambda b, r: b + r,
        )
        D = features.DummyFeature(
            c=C.total,
            r=lambda: np.random.randint(10) * 1,
            total=lambda c, r: c + r,
        )

        self.assertEqual(D.total(), A.r() + B.r() + C.r() + D.r())


    def test_propagate_data_to_dependencies(self):
        feature = (
            features.Value(value=np.ones((2, 2)))
            >> features.Add(b=lambda: 1.0)
            >> features.Multiply(b=lambda: 2.0)
        )

        out = feature()  # (1 + 1) * 2 = 4
        np.testing.assert_array_equal(out, 4.0 * np.ones((2, 2)))

        features.propagate_data_to_dependencies(feature, b=3.0)
        out_default = feature()  # (1 + 3) * 3 = 12
        np.testing.assert_array_equal(out_default, 12.0 * np.ones((2, 2)))

        # With _ID
        feature = (
            features.Value(value=np.ones((2, 2)))
            >> features.Add(b=lambda: 1.0)
            >> features.Multiply(b=lambda: 2.0)
        )

        features.propagate_data_to_dependencies(feature, _ID=(1,), b=3.0)

        out_ID_0 = feature(_ID=(0,))  # (1 + 1) * 2 = 4
        np.testing.assert_array_equal(out_ID_0, 4.0 * np.ones((2, 2)))

        out_ID_1 = feature(_ID=(1,))  # (1 + 3) * 3 = 12
        np.testing.assert_array_equal(out_ID_1, 12.0 * np.ones((2, 2)))


    def test_Chain(self):

        class Addition(features.Feature):
            """Simple feature that adds a constant."""
            def get(self, inputs, **kwargs):
                # 'addend' is a property set via self.properties (default: 0).
                return inputs + self.properties.get("addend", 0)()

        class Multiplication(features.Feature):
            """Simple feature that multiplies by a constant."""
            def get(self, inputs, **kwargs):
                # 'multiplier' is a property set via self.properties
                # (default: 1).
                return inputs * self.properties.get("multiplier", 1)()

        A = Addition(addend=10)
        M = Multiplication(multiplier=0.5)

        inputs = np.ones((2, 3))

        chain_AM = features.Chain(A, M)
        self.assertTrue(
            np.array_equal(
                chain_AM(inputs),
                (np.ones((2, 3)) + A.properties["addend"]())
                * M.properties["multiplier"](),
            )
        )
        self.assertTrue(
            np.array_equal(
                chain_AM(inputs),
                (A >> M)(inputs),
            )
        )

        chain_MA = features.Chain(M, A)
        self.assertTrue(
            np.array_equal(
                chain_MA(inputs),
                (np.ones((2, 3)) * M.properties["multiplier"]()
                + A.properties["addend"]()),
            )
        )
        self.assertTrue(
            np.array_equal(
                chain_MA(inputs),
                (M >> A)(inputs),
            )
        )

        if TORCH_AVAILABLE:
            inputs = torch.ones((2, 3))

            chain_AM = features.Chain(A, M)
            self.assertTrue(
                torch.allclose(
                    chain_AM(inputs),
                    (torch.ones((2, 3)) + A.properties["addend"]())
                    * M.properties["multiplier"](),
                )
            )
            self.assertTrue(
                torch.allclose(
                    chain_AM(inputs),
                    (A >> M)(inputs),
                )
            )

            chain_MA = features.Chain(M, A)
            self.assertTrue(
                torch.allclose(
                    chain_MA(inputs),
                    (torch.ones((2, 3)) * M.properties["multiplier"]()
                    + A.properties["addend"]()),
                )
            )
            self.assertTrue(
                torch.allclose(
                    chain_MA(inputs),
                    (M >> A)(inputs),
                )
            )


    def test_DummyFeature(self):
        # DummyFeature properties must be callable and updatable.
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

        # DummyFeature returns input unchanged and supports call syntax.
        feature = features.DummyFeature()
        input_array = np.random.rand(10, 10)
        output_array = feature.get(input_array)
        self.assertIs(output_array, input_array)
        # For callability via __call__ (as per DeepTrack2)
        output_array_call = feature(input_array)
        self.assertIs(output_array_call, input_array)

        # Test with NumPy array
        arr = np.zeros((3, 3))
        self.assertIs(feature.get(arr), arr)
        self.assertIs(feature(arr), arr)

        # Test with list of NumPy arrays
        arr_list = [np.ones((2, 2)), np.zeros((2, 2))]
        self.assertEqual(feature.get(arr_list), arr_list)
        self.assertEqual(feature(arr_list), arr_list)

        # Test with PyTorch
        if TORCH_AVAILABLE:
            # Test with PyTorch tensor
            tensor = torch.ones(4, 4)
            self.assertIs(feature.get(tensor), tensor)
            self.assertIs(feature(tensor), tensor)

            # Test with list of PyTorch tensors
            tensor_list = [torch.zeros(2, 2), torch.ones(2, 2)]
            self.assertEqual(feature.get(tensor_list), tensor_list)
            self.assertEqual(feature(tensor_list), tensor_list)


    def test_Value(self):
        # Scalar value tests
        value = features.Value(value=1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)
        self.assertEqual(value(value=2), 2)
        self.assertEqual(value(), 2)
        self.assertEqual(value.value(), 2)

        value = features.Value(value=lambda: 1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)
        self.assertNotEqual(value(value=lambda: 2), 2)
        self.assertNotEqual(value(), 2)
        self.assertNotEqual(value.value(), 2)

        # NumPy array value tests
        arr = np.arange(4)
        value_arr = features.Value(value=arr)
        self.assertTrue(np.array_equal(value_arr(), arr))
        self.assertTrue(np.array_equal(value_arr.value(), arr))
        # Override with a new array
        override_arr = np.array([10, 20, 30, 40])
        self.assertTrue(
            np.array_equal(value_arr(value=override_arr), override_arr)
        )
        self.assertTrue(np.array_equal(value_arr(), override_arr))
        self.assertTrue(np.array_equal(value_arr.value(), override_arr))

        # PyTorch tensor value tests
        if TORCH_AVAILABLE:
            tensor = torch.tensor([1., 2., 3.])
            value_tensor = features.Value(value=tensor)
            self.assertTrue(torch.equal(value_tensor(), tensor))
            self.assertTrue(torch.equal(value_tensor.value(), tensor))
            # Override with a new tensor
            override_tensor = torch.tensor([10., 20., 30.])
            self.assertTrue(torch.equal(
                value_tensor(value=override_tensor), override_tensor
            ))
            self.assertTrue(torch.equal(value_tensor(), override_tensor))
            self.assertTrue(torch.equal(
                value_tensor.value(), override_tensor
            ))


    def test_ArithmeticOperationFeature(self):
        # Basic addition with lists
        addition_feature = features.ArithmeticOperationFeature(
            operator.add, b=10,
        )
        input_values = [1, 2, 3, 4]
        expected_output = [11, 12, 13, 14]
        output = addition_feature(input_values)
        self.assertEqual(output, expected_output)

        # Scalar input and scalar value
        output = addition_feature(5)
        self.assertEqual(output, 15)

        # List input, scalar value (broadcast)
        input_values = [10, 20, 30]
        output = addition_feature(input_values)
        self.assertEqual(output, [20, 30, 40])

        # List input, list value (same length)
        addition_feature = features.ArithmeticOperationFeature(
            operator.add, b=[1, 2, 3],
        )
        input_values = [10, 20, 30]
        self.assertEqual(addition_feature(input_values), [11, 22, 33])

        # List input, list value (different lengths, value list cycles)
        addition_feature = features.ArithmeticOperationFeature(
            operator.add, b=[1, 2],
        )
        input_values = [10, 20, 30, 40, 50]
        # value cycles as 1,2,1,2,1
        self.assertEqual(addition_feature(input_values), [11, 22, 31, 42, 51])

        # NumPy array input, scalar value
        addition_feature = features.ArithmeticOperationFeature(
            operator.add, b=5,
        )
        arr = np.array([1, 2, 3])
        self.assertEqual(addition_feature(arr.tolist()), [6, 7, 8])

        # NumPy array input, NumPy array value
        addition_feature = features.ArithmeticOperationFeature(
            operator.add, b=[4, 5, 6],
        )
        arr_input = [
            np.array([1, 2]), np.array([3, 4]), np.array([5, 6]),
        ]
        arr_value = [
            np.array([10, 20]), np.array([30, 40]), np.array([50, 60]),
        ]
        feature = features.ArithmeticOperationFeature(
            lambda a, b: np.add(a, b), b=arr_value,
        )
        for output, expected in zip(
            feature(arr_input),
            [np.array([11, 22]), np.array([33, 44]), np.array([55, 66])],
        ):
            self.assertTrue(np.array_equal(output, expected))

        # PyTorch tensor input (if available)
        if TORCH_AVAILABLE:
            addition_feature = features.ArithmeticOperationFeature(
                lambda a, b: a + b, b=5,
            )
            tensors = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
            expected = [torch.tensor(6), torch.tensor(7), torch.tensor(8)]
            output = addition_feature(tensors)
            for out, exp in zip(output, expected):
                self.assertTrue(torch.equal(out, exp))

            # Tensor input, tensor value (elementwise)
            t_input = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
            t_value = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0])]
            feature = features.ArithmeticOperationFeature(
                lambda a, b: a + b, b=t_value,
            )
            for output, expected in zip(
                feature(t_input),
                [torch.tensor([11.0, 22.0]), torch.tensor([33.0, 44.0])],
            ):
                self.assertTrue(torch.equal(output, expected))


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


    def test_Equals(self):  # TODO
        """
        Important Notes
        ---------------
        - Unlike other arithmetic operators, `Equals` does not define `__eq__` 
          (`==`) and `__req__` (`==`) in `DeepTrackNode` and `Feature`, as this 
          would affect Python’s built-in identity comparison.
        - This means that the standard `==` operator is overloaded only for 
          expressions involving `Feature` instances but not for comparisons 
          involving regular Python objects.
        - Always use `>>` to apply `Equals` correctly in a feature chain.
        """

        equals_feature = features.Equals(b=2)
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
                *(
                    a["value"]
                    if isinstance(a["value"], list)
                    else [a["value"]]
                ),
                *(
                    b["value"]
                    if isinstance(b["value"], list)
                    else [b["value"]]
                ),
            ],
            operator.__and__,
        )

        # Stack scalar with scalar
        feature = features.Stack(value=2)
        result = feature(1)
        self.assertEqual(result, [1, 2])

        # Stack scalar with list
        feature = features.Stack(value=[3, 4])
        result = feature(2)
        self.assertEqual(result, [2, 3, 4])

        # Stack list with scalar
        feature = features.Stack(value=5)
        result = feature([1, 2, 3])
        self.assertEqual(result, [1, 2, 3, 5])

        # Stack list with list
        feature = features.Stack(value=[4, 5])
        result = feature([1, 2, 3])
        self.assertEqual(result, [1, 2, 3, 4, 5])

        # Stack with empty lists
        feature = features.Stack(value=[])
        result = feature([1, 2])
        self.assertEqual(result, [1, 2])

        feature = features.Stack(value=[1, 2])
        result = feature([])
        self.assertEqual(result, [1, 2])

        # Stack using Value feature
        pipeline = features.Value([1, 2]) >> features.Stack(value=features.Value([3, 4]))
        result = pipeline()
        self.assertEqual(result, [1, 2, 3, 4])

        # Stack using & operator (Value & list)
        pipeline = features.Value([1, 2]) & [3, 4]
        self.assertEqual(pipeline.resolve(), [1, 2, 3, 4])

        # Stack using & operator (list & Value)
        pipeline = [3, 4] & features.Value([1, 2])
        self.assertEqual(pipeline.resolve(), [3, 4, 1, 2])

        # Stack NumPy arrays
        arr1 = np.array([1, 2])
        arr2 = np.array([3, 4])
        feature = features.Stack(value=arr2)
        result = feature(arr1)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.array_equal(result[0], arr1))
        self.assertTrue(np.array_equal(result[1], arr2))

        # Stack PyTorch tensors
        if TORCH_AVAILABLE:
            t1 = torch.tensor([1, 2])
            t2 = torch.tensor([3, 4])
            feature = features.Stack(value=t2)
            result = feature(t1)
            self.assertEqual(len(result), 2)
            self.assertTrue(torch.equal(result[0], t1))
            self.assertTrue(torch.equal(result[1], t2))


    def test_Arguments(self):  # TODO
        from tempfile import NamedTemporaryFile
        from PIL import Image as PIL_Image
        import os 

        # Create a temporary test image.
        test_image_array = (np.ones((50, 50)) * 128).astype(np.uint8)
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
            PIL_Image.fromarray(test_image_array).save(temp_png.name)

        try:  # Ensure removal of test image.
            # Test pipeline behavior when toggling `is_label`.
            arguments = features.Arguments(is_label=False)
            image_pipeline = (
                features.LoadImage(path=temp_png.name)
                >> Gaussian(sigma=(1 - arguments.is_label) * 5)
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise

            # Test pipeline behavior with dynamically computed sigma.
            arguments = features.Arguments(is_label=False)
            image_pipeline = (
                features.LoadImage(path=temp_png.name)
                >> Gaussian(
                    is_label=arguments.is_label,
                    sigma=lambda is_label: 0 if is_label else 5,
                )
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise

            # Test passing arguments dynamically using **arguments.properties.
            arguments = features.Arguments(is_label=False, noise_sigma=5)
            image_pipeline = (
                features.LoadImage(path=temp_png.name) >>
                Gaussian(
                    sigma=lambda is_label, noise_sigma:
                        0 if is_label else noise_sigma,
                    **arguments.properties,
                )
            )
            image_pipeline.bind_arguments(arguments)

            # Test noisy image
            image = image_pipeline()
            self.assertGreater(image.std(), 0)  # Expecting noise around 5

            # Test raw image with `is_label=True`
            image = image_pipeline(is_label=True)
            self.assertAlmostEqual(image.std(), 0.0, places=3)  # No noise

        except Exception:
            raise
        finally:
            if os.path.exists(temp_png.name):
                os.remove(temp_png.name)

    def test_Arguments_feature_passing(self):  # TODO
        # Tests that arguments are correctly passed and updated.

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
            p2=lambda p1: p1 + "baz",  # "foobaz"
        )

        # Second feature dependent on the first
        f2 = features.DummyFeature(
            p1=f1.p2,  # Should be "foobaz"
            p2=arguments.d,  # Random value
        )

        # Assertions
        self.assertEqual(f1.properties["p1"](), "foo")  # Check that p1 is set
                                                        # correctly
        self.assertEqual(f1.properties["p2"](), "foobaz")  # Check lambda
                                                           # evaluation
        self.assertEqual(f2.properties["p1"](), "foobaz")  # Check dependency
                                                           # resolution

        # Ensure p2 in f2 is a valid float between 0 and 1
        self.assertTrue(0 <= f2.properties["p2"]() <= 1)

        # Ensure `c` was computed correctly
        self.assertEqual(arguments.c(), "foobar")  # Should concatenate
                                                   # "foo" + "bar"

        # Test that d is dynamic (generates new values)
        first_d = arguments.d.update()()
        second_d = arguments.d.update()()
        self.assertNotEqual(first_d, second_d)  # Check that values change

    def test_Arguments_binding(self):  # TODO
        # Create a dynamic argument container
        arguments = features.Arguments(x=10)

        # Create a simple pipeline: Value(100) + x + 1
        pipeline = (
            features.Value(100)
            >> features.Add(b=arguments.x)
            >> features.Add(1)
        )

        # Evaluate pipeline with default x=10
        result = pipeline()
        self.assertEqual(result, 111)  # 100 + 10 + 1

        result_no_binding = pipeline(x=20)
        self.assertEqual(result_no_binding, 111)  # 100 + 10 + 1

        # Bind the arguments to the pipeline
        pipeline.bind_arguments(arguments)

        # Override x at runtime to 20
        result_binding = pipeline(x=20)
        self.assertEqual(result_binding, 121)  # 100 + 20 + 1


    def test_Probability(self):  # TODO
        # Set seed for reproducibility of random trials
        np.random.seed(42)

        input_image = np.ones((5, 5))
        add_feature = features.Add(b=2)

        # Helper: Check if feature was applied
        def is_transformed(output):
            return np.array_equal(output, input_image + 2)

        # 1. Test probabilistic application over many runs
        probabilistic_feature = features.Probability(
            feature=add_feature,
            probability=0.7
        )

        applied_count = 0
        total_runs = 300

        for _ in range(total_runs):
            output_image = probabilistic_feature.update().resolve(input_image)
            if is_transformed(output_image):
                applied_count += 1
            else:
                self.assertTrue(np.array_equal(output_image, input_image))

        observed_probability = applied_count / total_runs
        self.assertTrue(0.65 <= observed_probability <= 0.75,
                        f"Observed probability: {observed_probability}")

        # 2. Edge case: probability = 0 (feature should never apply)
        never_applied = features.Probability(feature=add_feature,
                                             probability=0.0)
        output = never_applied.update().resolve(input_image)
        self.assertTrue(np.array_equal(output, input_image))

        # 3. Edge case: probability = 1 (feature should always apply)
        always_applied = features.Probability(feature=add_feature,
                                              probability=1.0)
        output = always_applied.update().resolve(input_image)
        self.assertTrue(is_transformed(output))

        # 4. Cached behavior: result is the same without update()
        cached_feature = features.Probability(feature=add_feature,
                                              probability=1.0)
        output_1 = cached_feature.update().resolve(input_image)
        output_2 = cached_feature.resolve(input_image)  # same random number
        self.assertTrue(np.array_equal(output_1, output_2))

        # 5. Manual override: force behavior using random_number
        manual = features.Probability(feature=add_feature, probability=0.5)

        # Should NOT apply (0.9 > 0.5)
        output = manual.resolve(input_image, random_number=0.9)
        self.assertTrue(np.array_equal(output, input_image))

        # Should apply (0.1 < 0.5)
        output = manual.resolve(input_image, random_number=0.1)
        self.assertTrue(is_transformed(output))


    def test_Repeat(self):
        # Define a simple feature and pipeline
        add_ten = features.Add(b=10)
        pipeline = features.Repeat(add_ten, N=3)

        input_data = [1, 2, 3]
        expected_output = [31, 32, 33]

        # Test standard Repeat behavior
        output_data = pipeline.resolve(input_data)
        self.assertEqual(output_data, expected_output)

        # Test shorthand syntax (^) produces same result
        pipeline_shorthand = features.Add(b=10) ^ 3
        output_data_shorthand = pipeline_shorthand.resolve(input_data)
        self.assertEqual(output_data_shorthand, expected_output)

        # Test dynamic override of N
        output_override = pipeline(input_data, N=2)
        self.assertEqual(output_override, [21, 22, 23])


    def test_Combine(self):  # TODO

        noise_feature = Gaussian(mu=0, sigma=2)
        add_feature = features.Add(b=10)
        combined_feature = features.Combine([noise_feature, add_feature])

        input_image = np.ones((10, 10))
        output_list = combined_feature.resolve(input_image)

        self.assertTrue(isinstance(output_list, list))
        self.assertTrue(len(output_list) == 2)

        for output in output_list:
            self.assertTrue(output.shape == input_image.shape)

        noisy_image = output_list[0]
        added_image = output_list[1]

        self.assertFalse(np.all(noisy_image == 1))
        self.assertTrue(np.allclose(added_image, input_image + 10))


    def test_Slice_constant(self):
        inputs = np.arange(9).reshape((3, 3))

        A = features.DummyFeature()

        A0 = A[0]
        a0 = A0.resolve(inputs)
        self.assertEqual(a0.tolist(), inputs[0].tolist())

        A1 = A[1]
        a1 = A1.resolve(inputs)
        self.assertEqual(a1.tolist(), inputs[1].tolist())

        A22 = A[2, 2]
        a22 = A22.resolve(inputs)
        self.assertEqual(a22, inputs[2, 2])

        A12 = A[1, lambda: -1]
        a12 = A12.resolve(inputs)
        self.assertEqual(a12, inputs[1, -1])

    def test_Slice_colon(self):
        inputs = np.arange(16).reshape((4, 4))

        A = features.DummyFeature()

        A0 = A[0, :1]
        a0 = A0.resolve(inputs)
        self.assertEqual(a0.tolist(), inputs[0, :1].tolist())

        A1 = A[1, lambda: 0 : lambda: 4 : lambda: 2]
        a1 = A1.resolve(inputs)
        self.assertEqual(a1.tolist(), inputs[1, 0:4:2].tolist())

        A2 = A[lambda: slice(0, 4, 1), 2]
        a2 = A2.resolve(inputs)
        self.assertEqual(a2.tolist(), inputs[:, 2].tolist())

        A3 = A[lambda: 0 : lambda: 2, :]
        a3 = A3.resolve(inputs)
        self.assertEqual(a3.tolist(), inputs[0:2, :].tolist())

    def test_Slice_ellipse(self):
        inputs = np.arange(16).reshape((4, 4))

        A = features.DummyFeature()

        A0 = A[..., :1]
        a0 = A0.resolve(inputs)
        self.assertEqual(a0.tolist(), inputs[..., :1].tolist())

        A1 = A[..., lambda: 0 : lambda: 4 : lambda: 2]
        a1 = A1.resolve(inputs)
        self.assertEqual(a1.tolist(), inputs[..., 0:4:2].tolist())

        A2 = A[lambda: slice(0, 4, 1), ...]
        a2 = A2.resolve(inputs)
        self.assertEqual(a2.tolist(), inputs[:, ...].tolist())

        A3 = A[lambda: 0 : lambda: 2, lambda: ...]
        a3 = A3.resolve(inputs)
        self.assertEqual(a3.tolist(), inputs[0:2, ...].tolist())

    def test_Slice_static_dynamic(self):
        inputs = np.arange(27).reshape((3, 3, 3))
        expected_output = inputs[:, 1:2, ::-2]

        feature = features.DummyFeature()

        static_slicing = feature[:, 1:2, ::-2]
        static_output = static_slicing.resolve(inputs)
        self.assertTrue(np.array_equal(static_output, expected_output))

        dynamic_slicing = feature >> features.Slice(
            slices=(slice(None), slice(1, 2), slice(None, None, -2))
        )
        dinamic_output = dynamic_slicing.resolve(inputs)
        self.assertTrue(np.array_equal(dinamic_output, expected_output))


    def test_Bind(self):  # TODO

        value = features.Value(
            value=lambda input_value: input_value,
            input_value=10,
        )
        pipeline = (value + 10) / value
        res = pipeline.update().resolve()
        self.assertEqual(res, 2)

        pipeline_with_small_input = features.Bind(pipeline, input_value=1)
        res = pipeline_with_small_input.update().resolve()
        self.assertEqual(res, 11)

        with self.assertWarns(DeprecationWarning):
            res = pipeline_with_small_input.update(input_value=10).resolve()
            self.assertEqual(res, 11)

    def test_Bind_gaussian_noise(self):  # TODO
        # Define the Gaussian noise feature and bind its properties
        gaussian_noise = Gaussian()
        bound_feature = features.Bind(gaussian_noise, mu=-5, sigma=2)

        # Create the input image
        input_image = np.zeros((128, 128))

        # Resolve the feature to get the output image
        output_image = bound_feature.resolve(input_image)

        # Calculate the mean and standard deviation of the output
        output_mean = np.mean(output_image)
        output_std = np.std(output_image)

        # Assert that the mean and standard deviation are close to the bound values
        self.assertAlmostEqual(output_mean, -5, delta=0.2)
        self.assertAlmostEqual(output_std, 2, delta=0.2)


    def test_BindResolve(self):  # TODO

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

        with self.assertWarns(DeprecationWarning):
            res = pipeline_with_small_input.update(input_value=10).resolve()
            self.assertEqual(res, 11)


    def test_BindUpdate(self):  # TODO
        value = features.Value(
            value=lambda input_value: input_value, 
            input_value=10,
            )
        value = features.Value(
            value=lambda input_value: input_value, 
            input_value=10,
            )
        pipeline = (value + 10) / value

        with self.assertWarns(DeprecationWarning):
            pipeline_with_small_input = features.BindUpdate(
                pipeline,
                input_value=1,
            )

        res = pipeline.update().resolve()
        self.assertEqual(res, 2)

        res = pipeline_with_small_input.update().resolve()
        self.assertEqual(res, 11)

        with self.assertWarns(DeprecationWarning):
            res = pipeline_with_small_input.update(input_value=10).resolve()
            self.assertEqual(res, 11)

    def test_BindUpdate_gaussian_noise(self):  # TODO
        # Define the Gaussian noise feature and bind its properties
        gaussian_noise = Gaussian()
        with self.assertWarns(DeprecationWarning):
            bound_feature = features.BindUpdate(gaussian_noise, mu=5, sigma=3)

        # Create the input image
        input_image = np.zeros((128, 128))

        # Resolve the feature to get the output image
        output_image = bound_feature.resolve(input_image)

        # Calculate the mean and standard deviation of the output
        output_mean = np.mean(output_image)
        output_std = np.std(output_image)

        # Assert mean and standard deviation close to the bound values
        self.assertAlmostEqual(output_mean, 5, delta=0.5)
        self.assertAlmostEqual(output_std, 3, delta=0.5)


    def test_ConditionalSetProperty(self):  # TODO

        # Set up a Gaussian feature and a test image before each test.
        gaussian_noise = Gaussian(sigma=0)
        image = np.ones((128, 128))

        # Test that sigma is correctly applied when condition is a boolean.
        with self.assertWarns(DeprecationWarning):
            conditional_feature = features.ConditionalSetProperty(
                gaussian_noise, sigma=5,
            )

        # Test with condition met (should apply sigma=5)
        noisy_image = conditional_feature(image, condition=True)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Test without condition met (should apply sigma=0)
        clean_image = conditional_feature.update()(image, condition=False)
        self.assertEqual(clean_image.std(), 0)

        # Test sigma is correctly applied when condition is string property.
        with self.assertWarns(DeprecationWarning):
            conditional_feature = features.ConditionalSetProperty(
                gaussian_noise, sigma=5, condition="is_noisy",
            )

        # Test with condition met (should apply sigma=5)
        noisy_image = conditional_feature(image, is_noisy=True)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Test without condition met (should apply sigma=0)
        clean_image = conditional_feature.update()(image, is_noisy=False)
        self.assertEqual(clean_image.std(), 0)


    def test_ConditionalSetFeature(self):  # TODO
        # Set up Gaussian noise features and test image before each test.
        true_feature = Gaussian(sigma=0)    # Clean image (no noise)
        false_feature = Gaussian(sigma=5)   # Noisy image (sigma=5)
        image = np.ones((512, 512))

        # Test using a direct boolean condition.
        with self.assertWarns(DeprecationWarning):
            conditional_feature = features.ConditionalSetFeature(
                on_true=true_feature,
                on_false=false_feature,
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

        # Test using a string-based condition.
        with self.assertWarns(DeprecationWarning):
            conditional_feature = features.ConditionalSetFeature(
                on_true=true_feature,
                on_false=false_feature,
                condition="is_noisy",
            )

        # Condition is False (sigma=5)
        noisy_image = conditional_feature(image, is_noisy=False)
        self.assertAlmostEqual(noisy_image.std(), 5, delta=0.5)

        # Condition is True (sigma=0)
        clean_image = conditional_feature(image, is_noisy=True)
        self.assertEqual(clean_image.std(), 0)


    def test_Lambda_dependence(self):  # TODO
        # Without Lambda
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a"
                             else (A.b() if key == "b"
                                   else A.c()),
        )

        B.update()
        self.assertEqual(B.prop(), 1)

        B.key.set_value("b")
        self.assertEqual(B.prop(), 2)

        B.key.set_value("c")
        self.assertEqual(B.prop(), 3)

        B.key.set_value("a")
        self.assertEqual(B.prop(), 1)

        # With Lambda
        A = features.DummyFeature(a=1, b=2, c=3)

        def func_factory(key="a"):
            def func(A):
                return A.a() if key == "a" else (A.b() if key == "b" else A.c())
            return func

        B = features.Lambda(function=func_factory, key="a")

        B.update()
        self.assertEqual(B(A), 1)

        B.key.set_value("b")
        self.assertEqual(B(A), 2)

        B.key.set_value("c")
        self.assertEqual(B(A), 3)

        B.key.set_value("a")
        self.assertEqual(B(A), 1)

    def test_Lambda_dependence_twice(self):  # TODO
        # Without Lambda
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a"
                             else (A.b() if key == "b"
                                   else A.c()),
            prop2=lambda prop: prop * 2,
        )

        B.update()
        self.assertEqual(B.prop2(), 2)

        B.key.set_value("b")
        self.assertEqual(B.prop2(), 4)

        B.key.set_value("c")
        self.assertEqual(B.prop2(), 6)

        B.key.set_value("a")
        self.assertEqual(B.prop2(), 2)

    def test_Lambda_dependence_other_feature(self):  # TODO

        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a() if key == "a"
                             else (A.b() if key == "b"
                                   else A.c()),
            prop2=lambda prop: prop * 2,
        )

        C = features.DummyFeature(B_prop=B.prop2,
                                  prop=lambda B_prop: B_prop * 2)

        C.update()
        self.assertEqual(C.prop(), 4)

        B.key.set_value("b")
        self.assertEqual(C.prop(), 8)

        B.key.set_value("c")
        self.assertEqual(C.prop(), 12)

        B.key.set_value("a")
        self.assertEqual(C.prop(), 4)

    def test_Lambda_scaling(self):  # TODO
        def scale_function_factory(scale=2):
            def scale_function(image):
                return image * scale
            return scale_function

        lambda_feature = features.Lambda(
            function=scale_function_factory,
            scale=5,
        )
        input_image = np.ones((5, 5))
        output_image = lambda_feature.resolve(input_image)
        self.assertTrue(np.array_equal(output_image, np.ones((5, 5)) * 5))

        lambda_feature = features.Lambda(
            function=scale_function_factory,
            scale=3,
        )
        output_image = lambda_feature.resolve(input_image)
        self.assertTrue(np.array_equal(output_image, np.ones((5, 5)) * 3))


    def test_Merge(self):  # TODO

        def merge_function_factory():
            def merge_function(images):
                return np.mean(np.stack(images), axis=0)
            return merge_function

        merge_feature = features.Merge(function=merge_function_factory)

        image_1 = np.ones((5, 5)) * 2
        image_2 = np.ones((5, 5)) * 4
        output_image = merge_feature.resolve([image_1, image_2])
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                output_image, np.ones((5, 5)) * 3,
            )
        )

        image_1 = np.ones((5, 5)) * 2
        image_2 = np.ones((3, 3)) * 4
        with self.assertRaises(ValueError):
            merge_feature.resolve([image_1, image_2])

        image_1 = np.ones((5, 5)) * 2
        output_image = merge_feature.resolve([image_1])
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                output_image, image_1,
            )
        )


    def test_OneOf(self):  # TODO
        # Set up the features and input image for testing.
        feature_1 = features.Add(b=10)
        feature_2 = features.Multiply(b=2)
        input_image = np.array([1, 2, 3])

        # Test that OneOf applies one of the features randomly.
        one_of_feature = features.OneOf([feature_1, feature_2])
        output_image = one_of_feature.resolve(input_image)

        # The output should either be:
        # - self.input_image + 10 (if feature_1 is chosen)
        # - self.input_image * 2  (if feature_2 is chosen)
        expected_outputs = [
            input_image + 10,
            input_image * 2,
        ]
        self.assertTrue(
            any(
                np.array_equal(output_image, expected) 
                for expected in expected_outputs
            )
        )

        # Test that OneOf applies the selected feature when `key` is provided.
        controlled_feature = features.OneOf([feature_1, feature_2], key=0)
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image + 10
        self.assertTrue(np.array_equal(output_image, expected_output))

        controlled_feature = features.OneOf([feature_1, feature_2], key=1)
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image * 2
        self.assertTrue(np.array_equal(output_image, expected_output))

    def test_OneOf_list(self):  # TODO

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

    def test_OneOf_tuple(self):  # TODO

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

    def test_OneOf_set(self):  # TODO

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


    def test_OneOfDict_basic(self):  # TODO

        values = features.OneOfDict(
            {
                "1": features.Value(1),
                "2": features.Value(2),
                "3": features.Value(3),
            }
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

    def test_OneOfDict(self):  # TODO
        features_dict = {
            "add": features.Add(b=10),
            "multiply": features.Multiply(b=2),
        }
        one_of_dict_feature = features.OneOfDict(features_dict)

        input_image = np.array([1, 2, 3])

        # Test OneOfDict selects a feature randomly and applies it correctly.
        output_image = one_of_dict_feature.resolve(input_image)
        expected_outputs = [
            input_image + 10,  # "add"
            input_image * 2,  # "multiply"
        ]
        self.assertTrue(any(np.array_equal(output_image, expected)
                            for expected in expected_outputs))

        # Test OneOfDict selects the correct feature when a key is specified.
        controlled_feature = features.OneOfDict(features_dict, key="add")
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image + 10
        self.assertTrue(np.array_equal(output_image, expected_output))

        controlled_feature = features.OneOfDict(features_dict, key="multiply")
        output_image = controlled_feature.resolve(input_image)
        expected_output = input_image * 2
        self.assertTrue(np.array_equal(output_image, expected_output))


    def test_LoadImage(self):  # TODO
        return

        from tempfile import NamedTemporaryFile
        from PIL import Image as PIL_Image
        import os

        # Create temporary image files in multiple formats for testing.
        test_image_array = (np.random.rand(50, 50) * 255).astype(np.uint8)

        try:
            with NamedTemporaryFile(suffix=".npy", delete=False) as temp_npy:
                pass
            np.save(temp_npy.name, test_image_array)
                # npy_filename = temp_npy.name

            with NamedTemporaryFile(suffix=".npy", delete=False) as temp_npy2:
                pass
            np.save(temp_npy2.name, test_image_array)

            with NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
                PIL_Image.fromarray(test_image_array).save(temp_png.name)
                # png_filename = temp_png.name

            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_jpg:
                PIL_Image.fromarray(test_image_array).convert("RGB") \
                    .save(temp_jpg.name)
                # jpg_filename = temp_jpg.name

            # Test loading a .npy file.
            load_feature = features.LoadImage(path=temp_npy.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2],
                             test_image_array.shape[:2])

            # Test loading a .png file.
            load_feature = features.LoadImage(path=temp_png.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2],
                             test_image_array.shape[:2])

            # Test loading a .jpg file.
            load_feature = features.LoadImage(path=temp_jpg.name)
            loaded_image = load_feature.resolve()
            self.assertEqual(loaded_image.shape[:2],
                             test_image_array.shape[:2])

            # Test loading an image and converting it to grayscale.
            load_feature = features.LoadImage(path=temp_png.name,
                                              to_grayscale=True)
            # loaded_image = load_feature.resolve()  # TODO Check this
            # self.assertEqual(loaded_image.shape[-1], 1)

            # Test ensuring a minimum number of dimensions.
            load_feature = features.LoadImage(path=temp_png.name, ndim=4)
            loaded_image = load_feature.resolve()
            self.assertGreaterEqual(len(loaded_image.shape), 4)

            # Test loading a list of images
            load_feature = features.LoadImage(
                path=[temp_npy.name, temp_npy2.name], as_list=True
            )
            loaded_list = load_feature.resolve()
            self.assertIsInstance(loaded_list, list)
            self.assertEqual(len(loaded_list), 2)

            for img in loaded_list:
                self.assertTrue(isinstance(img, np.ndarray))

            # Test loading a random image from a list of images
            load_feature = features.LoadImage(
                path=[temp_npy.name, temp_npy2.name],
                ndim=4,
                as_list=True,
                get_one_random=True,
            )
            loaded_image = load_feature.resolve()
            self.assertTrue(
                np.allclose(
                    loaded_image[:, :, 0, 0], test_image_array, rtol=1.e-3
                )
            )
            self.assertEqual(loaded_image.shape, (50, 50, 1, 1))

            import gc
            gc.collect()

            # Test loading an image as a torch tensor.
            if TORCH_AVAILABLE:
                load_feature = features.LoadImage(path=temp_png.name)
                load_feature.torch()
                loaded_image = load_feature.resolve()
                self.assertIsInstance(loaded_image, torch.Tensor)
                self.assertEqual(
                    loaded_image.shape[:2], test_image_array.shape
                )

                loaded_image_np = loaded_image.numpy()
                self.assertTrue(
                    np.allclose(
                        test_image_array, loaded_image_np[:, :, 0], rtol=1.e-3
                    )
                )

        finally:
            for file in [
                temp_npy.name,
                temp_png.name,
                temp_jpg.name,
                temp_npy2.name
            ]:
                os.remove(file)


    def test_AsType(self):  # TODO

        # Test for Numpy arrays.
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

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_image_torch = torch.tensor([1.5, 2.5, 3.5])

            data_types_torch = [
                "float64",
                "int32",
                "int16",
                "uint8",
                "int8",
                "torch.float64",
                "torch.int32",
            ]

            torch_dtypes_map = {
                "float64": torch.float64,
                "int32": torch.int32,
                "int16": torch.int16,
                "uint8": torch.uint8,
                "int8": torch.int8,
                "torch.float64": torch.float64,
                "torch.int32": torch.int32,
            }

            for dtype in data_types_torch:
                astype_feature = features.AsType(dtype=dtype)
                output_image = astype_feature.get(
                    input_image_torch, dtype=dtype
                )
                expected_dtype = torch_dtypes_map[dtype]
                self.assertEqual(output_image.dtype, expected_dtype)

                # Additional check for specific behavior of integers.
                if expected_dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.uint8,
                ]:
                    # Verify that fractional parts are truncated
                    expected = torch.tensor([1, 2, 3], dtype=expected_dtype)
                    self.assertTrue(torch.equal(output_image, expected))


    def test_ChannelFirst2d(self):  # TODO

        with self.assertWarns(DeprecationWarning):
            channel_first_feature = features.ChannelFirst2d()

        # Numpy shapes
        input_image = np.zeros((10, 20, 1))
        output_image = channel_first_feature.get(input_image, axis=-1)
        self.assertEqual(output_image.shape, (1, 10, 20))

        input_image = np.zeros((10, 20, 3))
        output_image = channel_first_feature.get(input_image, axis=-1)
        self.assertEqual(output_image.shape, (3, 10, 20))

        # Numpy values
        input_image = np.array([[[1, 2, 3], [4, 5, 6]]])
        output_image = channel_first_feature.get(input_image, axis=-1)
        self.assertEqual(output_image.shape, (3, 1, 2))
        np.testing.assert_array_equal(output_image, np.moveaxis(input_image, -1, 0))

        if TORCH_AVAILABLE:
            # Torch shapes
            input_image = torch.zeros(10, 20)
            output_image = channel_first_feature.get(input_image, axis=-1)
            self.assertEqual(tuple(output_image.shape), (1, 10, 20))

            input_image = torch.zeros(10, 20, 3)
            output_image = channel_first_feature.get(input_image, axis=-1)
            self.assertEqual(tuple(output_image.shape), (3, 10, 20))

            # Torch values
            input_image = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
            output_image = channel_first_feature.get(input_image, axis=-1)
            self.assertEqual(output_image.shape, (3, 1, 2))
            self.assertTrue(torch.equal(output_image, input_image.permute(2, 0, 1)))


    def test_Store(self):  # TODO
        value_feature = features.Value(lambda: np.random.rand())

        store_feature = features.Store(feature=value_feature, key="example")

        output = store_feature(None, key="example", replace=False)

        value_feature.update()
        cached_output = store_feature(None, key="example", replace=False)
        self.assertEqual(cached_output, output)
        self.assertNotEqual(cached_output, value_feature())

        value_feature.update()
        cached_output = store_feature(None, key="example", replace=True)
        self.assertNotEqual(cached_output, output)
        self.assertEqual(cached_output, value_feature())

        if TORCH_AVAILABLE:

            value_feature = features.Value(lambda: torch.rand(1))

            store_feature = features.Store(
                feature=value_feature, key="example"
            )

            output = store_feature(None, key="example", replace=False)

            value_feature.update()
            cached_output = store_feature(None, key="example", replace=False)
            torch.testing.assert_close(cached_output, output)
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(cached_output, value_feature())

            value_feature.update()
            cached_output = store_feature(None, key="example", replace=True)
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(cached_output, output)
            torch.testing.assert_close(cached_output, value_feature())


    def test_Squeeze(self):  # TODO
        ### Test with NumPy array
        input_image = np.array([[[[3], [2], [1]]], [[[1], [2], [3]]]])
        # shape: (2, 1, 3, 1)

        # Squeeze axis 1
        squeeze_feature = features.Squeeze(axis=1)
        output_image = squeeze_feature(input_image)
        self.assertEqual(output_image.shape, (2, 3, 1))
        expected_output = np.squeeze(input_image, axis=1)
        np.testing.assert_array_equal(output_image, expected_output)

        # Squeeze all singleton dimensions
        squeeze_feature = features.Squeeze()
        output_image = squeeze_feature(input_image)
        self.assertEqual(output_image.shape, (2, 3))
        expected_output = np.squeeze(input_image)
        np.testing.assert_array_equal(output_image, expected_output)

        # Squeeze multiple axes
        squeeze_feature = features.Squeeze(axis=(1, 3))
        output_image = squeeze_feature(input_image)
        self.assertEqual(output_image.shape, (2, 3))
        expected_output = np.squeeze(np.squeeze(input_image, axis=3), axis=1)
        np.testing.assert_array_equal(output_image, expected_output)

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor([[[[3], [2], [1]]], [[[1], [2], [3]]]])
            # shape: (2, 1, 3, 1)

            squeeze_feature = features.Squeeze(axis=1)
            output_tensor = squeeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (2, 3, 1))
            expected_tensor = input_tensor.squeeze(1)
            torch.testing.assert_close(output_tensor, expected_tensor)

            squeeze_feature = features.Squeeze()
            output_tensor = squeeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (2, 3))
            expected_tensor = input_tensor.squeeze()
            torch.testing.assert_close(output_tensor, expected_tensor)

            squeeze_feature = features.Squeeze(axis=(1, 3))
            output_tensor = squeeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (2, 3))
            expected_tensor = input_tensor.squeeze(3).squeeze(1)
            torch.testing.assert_close(output_tensor, expected_tensor)


    def test_Unsqueeze(self):  # TODO
        ### Test with NumPy array
        input_image = np.array([1, 2, 3])

        unsqueeze_feature = features.Unsqueeze(axis=0)
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (1, 3))

        unsqueeze_feature = features.Unsqueeze()
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (3, 1))

        # Multiple axes
        unsqueeze_feature = features.Unsqueeze(axis=(0, 2))
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (1, 3, 1))

        # Multiple axes
        unsqueeze_feature = features.Unsqueeze(axis=(0, 2))
        output_image = unsqueeze_feature(input_image)
        self.assertEqual(output_image.shape, (1, 3, 1))

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor([1, 2, 3])

            unsqueeze_feature = features.Unsqueeze(axis=0)
            output_tensor = unsqueeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (1, 3))
            torch.testing.assert_close(output_tensor,
                                       input_tensor.unsqueeze(0))

            unsqueeze_feature = features.Unsqueeze()
            output_tensor = unsqueeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (3, 1))
            torch.testing.assert_close(output_tensor,
                                       input_tensor.unsqueeze(-1))

            # Multiple axes
            unsqueeze_feature = features.Unsqueeze(axis=(0, 2))
            output_tensor = unsqueeze_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (1, 3, 1))
            expected_tensor = input_tensor.unsqueeze(0).unsqueeze(2)
            torch.testing.assert_close(output_tensor, expected_tensor)


    def test_MoveAxis(self):  # TODO
        ### Test with NumPy array
        input_image = np.random.rand(2, 3, 4)

        move_axis_feature = features.MoveAxis(source=0, destination=2)
        output_image = move_axis_feature(input_image)
        self.assertEqual(output_image.shape, (3, 4, 2))

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_tensor = torch.rand(2, 3, 4)

            move_axis_feature = features.MoveAxis(source=0, destination=2)
            output_tensor = move_axis_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (3, 4, 2))


    def test_Transpose(self):  # TODO
        ### Test with NumPy array
        input_image = np.random.rand(2, 3, 4)

        # Explicit axes
        transpose_feature = features.Transpose(axes=(1, 2, 0))
        output_image = transpose_feature(input_image)
        self.assertEqual(output_image.shape, (3, 4, 2))
        expected_output = np.transpose(input_image, (1, 2, 0))
        self.assertTrue(np.allclose(output_image, expected_output))

        # Reversed axes
        transpose_feature = features.Transpose()
        output_image = transpose_feature(input_image)
        self.assertEqual(output_image.shape, (4, 3, 2))
        expected_output = np.transpose(input_image)
        self.assertTrue(np.allclose(output_image, expected_output))

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_tensor = torch.rand(2, 3, 4)

            # Explicit axes
            transpose_feature = features.Transpose(axes=(1, 2, 0))
            output_tensor = transpose_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (3, 4, 2))
            expected_tensor = input_tensor.permute(1, 2, 0)
            self.assertTrue(torch.allclose(output_tensor, expected_tensor))

            # Reversed axes
            transpose_feature = features.Transpose()
            output_tensor = transpose_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (4, 3, 2))
            expected_tensor = input_tensor.permute(2, 1, 0)
            self.assertTrue(torch.allclose(output_tensor, expected_tensor))


    def test_OneHot(self):  # TODO
        ### Test with NumPy array
        input_image = np.array([0, 1, 2])
        one_hot_feature = features.OneHot(num_classes=3)
        output_image = one_hot_feature(input_image)

        expected_output = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.assertEqual(output_image.shape, (3, 3))
        np.testing.assert_array_equal(output_image, expected_output)

        ### Test with singleton last dimension
        input_image = np.array([[0], [1], [2]])  # shape (3, 1)
        output_image = one_hot_feature(input_image)
        self.assertEqual(output_image.shape, (3, 3))
        np.testing.assert_array_equal(output_image, expected_output)

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            input_tensor = torch.tensor([0, 1, 2])
            output_tensor = one_hot_feature(input_tensor)

            expected_tensor = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)

            self.assertEqual(output_tensor.shape, (3, 3))
            torch.testing.assert_close(output_tensor, expected_tensor)

            # Test with singleton dimension
            input_tensor = torch.tensor([[0], [1], [2]])
            output_tensor = one_hot_feature(input_tensor)
            self.assertEqual(output_tensor.shape, (3, 3))
            torch.testing.assert_close(output_tensor, expected_tensor)


    def test_TakeProperties(self):  # TODO
        # with custom feature
        class ExampleFeature(features.Feature):
            def __init__(self, my_property, **kwargs):
                super().__init__(my_property=my_property, **kwargs)

        feature = ExampleFeature(my_property=properties.Property(42))

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

        # with `Gaussian` feature with float properties
        noise_feature = Gaussian(mu=7.123, sigma=12.123)

        take_properties = features.TakeProperties(noise_feature)
        output = take_properties.get(image=None, names=["mu", "sigma"])
        self.assertEqual(output, ([7.123], [12.123]))
        self.assertEqual(output[0][0], 7.123)
        self.assertEqual(output[1][0], 12.123)

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            class ExampleFeature(features.Feature):
                def __init__(self, my_property, **kwargs):
                    super().__init__(my_property=my_property, **kwargs)

            feature = ExampleFeature(my_property=
                properties.Property(torch.tensor(42.123)))

            take_properties = features.TakeProperties(feature)
            output = take_properties.get(image=None, names=["my_property"])
            torch.testing.assert_close(output[0], torch.tensor(42.123))

            # with `Gaussian` feature
            noise_feature = Gaussian(
                mu=torch.tensor(7), sigma=torch.tensor(12)
            )

            take_properties = features.TakeProperties(noise_feature)
            output = take_properties.get(image=None, names=["mu"])
            torch.testing.assert_close(output[0], torch.tensor(7))
            output = take_properties.get(image=None, names=["sigma"])
            torch.testing.assert_close(output[0], torch.tensor(12))

            # with `Gaussian` feature with float properties
            random_mu = torch.rand(1)
            random_sigma = torch.rand(1)
            noise_feature = Gaussian(mu=random_mu, sigma=random_sigma)

            take_properties = features.TakeProperties(noise_feature)
            output = take_properties.get(image=None, names=["mu", "sigma"])
            torch.testing.assert_close(output, ([random_mu], [random_sigma]))
            torch.testing.assert_close(output[0][0], random_mu)
            torch.testing.assert_close(output[1][0], random_sigma)


if __name__ == "__main__":
    unittest.main()
