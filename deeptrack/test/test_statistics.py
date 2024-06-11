import sys

# sys.path.append(".")  # Adds the module to path

import unittest
import operator
import itertools
from numpy.core.numeric import array_equal

from numpy.testing._private.utils import assert_almost_equal

from .. import statistics, features, Image

import numpy as np
import numpy.testing
import inspect


def grid_test_features(
    tester,
    feature,
    feature_inputs,
    expected_result_function,
):
    

    for f_a_input in feature_inputs:

        for axis in [None]:
            for distributed in [True, False]:
                inp = features.Value(f_a_input)

                f_a = feature(
                    inp,
                    axis=axis,
                    distributed=distributed,
                    q=lambda: np.random.rand(),
                )
                f_b = inp >> feature(
                    axis=axis,
                    distributed=distributed,
                    q=lambda: np.random.rand(),
                )
                f_a.store_properties()
                f_b.store_properties()

                for f in [f_a, f_b]:
                    try:
                        output = f()
                    except Exception as e:
                        tester.assertRaises(
                            type(e),
                            lambda: expected_result_function(
                                f_a_input,
                                axis=axis,
                                q=0.95,
                            ),
                        )
                        continue

                    if distributed and isinstance(f_a_input, list):
                        expected_result = [
                            expected_result_function(
                                i,
                                axis=axis,
                                q=0.95,
                            )
                            for i in f_a_input
                        ]
                    elif not distributed and not isinstance(f_a_input, list):
                        expected_result = expected_result_function(
                            [f_a_input],
                            axis=axis,
                            q=output.get_property("q"),
                        )
                    else:
                        expected_result = expected_result_function(
                            f_a_input,
                            axis=axis,
                            q=output.get_property("q"),
                        )

                    if isinstance(output, list) and isinstance(expected_result, list):
                        [
                            np.testing.assert_almost_equal(np.array(a), np.array(b))
                            for a, b in zip(output, expected_result)
                        ]

                    else:
                        np.testing.assert_almost_equal(
                            np.array(output), np.array(expected_result)
                        )


# def array_equal(a, b):

#     assert a.shape == b.shape, "Shape mismatch {} vs {}".


def create_test(cl):
    testname = "test_{}".format(cl.__name__)

    cl_name = cl.__name__.lower()

    if cl_name == "peaktopeak":
        cl_name = "ptp"
    if cl_name == "variance":
        cl_name = "var"

    def expected(val, **kwargs):
        try:
            return np.__dict__[cl_name](val, **kwargs)
        except TypeError as e:
            kwargs.pop("q")
            return np.__dict__[cl_name](val, **kwargs)

    def test(self):
        grid_test_features(
            self,
            cl,
            [
                -1,
                0,
                1,
                (np.random.rand(3, 5) - 0.5) * 100,
                np.inf,
                np.nan,
                [np.zeros((3, 4)), np.ones((3, 4))],
                np.random.rand(2, 3, 2, 3),
            ],
            expected,
        )

    test.__name__ = testname

    return testname, test


class TestFeatures(unittest.TestCase):
    def test_broadcast_list(self):

        inp = features.Value([1, 0])

        pipeline = inp - statistics.Mean(inp)
        self.assertListEqual(pipeline(), [0, 0])

        pipeline = inp - (inp >> statistics.Mean())
        self.assertListEqual(pipeline(), [0, 0])


classes = inspect.getmembers(statistics, inspect.isclass)

for clname, cl in classes:

    if not issubclass(cl, statistics.Reducer) or (cl is statistics.Reducer):
        continue

    testname, test_method = create_test(cl)
    setattr(TestFeatures, testname, test_method)


if __name__ == "__main__":
    unittest.main()
