import sys

# sys.path.append(".")  # Adds the module to path

import unittest
import operator
import itertools
from numpy.core.numeric import array_equal

from numpy.testing._private.utils import assert_almost_equal

from .. import elementwise, features, Image

import numpy as np
from deeptrack.backend._config import cupy as cp

import numpy.testing
import inspect


def grid_test_features(
    tester,
    feature,
    feature_inputs,
    expected_result_function,
):

    for f_a_input in feature_inputs:

        inp = features.Value(f_a_input)

        f_a = feature(inp)
        f_b = inp >> feature()

        for f in [f_a, f_b]:
            try:
                output = f()
            except Exception as e:
                tester.assertRaises(
                    type(e),
                    lambda: expected_result_function(f_a_input),
                )
                continue

            expected_result = expected_result_function(f_a_input)
            output = np.array(output)
            try:
                expected_result = np.array(expected_result)
            except TypeError:
                expected_result = expected_result.get()

            if isinstance(output, list) and isinstance(expected_result, list):
                [
                    np.testing.assert_almost_equal(np.array(a), np.array(b))
                    for a, b in zip(output, expected_result)
                ]

            else:
                is_equal = np.allclose(
                    np.array(output), np.array(expected_result), equal_nan=True
                )

                tester.failIf(
                    not is_equal,
                    "Feature output {} is not equal to expect result {}.\n Using arguments {}".format(
                        output, expected_result, f_a_input
                    ),
                )


def create_test(cl):
    testname = "test_{}".format(cl.__name__)

    def test(self):
        grid_test_features(
            self,
            cl,
            [
                -1,
                0,
                1,
                (np.random.rand(50, 500) - 0.5) * 100,
                (cp.random.rand(50, 500) - 0.5) * 100,
            ],
            np.__dict__[cl.__name__.lower()],
        )

    test.__name__ = testname

    return testname, test


class TestFeatures(unittest.TestCase):
    pass


classes = inspect.getmembers(elementwise, inspect.isclass)

for clname, cl in classes:

    if not issubclass(cl, elementwise.ElementwiseFeature) or (
        cl is elementwise.ElementwiseFeature
    ):
        continue

    testname, test_method = create_test(cl)
    setattr(TestFeatures, testname, test_method)


if __name__ == "__main__":
    unittest.main()
