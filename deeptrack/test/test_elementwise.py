import sys

# sys.path.append(".")  # Adds the module to path

import unittest
import operator
import itertools
from numpy.core.numeric import array_equal

from numpy.testing._private.utils import assert_almost_equal

from .. import elementwise, features, Image

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

            if isinstance(output, list) and isinstance(expected_result, list):
                [
                    np.testing.assert_almost_equal(Image(a), Image(b))
                    for a, b in zip(output, expected_result)
                ]

            else:
                is_equal = np.array_equal(output, expected_result, equal_nan=True)

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
            [-1, 0, 1, (np.random.rand(50, 500) - 0.5) * 100, np.inf, np.nan],
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
