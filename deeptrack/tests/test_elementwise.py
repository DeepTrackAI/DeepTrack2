# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import inspect
import unittest

import numpy as np

from deeptrack import elementwise, features, TORCH_AVAILABLE, xp

if TORCH_AVAILABLE:
    import torch

def grid_test_features(
    tester,
    elementwise_class,
    feature_inputs,
    function_name,
):
    for feature_input in feature_inputs:
        pip_a = elementwise_class(features.Value(feature_input))
        pip_b = features.Value(feature_input) >> elementwise_class()

        for pip in [pip_a, pip_b]:
            result = pip()

            if TORCH_AVAILABLE and isinstance(result, torch.Tensor):
                function = torch.__dict__[function_name]
                expected_result = function(feature_input)

                # In PyTorch, NaNs are unequal by default
                valid_mask = ~(torch.isnan(result) 
                               | torch.isnan(expected_result))

                torch.testing.assert_close(
                    result[valid_mask],
                    expected_result[valid_mask],
                    rtol=1e-5,
                    atol=1e-8,
                    msg=f"{elementwise_class.__name__} failed with PyTorch.",
                )
            else:
                function = np.__dict__[function_name]
                expected_result = function(feature_input)

                # In NumPy, NaNs are ignored

                np.testing.assert_allclose(
                    result,
                    expected_result,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"{elementwise_class.__name__} failed with NumPy.",
                )


def create_test(elementwise_class):
    testname = f"test_{elementwise_class.__name__}"

    def test(self):
        inputs = [
            np.array(-1.0),
            np.array(0.0),
            np.array(1.0),
            (np.random.rand(8, 15) - 0.5) * 100,
        ]

        if TORCH_AVAILABLE:
            inputs.extend([
                torch.tensor([-1.0, 0.0, 1.0]),
                (torch.rand(8, 15) - 0.5) * 100,
            ])

        grid_test_features(
            self,
            elementwise_class,
            inputs,
            elementwise_class.__name__.lower(),
        )

    test.__name__ = testname

    return testname, test


class TestElementwiseFeatures(unittest.TestCase):
    pass


elementwise_classes = inspect.getmembers(elementwise, inspect.isclass)

for class_name, elementwise_class in elementwise_classes:

    if (
        elementwise_class is elementwise.ElementwiseFeature
        or
        not issubclass(elementwise_class, elementwise.ElementwiseFeature)
    ):
        continue

    test_name, test_method = create_test(elementwise_class)
    setattr(TestElementwiseFeatures, test_name, test_method)


if __name__ == "__main__":
    unittest.main()
