# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import inspect
import unittest
import warnings

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from deeptrack import elementwise, features, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


# NumPy uses arc* names (np.arcsin, np.arccosh, ...)
NUMPY_NAME_MAP = {
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arcsinh": "arcsinh",
    "arccosh": "arccosh",
    "arctanh": "arctanh",
    "conjugate": "conj",  # DeepTrack uses Conjugate alias
}

# Torch uses short names (torch.asin, torch.acosh, ...)
TORCH_NAME_MAP = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "conjugate": "conj",
}

# Functions that should not be tested on complex inputs
# (backend-specific reality)
DISALLOW_COMPLEX_NUMPY = {"Floor", "Ceil", "Round"}
DISALLOW_COMPLEX_TORCH = {"Floor", "Ceil", "Round", "Sign"}


def _is_complex_input(x: np.ndarray | torch.Tensor) -> bool:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.is_complex(x)
    return np.iscomplexobj(x)


def _torch_expected(function_name: str, x: torch.Tensor) -> torch.Tensor:
    torch_name = TORCH_NAME_MAP.get(function_name, function_name)
    function = getattr(torch, torch_name)

    if function is torch.imag:
        return torch.imag(x) if torch.is_complex(x) else torch.zeros_like(x)

    return function(x)


def _numpy_expected(function_name: str, x: NDArray) -> NDArray:
    numpy_name = NUMPY_NAME_MAP.get(function_name, function_name)
    function = getattr(np, numpy_name)
    return function(x)


def grid_test_features(
    elementwise_class: type[elementwise.ElementwiseFeature],
    feature_inputs: Iterable[
        NDArray[np.floating] | NDArray[np.complexfloating] | torch.Tensor
    ],
    function_name: str,
):
    for feature_input in feature_inputs:
        # Skip before evaluating the pipeline
        # (otherwise crash inside pip()).
        if _is_complex_input(feature_input):
            if TORCH_AVAILABLE and isinstance(feature_input, torch.Tensor):
                if elementwise_class.__name__ in DISALLOW_COMPLEX_TORCH:
                    continue
            else:
                if elementwise_class.__name__ in DISALLOW_COMPLEX_NUMPY:
                    continue

        pip_a = elementwise_class(features.Value(feature_input))
        pip_b = features.Value(feature_input) >> elementwise_class()

        for pip in [pip_a, pip_b]:
            # Silence expected domain warnings
            # (log, sqrt, arctanh, arccosh, ...)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = pip()

            # Torch branch (decide from input type, not result type)
            if TORCH_AVAILABLE and isinstance(feature_input, torch.Tensor):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected_result = _torch_expected(
                        function_name, feature_input
                    )

                try:
                    torch.testing.assert_close(
                        result,
                        expected_result,
                        rtol=1e-5,
                        atol=1e-8,
                        equal_nan=True,
                        msg=(
                            f"{elementwise_class.__name__} failed with PyTorch"
                            f" (dtype={feature_input.dtype}, "
                            f"shape={tuple(feature_input.shape)})."
                        ),
                    )
                except:
                    print(
                        f"Result: {result} \n" f"Expect: {expected_result}\n\n"
                    )

            # NumPy branch
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected_result = _numpy_expected(
                        function_name, feature_input
                    )

                np.testing.assert_allclose(
                    result,
                    expected_result,
                    rtol=1e-5,
                    atol=1e-8,
                    equal_nan=True,
                    err_msg=(
                        f"{elementwise_class.__name__} failed with NumPy "
                        f"(dtype={getattr(feature_input, 'dtype', None)}, "
                        f"shape={getattr(feature_input, 'shape', None)})."
                    ),
                )


def create_test(elementwise_class: type[elementwise.ElementwiseFeature]):
    testname = f"test_{elementwise_class.__name__}"

    def test(self):
        # Keep inputs broad but lightweight.
        inputs = [
            np.array(-1.0),
            np.array(0.0),
            np.array(1.0),
            (np.random.rand(4, 5) - 0.5) * 100,
            np.array([1 + 2j, -3 + 0j], dtype=np.complex64),
        ]

        if TORCH_AVAILABLE:
            inputs.extend(
                [
                    torch.tensor([-1.0, 0.0, 1.0]),
                    (torch.rand(4, 5) - 0.5) * 100,
                    torch.tensor([1 + 2j, -3 + 0j], dtype=torch.complex64),
                ]
            )

        grid_test_features(
            elementwise_class=elementwise_class,
            feature_inputs=inputs,
            function_name=elementwise_class.__name__.lower(),
        )

    test.__name__ = testname
    return testname, test


class TestElementwiseFeatures(unittest.TestCase):
    pass


elementwise_classes = sorted(
    inspect.getmembers(elementwise, inspect.isclass),
    key=lambda kv: kv[0],
)

for _, elementwise_class in elementwise_classes:
    if elementwise_class is elementwise.ElementwiseFeature or not issubclass(
        elementwise_class, elementwise.ElementwiseFeature
    ):
        continue

    test_name, test_method = create_test(elementwise_class)
    setattr(TestElementwiseFeatures, test_name, test_method)


if __name__ == "__main__":
    unittest.main()
