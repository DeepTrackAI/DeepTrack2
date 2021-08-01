import sys
import numpy as np
import itertools
import deeptrack as dt
import pytest
import itertools
from deeptrack.backend._config import cupy as cp

u = dt.units


def create_pipeline(elements=1024):
    value = dt.Value(np.zeros((elements,)))
    value = value + 14
    value = value * (np.ones((elements,)) * 2)
    value = value / 1.5
    value = value ** 2
    return value


@pytest.mark.parametrize(
    "elements,gpu,image",
    [*itertools.product((1000, 10000, 100000, 1000000), [True, False], [True, False])],
)
def test_arithmetic(elements, gpu, image, benchmark):
    benchmark.group = "add_{}_elements".format(elements)
    benchmark.name = "test_{}_{}".format(
        "Image" if image else "array", "gpu" if gpu else "cpu"
    )

    a = np.random.randn(elements)
    b = np.random.randn(elements)

    if gpu:
        a = cp.array(a)
        b = cp.array(b)
    if image:
        a = dt.image.Image(a)
        b = dt.image.Image(b)

    benchmark(
        lambda: a + b,
    )
