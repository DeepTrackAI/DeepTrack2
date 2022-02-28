import sys
import numpy as np
import itertools
import deeptrack as dt
import pytest
import itertools


u = dt.units


def create_pipeline(elements=1024):

    value = dt.Value(np.zeros((elements,)))

    value = value + 14

    value = value * (np.ones((elements,)) * 2)

    value = value / 1.5

    value = value ** 2

    return value


@pytest.mark.parametrize(
    "elements,gpu",
    [*itertools.product((1000, 5000, 10000, 50000, 100000, 500000), [True, False])],
)
def test_arithmetic(elements, gpu, benchmark):
    benchmark.group = "arithm_{}_elements".format(elements)
    benchmark.name = "test_arithmetic_{}".format("gpu" if gpu else "cpu")
    if gpu:
        dt.config.enable_gpu()
    else:
        dt.config.disable_gpu()
    pipeline = create_pipeline(elements=elements)
    benchmark(
        lambda: pipeline.update()(),
    )
