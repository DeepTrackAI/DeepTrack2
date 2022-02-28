import sys
import numpy as np
import itertools
import deeptrack as dt
import pytest


u = dt.units


def create_pipeline(output_region=(0, 0, 128, 128), num_particles=1):

    optics = dt.Fluorescence(output_region=output_region)

    mie = dt.Sphere(
        radius=2e-6,
        refractive_index=1.45,
        z=10,
        position=lambda: output_region[2:] * np.random.randn(2),
    )

    field = optics(mie ^ num_particles)
    return field


@pytest.mark.parametrize(
    "size,gpu",
    [
        *itertools.product(
            (64, 256, 512),
            [True, False],
        )
    ],
)
def test_simulate_mie(size, gpu, benchmark):
    benchmark.group = f"fluorescence_{size}_px_image"
    benchmark.name = f"test_fluorescence_{'gpu' if gpu else 'cpu'}"
    if gpu:
        dt.config.enable_gpu()
    else:
        dt.config.disable_gpu()
    pipeline = create_pipeline(output_region=(0, 0, size, size), num_particles=1)
    # One cold run for performance
    pipeline.update()()
    benchmark(
        lambda: pipeline.update()(),
    )
