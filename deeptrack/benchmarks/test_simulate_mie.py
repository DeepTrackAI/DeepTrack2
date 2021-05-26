import sys
import numpy as np
import itertools
import deeptrack as dt
import pytest


u = dt.units


def create_pipeline(output_region=(0, 0, 128, 128), num_particles=1):

    optics = dt.Brightfield(output_region=output_region)

    mie = dt.MieSphere(
        radius=0.5e-6,
        refractive_index=1.45,
        z=lambda: np.random.randn() * 10,
        position=lambda: output_region[2:] * np.random.randn(2),
        L=20,
    )

    field = optics(mie ^ num_particles)
    return field


@pytest.mark.parametrize(
    "size,gpu",
    [*itertools.product((64, 128, 256, 512, 728), [True, False])],
)
def test_simulate_mie(size, gpu, benchmark):
    benchmark.group = "mie_{}_px_image".format(size)
    benchmark.name = "test_simulate_mie_{}".format("gpu" if gpu else "cpu")
    if gpu:
        dt.config.enable_gpu()
    else:
        dt.config.disable_gpu()
    pipeline = create_pipeline(output_region=(0, 0, size, size), num_particles=2)
    benchmark(
        lambda: pipeline.update()(),
    )
