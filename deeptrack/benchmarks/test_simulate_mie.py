import sys
import numpy as np
import itertools
import deeptrack as dt

u = dt.units


def create_pipeline(output_region=(0, 0, 128, 128), num_particles=1):

    optics = dt.Brightfield(output_region=output_region)

    mie = dt.MieSphere(
        radius=0.5e-6,
        refractive_index=1.45,
        z=lambda: np.random.randn() * 10,
        position=lambda: output_region[2:] * np.random.randn(2),
        L=10,
    )

    field = optics(mie ^ num_particles)
    return field


def test_simulate_mie_128_1(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 128, 128), num_particles=1)
    benchmark(
        lambda: pipeline.update()(),
    )


def test_simulate_mie_128_5(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 128, 128), num_particles=5)
    benchmark(
        lambda: pipeline.update()(),
    )


def test_simulate_mie_512_1(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 512, 512), num_particles=1)
    benchmark(
        lambda: pipeline.update()(),
    )


def test_simulate_mie_512_5(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 512, 512), num_particles=5)
    benchmark(
        lambda: pipeline.update()(),
    )


def test_simulate_mie_1024_1(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 1024, 1024), num_particles=1)
    benchmark(
        lambda: pipeline.update()(),
    )


def test_simulate_mie_1024_5(benchmark):
    pipeline = create_pipeline(output_region=(0, 0, 1024, 1024), num_particles=5)
    benchmark(
        lambda: pipeline.update()(),
    )