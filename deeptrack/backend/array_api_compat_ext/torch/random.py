from __future__ import annotations

import torch
import numpy as np

__all__ = [
    "rand",
    "random",
    "random_sample",
    "randn",
    "beta",
    "binomial",
    "choice",
    "multinomial",
    "randint",
    "shuffle",
    "uniform",
    "normal",
    "poisson",
]


def rand(
    *args: int,
    dtype: torch.dtype=torch.float32,
    device: torch.device | str = torch.device("cpu"),    
) -> torch.Tensor:
    return torch.rand(*args, dtype=dtype, device=device)


def random(size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.rand(*size) if size else torch.rand()


def random_sample(size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.rand(*size) if size else torch.rand()


def randn(
    *args: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.randn(*args, dtype=dtype, device=device)


def beta(
    a: float,
    b: float,
    size: int | tuple[int, ...] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(np.random.beta(a, b, size), dtype=dtype, device=device)


def binomial(
    n: int,
    p: float,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    #return torch.bernoulli(torch.full(size, p))
    return torch.tensor(np.random.binomial(n, p, size), dtype=dtype, device=device)


def choice(
    a: torch.Tensor,
    size: tuple[int, ...] | None = None,
    replace: bool = True,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError(
        "the choice function is not implemented in torch"
    )


def multinomial(
    n: int,
    pvals: torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.multinomial(pvals, n, size)


def randint(
    low: int,
    high: int,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.randint(low, high, size)


def shuffle(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.shape[0])]


def uniform(
    low: float,
    high: float,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.rand(*size) * (high - low) + low


def normal(
    loc: float,
    scale: float,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.randn(*size) * scale + loc


def poisson(
    lam: float,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.poisson(torch.full(size, lam))


# TODO: implement the rest of the functions as they are needed
