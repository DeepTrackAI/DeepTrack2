from __future__ import annotations
import numpy as np
import torch
from typing import Optional

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


def rand(*args: int) -> torch.Tensor:
    return torch.rand(*args)


def random(size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.rand(*size)


def random_sample(size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.rand(*size)


def randn(*args: int) -> torch.Tensor:
    return torch.randn(*args)


def beta(a: float, b: float, size: tuple[int, ...] | None = None) -> torch.Tensor:
    raise NotImplementedError("the beta distribution is not implemented in torch")


# np.random.


def binomial(n: int, p: float, size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.bernoulli(torch.full(size, p))


def choice(
    a: torch.Tensor,
    size: tuple[int, ...] | None = None,
    replace: bool = True,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    raise NotImplementedError("the choice function is not implemented in torch")


def multinomial(
    n: int, pvals: torch.Tensor, size: tuple[int, ...] | None = None
) -> torch.Tensor:
    return torch.multinomial(pvals, n, size)


def randint(low: int, high: int, size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.randint(low, high, size)


def shuffle(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.shape[0])]


def uniform(
    low: float, high: float, size: tuple[int, ...] | None = None
) -> torch.Tensor:
    return torch.rand(*size) * (high - low) + low


def normal(
    loc: float, scale: float, size: tuple[int, ...] | None = None
) -> torch.Tensor:
    return torch.randn(*size) * scale + loc


def poisson(lam: float, size: tuple[int, ...] | None = None) -> torch.Tensor:
    return torch.poisson(torch.full(size, lam))


# TODO: implement the rest of the functions
