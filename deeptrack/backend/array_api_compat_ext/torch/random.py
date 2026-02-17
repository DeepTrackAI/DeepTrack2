from __future__ import annotations

import torch

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


def rand(*size: int) -> torch.Tensor:
    """Sample uniform random numbers in [0, 1) with a given shape.

    This function mirrors `numpy.random.rand`, i.e., it takes the output shape
    as positional integer arguments.

    Parameters
    ----------
    *size: int
        Output shape given as positional integers. If empty, returns a scalar
        0D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape `size` (or scalar if `size` is empty) with values
        sampled uniformly from [0, 1).

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.rand(2, 3).shape
    torch.Size([2, 3])

    Scalar sample:

    >>> rnd.rand()
    tensor(0.1735)

    """

    if not size:
        return torch.rand(())

    return torch.rand(*size)


def random(size: tuple[int, ...] | None = None) -> torch.Tensor:
    """Sample uniform random numbers in [0, 1).

    This function mirrors `numpy.random.random`, which takes the output
    shape as a tuple. If `size` is `None`, a scalar 0D tensor is returned.

    Parameters
    ----------
    size: tuple[int, ...] or None, optional
        Output shape. If `None`, returns a scalar tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape `size` (or scalar if `size` is `None`) with values
        sampled uniformly from [0, 1).

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.random((2, 3)).shape
    torch.Size([2, 3])

    Scalar sample:

    >>> rnd.random()
    tensor(0.1124)

    """

    if size is None:
        return torch.rand(())

    return torch.rand(*size)


random_sample = random


def randn(*size: int) -> torch.Tensor:
    """Sample from the standard normal distribution.

    This function mirrors `numpy.random.randn`, i.e. it takes the output
    shape as positional integer arguments.

    Parameters
    ----------
    *size: int
        Output shape given as positional integers. If empty, returns a scalar
        0D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape `size` (or scalar if `size` is empty) with values
        sampled from a standard normal distribution.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.randn(2, 3).shape
    torch.Size([2, 3])

    Scalar sample:
    
    >>> rnd.randn()
    tensor(-2.2435)

    """

    if not size:
        return torch.randn(())

    return torch.randn(*size)


def beta(
    a: float,
    b: float,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    raise NotImplementedError("the beta distribution is not implemented in torch")


def binomial(
    n: int,
    p: float,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    return torch.bernoulli(torch.full(size, p))


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
