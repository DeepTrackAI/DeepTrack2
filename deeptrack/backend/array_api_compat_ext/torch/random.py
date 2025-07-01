"""xp compatibility module for Numpy functions

This module contains wrapper functions for various numpy.random functions
that return torch tensors when and can accept optional `dtype` and`device` arguments. 


Examples
--------
Sample the `beta` distribution:

>>> from torch import cuda, float16

>>> if cuda.is_available():
...     print(beta(1, 2, dtype=torch.float16, device="cuda"))

tensor(0.3315, device='cuda:0', dtype=torch.float16)


"""

from __future__ import annotations
from deeptrack.types import ArrayLike
import torch
import numpy as np

__all__ = [
    "rand",
    "random",
    "random_sample",
    "randn",
    "standard_normal",
    "beta",
    "binomial",
    "choice",
    "multinomial",
    "randint",
    "shuffle",
    "uniform",
    "normal",
    "poisson",
    "gamma",
]


def rand(
    *args: int,
    dtype: torch.dtype=torch.float32,
    device: torch.device | str = torch.device("cpu"),    
) -> torch.Tensor:
    return torch.rand(*args, dtype=dtype, device=device)


def random(
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype=torch.float32,
    device: torch.device | str = torch.device("cpu"),  
) -> torch.Tensor:
    return (
        torch.rand(*size, dtype=dtype, device=device)
        if size else torch.rand(dtype=dtype, device=device)
    )


def random_sample(
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype=torch.float32,
    device: torch.device | str = torch.device("cpu"),  
) -> torch.Tensor:
    return (
        torch.rand(*size, dtype=dtype, device=device)
        if size else torch.rand(dtype=dtype, device=device)
    )


def randn(
    *args: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.randn(*args, dtype=dtype, device=device)


def standard_normal(
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
    return (
        torch.tensor(np.random.beta(a, b, size), dtype=dtype, device=device)
    )


def binomial(
    n: int,
    p: float,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return (
        torch.tensor(np.random.binomial(n, p, size), dtype=dtype, device=device)
    )


def choice(
    a: torch.Tensor | np.ndarray,
    size: tuple[int, ...] | None = None,
    replace: bool = True,
    p: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    
    a_numpy = a.cpu().numpy()
    p_numpy = p.cpu().numpy() if p is not None else None
    return (
        torch.tensor(
            np.random_choice(
                a_numpy, size=size, replace=replace, p=p_numpy
            ), dtype=dtype, device=device
        )
    )
    

def multinomial(
    n: int,
    pvals: torch.Tensor,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.multinomial(pvals, n, size, dtype=dtype, device=device)


def randint(
    low: int,
    high: int,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.randint(low, high, size, dtype=dtype, device=device)


def shuffle(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.shape[0], device=x.device)]


def uniform(
    low: float,
    high: float,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.rand(*size, dtype=dtype, device=device) * (high - low) + low


def normal(
    loc: float,
    scale: float,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.randn(*size, dtype=dtype, device=device) * scale + loc


def poisson(
    lam: float,
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.poisson(torch.full(size, lam, dtype=dtype, device=device))


def gamma(
    shape: float | torch.Tensor,
    scale: float | torch.Tensor = 1.0, 
    size: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    
    shape = torch.as_tensor(shape, dtype=dtype, device=device)
    scale = torch.as_tensor(scale, dtype=dtype, device=device)
    if size is not None:
        shape = shape.expand(size)
        scale = scale.expand(size)
    return torch.distributions.Gamma(shape, scale).sample()


def exponential(
    scale: float | torch.Tensor = 1.0,
    size: tuple[int, ...] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    
    rate = torch.as_tensor(1.0/scale, dtype=dtype, device=device)
    if size is None:
        return torch.distributions.Exponential(rate).sample()
    return torch.distributions.Exponential(rate).sample(size)


def multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: tuple[int, ...]] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    
    mean = mean.to(dtype=dtype, device=device)
    cov = cov.to(dtype=dtype, device=device)
    if size is None:
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample()
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample(size)


def geometric(
    p: float | torch.Tensor,
    size: tuple[int, ...] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    
    p = torch.as_tensor(p, dtype=torch.float32, device=device)
    if size is None:
        return torch.distributions.Geometric(probs=p).sample().to(dtype)
    return torch.distributions.Geometric(probs=p).sample(size).to(dtype)


def dirichlet(
    alpha: torch.Tensor,
    size: tuple[int, ...] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    
    alpha = alpha.to(dtype=dtype, device=device)
    if size is None:
        return torch.distributions.Dirichlet(alpha).sample()
    return torch.distributions.Dirichlet(alpha).sample(size)

# TODO: implement the rest of the functions as they are needed
