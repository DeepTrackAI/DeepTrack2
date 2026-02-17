from __future__ import annotations

import torch
from torch.distributions import Beta, Binomial, Multinomial


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
    "permutation",
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
    size: tuple[int, ...] | None, optional
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
    a: float | torch.Tensor,
    b: float | torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a Beta distribution.

    Mirrors `numpy.random.beta`, including support for tensor parameters and
    broadcasting. If `a` and/or `b` are tensors, the output batch shape
    follows their broadcasted shape.

    Parameters
    ----------
    a: float | torch.Tensor
        First shape parameter (alpha). Can be a scalar or a tensor.
    b: float | torch.Tensor
        Second shape parameter (beta). Can be a scalar or a tensor.
    size: tuple[int, ...] | None, optional
        Sample shape prepended to the broadcasted parameter shape. If `None`,
        returns samples with the broadcasted parameter shape (scalar if both
        parameters are scalars).

    Returns
    -------
    torch.Tensor
        Samples drawn from Beta(a, b). Output shape is `size + batch_shape`,
        where `batch_shape` is the broadcasted shape of `a` and `b`.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    Scalar parameters:

    >>> rnd.beta(2.0, 5.0).ndim
    tensor(0.0784)

    Tensor parameters (broadcasted):

    >>> import torch
    >>>
    >>> a = torch.tensor([2.0, 3.0])
    >>> b = torch.tensor([5.0, 7.0])
    >>> rnd.beta(a, b)
    tensor([0.2679, 0.2765])

    With explicit sample shape:

    >>> rnd.beta(a, b, (4,)).shape
    torch.Size([4, 2])

    """

    dist = Beta(a, b)

    if size is None:
        return dist.sample()

    return dist.sample(size)


def binomial(
    n: int | torch.Tensor,
    p: float | torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a Binomial distribution.

    Mirrors `numpy.random.binomial`, including support for tensor
    parameters and broadcasting.

    Parameters
    ----------
    n: int | torch.Tensor
        Number of trials.
    p: float | torch.Tensor
        Probability of success.
    size: tuple[int, ...] | None, optional
        Sample shape. If `None`, returns samples with the broadcasted
        parameter shape.

    Returns
    -------
    torch.Tensor
        Samples drawn from Binomial(n, p). Output shape is
        `size + batch_shape`.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.binomial(10, 0.5)
    tensor(6.)

    >>> rnd.binomial(10, 0.5, (2, 3)).shape
    torch.Size([2, 3])

    """

    dist = Binomial(total_count=n, probs=p)

    if size is None:
        return dist.sample()

    return dist.sample(size)


def choice(
    a: int | torch.Tensor,
    size: tuple[int, ...] | None = None,
    replace: bool = True,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from a 1D tensor or from `range(a)`.

    This function mirrors `numpy.random.choice`.

    Parameters
    ----------
    a: int | torch.Tensor
        If an integer, samples are drawn from `torch.arange(a)`. If a tensor,
        it must be 1D and samples are drawn from its elements.
    size: tuple[int, ...] | None, optional
        Output shape. If `None`, returns a scalar 0D tensor.
    replace: bool, optional
        Whether sampling is with replacement. Defaults to `True`.
    p: torch.Tensor | None, optional
        Optional probability weights. Must have the same length as the
        population and sum to 1 (normalization is applied internally).

    Returns
    -------
    torch.Tensor
        Samples drawn from `a` (or from `range(a)` if `a` is an integer).

    Raises
    ------
    ValueError
        If `a` is a tensor and is not 1D, if `a` is an integer < 1, or if `p`
        has an incompatible shape.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    Sample a scalar from a tensor:

    >>> import torch
    >>>
    >>> a = torch.tensor([10, 20, 30, 40])
    >>> rnd.choice(a)
    tensor(40)

    Sample an array of shape (2, 3):

    >>> rnd.choice(a, (2, 3)).shape
    torch.Size([2, 3])

    Sample from `range(5)` (NumPy parity with `np.random.choice(5)`):

    >>> rnd.choice(5, (4,)).shape
    torch.Size([4])

    Use probabilities (always pick index 2 from `range(4)`):

    >>> p = torch.tensor([0.0, 0.0, 1.0, 0.0])
    >>> rnd.choice(4, (3,), p=p)
    tensor([2, 2, 2])

    """

    if isinstance(a, int):
        if a < 1:
            raise ValueError("`a` must be >= 1 when provided as an integer")
        population = torch.arange(a)
    else:
        if a.ndim != 1:
            raise ValueError("`a` must be 1D")
        population = a

    n = population.shape[0]

    if p is None:
        probs = torch.ones(n, dtype=torch.float, device=population.device)
    else:
        if p.shape != (n,):
            raise ValueError("`p` must have shape (len(a),)")
        probs = p.to(dtype=torch.float, device=population.device)

    probs = probs / probs.sum()

    if size is None:
        indices = torch.multinomial(probs, 1, replacement=replace)
        return population[indices].squeeze()

    num_samples = int(torch.tensor(size).prod().item())

    indices = torch.multinomial(
        probs,
        num_samples,
        replacement=replace,
    )

    return population[indices].reshape(size)


def multinomial(
    n: int | torch.Tensor,
    pvals: torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a multinomial distribution.

    Mirrors `numpy.random.multinomial`.

    Parameters
    ----------
    n: int | torch.Tensor
        Number of trials.
    pvals: torch.Tensor
        1D tensor of category probabilities.
    size: tuple[int, ...] | None, optional
        Sample shape. If `None`, returns a single draw.

    Returns
    -------
    torch.Tensor
        Counts per category. Output shape is
        `(len(pvals),)` if `size=None`,
        otherwise `size + (len(pvals),)`.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    Single draw:

    >>> import torch
    >>>
    >>> p = torch.tensor([0.2, 0.8])
    >>> rnd.multinomial(5, p)
    tensor([1., 4.])

    Multiple draws:

    >>> rnd.multinomial(5, p, (3,)).shape
    torch.Size([3, 2])

    """

    if pvals.ndim != 1:
        raise ValueError("`pvals` must be 1D")

    probs = pvals.to(dtype=torch.float)
    probs = probs / probs.sum()

    dist = Multinomial(total_count=n, probs=probs)

    if size is None:
        return dist.sample()

    return dist.sample(size)


def randint(
    low: int,
    high: int | None = None,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample integers from a uniform discrete distribution.

    Mirrors `numpy.random.randint`.

    Parameters
    ----------
    low: int
        Lowest integer (inclusive) if `high` is provided.
        If `high` is None, this is treated as the exclusive upper bound,
        and `low` is set to 0.
    high: int | None, optional
        Upper bound (exclusive).
    size: tuple[int, ...] | None, optional
        Output shape. If `None`, returns a scalar tensor.

    Returns
    -------
    torch.Tensor
        Random integers in `[low, high)`.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.randint(5)
    tensor(3)

    >>> rnd.randint(2, 10, (2, 3)).shape
    torch.Size([2, 3])

    """

    if high is None:
        high = low
        low = 0

    if size is None:
        return torch.randint(low, high, ()).to(torch.int64)

    return torch.randint(low, high, size).to(torch.int64)


def shuffle(x: torch.Tensor) -> None:
    """Shuffle a tensor in-place along the first axis.

    Mirrors `numpy.random.shuffle`.

    Parameters
    ----------
    x: torch.Tensor
        Tensor to shuffle along the first axis.

    Returns
    -------
    None
        The tensor is shuffled in-place.

    Examples
    --------
    >>> from deeptrack.backend.array_api_compat_ext.torch import random as rnd

    >>> import torch
    >>>
    >>> x = torch.tensor([1, 2, 3, 4])
    >>> rnd.shuffle(x)
    >>> x
    tensor([2, 1, 3, 4])

    """

    if x.ndim == 0:
        return

    perm = torch.randperm(x.shape[0], device=x.device)
    x[:] = x[perm]


def permutation(x: int | torch.Tensor) -> torch.Tensor:
    """Return a permuted sequence or tensor.

    Mirrors `numpy.random.permutation`.

    Parameters
    ----------
    x: int | torch.Tensor
        If an integer, returns a permutation of `torch.arange(x)`.
        If a tensor, returns a permuted copy along the first axis.

    Returns
    -------
    torch.Tensor
        A permuted tensor.

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.permutation(5)
    tensor([2, 4, 0, 1, 3])

    >>> import torch
    >>>
    >>> a = torch.arange(12).reshape(3, 4)
    >>> rnd.permutation(a).shape
    torch.Size([3, 4])

    """

    if isinstance(x, int):
        if x < 0:
            raise ValueError("`x` must be >= 0 when provided as an integer")
        return torch.randperm(x)

    if x.ndim == 0:
        return x.clone()

    perm = torch.randperm(x.shape[0], device=x.device)
    return x[perm]


def uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a uniform distribution on [low, high).

    Mirrors `numpy.random.uniform`.

    Parameters
    ----------
    low: float | torch.Tensor
        Lower bound.
    high: float | torch.Tensor
        Upper bound.
    size: tuple[int, ...] | None, optional
        Sample shape. If `None`, returns a scalar or broadcasted tensor.

    Returns
    -------
    torch.Tensor
        Samples drawn uniformly from [low, high).

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.uniform(0.0, 1.0)
    tensor(0.5488)

    >>> rnd.uniform(0.0, 1.0, (2, 3)).shape
    torch.Size([2, 3])

    """

    low_t = torch.as_tensor(low)
    high_t = torch.as_tensor(high)

    if size is None:
        base = torch.rand_like(
            torch.broadcast_tensors(low_t, high_t)[0]
        )
    else:
        base = torch.rand(size, dtype=low_t.dtype)

    return base * (high_t - low_t) + low_t


def normal(
    loc: float | torch.Tensor,
    scale: float | torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a normal distribution.

    Mirrors `numpy.random.normal`.

    Parameters
    ----------
    loc: float | torch.Tensor
        Mean of the distribution.
    scale: float | torch.Tensor
        Standard deviation (must be non-negative).
    size: tuple[int, ...] | None, optional
        Sample shape. If `None`, returns scalar or broadcasted tensor.

    Returns
    -------
    torch.Tensor
        Samples drawn from N(loc, scale^2).

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.normal(0.0, 1.0)
    tensor(1.5410...)

    >>> rnd.normal(0.0, 1.0, (2, 3)).shape
    torch.Size([2, 3])

    """

    loc_t = torch.as_tensor(loc)
    scale_t = torch.as_tensor(scale)

    if size is None:
        base_shape = torch.broadcast_shapes(
            loc_t.shape,
            scale_t.shape,
        )
        base = torch.randn(base_shape, dtype=loc_t.dtype)
    else:
        base = torch.randn(size, dtype=loc_t.dtype)

    return base * scale_t + loc_t


def poisson(
    lam: float | torch.Tensor,
    size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample from a Poisson distribution.

    Mirrors `numpy.random.poisson`.

    Parameters
    ----------
    lam: float | torch.Tensor
        Expected number of events (must be non-negative).
    size: tuple[int, ...] | None, optional
        Sample shape. If `None`, returns scalar or broadcasted tensor.

    Returns
    -------
    torch.Tensor
        Samples drawn from a Poisson distribution (int64).

    Examples
    --------
    >>> import deeptrack.backend.array_api_compat_ext.torch.random as rnd

    >>> rnd.poisson(3.0)
    tensor(4)

    >>> rnd.poisson(3.0, (2, 3)).shape
    torch.Size([2, 3])

    """

    lam_t = torch.as_tensor(lam, dtype=torch.float)

    if torch.any(lam_t < 0):
        raise ValueError("`lam` must be non-negative")

    if size is None:
        base = lam_t
    else:
        base = torch.broadcast_to(lam_t, size)

    samples = torch.poisson(base)

    return samples.to(torch.int64)
