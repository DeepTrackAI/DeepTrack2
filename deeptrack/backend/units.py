"""Unit management and conversion utilities for DeepTrack.

This module defines tools for handling unit conversions between optical and
simulation domains using the Pint library. It provides access to voxel sizes,
pixel scales, unit-aware contexts, and conversion utilities for numerical
values including NumPy arrays and PyTorch tensors.

Key Features
------------
- **Voxel and Pixel Scale Retrieval**

    Functions to retrieve voxel size and scaling factors between optical
    pixels and simulation pixels from the active unit registry.

- **Context-based Unit Definition**

    Functions to create Pint contexts that dynamically map pixel and
    simulation pixel units to metric units (meters), useful in simulations
    and training pipelines.

- **Flexible Quantity Conversion**

    A class for converting dictionaries of values to desired units,
    supporting floats, arrays, lists, and torch tensors.

Module Structure
----------------
Functions:

- `get_active_voxel_size() -> tuple[float, float, float]`

    Get the active voxel size in meters along the x, y, and z axes.

- `get_active_scale() -> tuple[float, float, float]`

    Get the scaling factors between simulation and optical pixels.

- `create_context(xpixel, ypixel, zpixel, xscale, yscale, zscale) -> Context`

    Create a unit context that defines how pixels map to meters.

Classes:

- `ConversionTable`: Convert dictionary values to target units.

    Utility class for converting numerical values (e.g., scalars, arrays)
    to specified target units using a user-defined conversion mapping.

Examples
--------
>>> from deeptrack.backend import units

Retrieve the active voxel size in meters:

>>> units.get_active_voxel_size()
(1e-06, 1e-06, 1e-06)

Retrieve the scaling factors between simulation and optical pixels:

>>> units.get_active_scale()
(1.0, 1.0, 1.0)

Create a custom unit context and use it to convert simulation pixels:

>>> from deeptrack import units_registry as u
>>>
>>> ctx = units.create_context(
...     xpixel=2e-6,
...     ypixel=1e-6,
...     zpixel=1e-6,
...     xscale=2,
...     yscale=1,
...     zscale=1,
... )
>>> with u.context(ctx):
...     print((1 * u.simulation_xpixel).to("meter"))
1e-06 meter

>>> print((1 * u.simulation_ypixel).to("meter"))
1e-06 meter

Use the ConversionTable to convert physical quantities to target units:

>>> conversion_table = units.ConversionTable(
...     length=(u.meter, u.micrometer),
...     time=(u.second, u.millisecond),
... )
>>> conversion_table.convert(length=1.2, time=0.5)
{'length': 1200000.0 <Unit('micrometer')>,
 'time': 500.0 <Unit('millisecond')>}

Support for PyTorch tensors:

>>> import torch
>>>
>>> conversion_table.convert(length=torch.tensor([1.0, 2.0]))
{'length': <Quantity([1000000. 2000000.], 'micrometer')>}

"""


from __future__ import annotations

from typing import Any

from numpy import ndarray
from pint import Quantity, Unit, Context

from deeptrack import TORCH_AVAILABLE
from deeptrack import units_registry as u

if TORCH_AVAILABLE:
    import torch

__all__ = [
    "get_active_voxel_size",
    "get_active_scale",
    "create_context",
    "ConversionTable",
]


def get_active_voxel_size() -> tuple[float, float, float]:
    """Get the active voxel size used for simulation, in meters.

    This function retrieves the size of one simulation voxel along the
    x, y, and z axes by converting from simulation pixel units (`sxpx`,
    `sypx`, and `szpx`) to meters using the DeepTrack unit registry.

    Returns
    -------
    tuple[float, float, float]
        The voxel size in meters as (x, y, z).

    Examples
    --------
    >>> from deeptrack.backend.units import get_active_voxel_size

    Get the standard voxel size in meters:
    >>> get_active_voxel_size()
    (1e-06, 1e-06, 1e-06)

    """

    grid_x = (1 * u.sxpx).to(u.m).magnitude
    grid_y = (1 * u.sypx).to(u.m).magnitude
    grid_z = (1 * u.szpx).to(u.m).magnitude

    return grid_x, grid_y, grid_z


def get_active_scale() -> tuple[float, float, float]:
    """Get the active scale between optical and simulation pixel units.

    This function computes the scaling factors along the x, y, and z axes
    that relate optical pixel units (`xpx`, `ypx`, `zpx`) to simulation
    pixel units (`sxpx`, `sypx`, `szpx`). The result can be used to convert
    between the two domains.

    Returns
    -------
    tuple[float, float, float]
        The scaling factors (x, y, z) such that:
        optical pixel = scale × simulation pixel

    Examples
    --------
    >>> from deeptrack.backend.units import get_active_scale

    Get the scale factors from simulation to optical domain:
    >>> get_active_scale()
    (1.0, 1.0, 1.0)

    """

    current_xscale = (1 * u.xpx / u.sxpx).to_base_units().magnitude or 1
    current_yscale = (1 * u.ypx / u.sypx).to_base_units().magnitude or 1
    current_zscale = (1 * u.zpx / u.szpx).to_base_units().magnitude or 1

    return current_xscale, current_yscale, current_zscale


def create_context(
    xpixel: float | None = None,
    ypixel: float | None = None,
    zpixel: float | None = None,
    xscale: int | None = None,
    yscale: int | None = None,
    zscale: int | None = None,
) -> Context:
    """Create a pint context for unit conversions between pixel and simulation units.

    This function returns a context mapping pixel units (`xpixel`, `ypixel`,
    `zpixel`) and simulation pixel units (`simulation_xpixel`, etc.) to 
    corresponding metric units (meters). If `xpixel`, `ypixel`, or `zpixel` 
    is not provided, the active values from the unit registry are used. 
    Likewise, if `xscale`, `yscale`, or `zscale` is provided, it multiplies
    the current scale to yield the new simulation scale.

    Parameters
    ----------
    xpixel, ypixel, zpixel : float or None, optional
        Size of a pixel in meters along x, y, and z axes. If None, the
        currently active values are used.
    xscale, yscale, zscale : int or None, optional
        Upscaling factors for internal simulation. If None, the current
        scaling is used.

    Returns
    -------
    Context
        A `pint.Context` object that defines pixel-to-meter mappings.

    Examples
    --------
    >>> from deeptrack.backend.units import create_context

    Create a context where one optical pixel is 1 µm and the
    y-direction is upscaled by a factor of 2 in the simulation:
    >>> ctx = create_context(
    ...     xpixel=1e-6,
    ...     ypixel=1e-6,
    ...     zpixel=1e-6,
    ...     yscale=2,
    ... )

    Load the unit registry:
    >>> from deeptrack import units_registry as u

    Use the context to convert 1 simulation y-pixel to meters:
    >>> with u.context(ctx):
    ...     print((1 * u.simulation_ypixel).to("meter"))
    5e-07 meter

    Outside the context:
    >>> print((1 * u.simulation_ypixel).to("meter"))
    1e-06 meter

    """

    current_xpixel = (1 * u.xpx).to(u.meter).magnitude
    current_ypixel = (1 * u.ypx).to(u.meter).magnitude
    current_zpixel = (1 * u.zpx).to(u.meter).magnitude

    current_xscale, current_yscale, current_zscale = get_active_scale()

    xpixel = xpixel if xpixel is not None else current_xpixel
    ypixel = ypixel if ypixel is not None else current_ypixel
    zpixel = zpixel if zpixel is not None else current_zpixel

    xscale = int(xscale * current_xscale) if xscale else int(current_xscale)
    yscale = int(yscale * current_yscale) if yscale else int(current_yscale)
    zscale = int(zscale * current_zscale) if zscale else int(current_zscale)

    ctx = Context()

    # Define optical pixel sizes
    ctx.redefine(f"pixel = {xpixel} meter")
    ctx.redefine(f"xpixel = {xpixel} meter")
    ctx.redefine(f"ypixel = {ypixel} meter")
    ctx.redefine(f"zpixel = {zpixel} meter")

    # Define simulation pixel sizes
    ctx.redefine(f"simulation_xpixel = {xpixel / xscale} meter")
    ctx.redefine(f"simulation_ypixel = {ypixel / yscale} meter")
    ctx.redefine(f"simulation_zpixel = {zpixel / zscale} meter")

    return ctx


class ConversionTable:
    """Convert a dictionary of values to desired units.

    The conversions are specified in the constructor with a dictionary, which
    is saved as `self.conversions`.
    Each key in the dictionary corresponds to the name of a property.
    The corresponding value is a tuple of two `Unit` objects:
    1) the default unit;
    2) the desired unit.

    To convert a dictionary of values to the desired units, the `convert()`
    method is called with the dictionary as an argument. The dictionary is
    converted to a dictionary of quantities in the desired units.
    If any value is not a quantity, it is assumed to be in the default unit.
    If a key is not in `self.conversions`, the corresponding value is left
    unchanged.

    Parameters
    ----------
    **conversions: dict[str, tuple[Unit, Unit]]
        A mapping from key names to unit pairs. Each pair defines the
        default and target units for that quantity.

    Attributes
    ----------
    conversions: dict[str, tuple[Unit, Unit]]
        A mapping from key names to unit pairs. Each pair defines the
        default and target units for that quantity.    

    Examples
    --------
    >>> from deeptrack.backend.units import ConversionTable

    Load the unit registry:
    >>> from deeptrack import units_registry as u

    Create a conversion table:
    >>> conversion_table = ConversionTable(
    ...     length=(u.meter, u.micrometer),
    ...     time=(u.second, u.millisecond)
    ... )

    Convert dictionary values:
    >>> conversion_table.convert(length=1.0, time=0.5)
    {'length': 1000000.0 <Unit('micrometer')>,
    'time': 500.0 <Unit('millisecond')>}

    """

    def __init__(
        self: ConversionTable,
        **conversions: dict[str, tuple[Unit, Unit]],
    ):
        """Initialize the conversion table with unit mappings.

        Parameters
        ----------
        **conversions: dict[str, tuple[Unit, Unit]]
            Keyword arguments where each key maps to a tuple of
            (default_unit, target_unit).

        """

        conversions: dict[str, tuple[Unit, Unit]]

        for key, value in conversions.items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"Conversion for '{key}' must be a tuple of two units"
            )
            assert isinstance(value[0], Unit) and isinstance(value[1], Unit), (
                f"Units for '{key}' must be instances of `Unit`"
            )

        self.conversions = conversions

    def convert(
        self: ConversionTable,
        **kwargs: Any,
    ) -> dict[str, Quantity | Any]:
        """Convert keyword arguments to their target units.

        Each keyword argument corresponding to a key in the conversion table is
        converted to its desired unit using Pint. Values are assumed to be in
        the default unit unless already specified as Pint Quantities.

        Values can be scalars, NumPy arrays, lists, tuples, or torch tensors.
        All are interpreted as being in the default unit unless already a Pint
        Quantity.

        Parameters
        ----------
        **kwargs: object
            Keyword arguments containing values to be converted.

        Returns
        -------
        dict[str, Quantity or Any]
            A dictionary with converted values. Keys not in the conversion
            table or with unsupported types are returned unchanged.

        """

        for key, value in self.conversions.items():
            if key not in kwargs:
                continue

            quantity = kwargs[key]

            # Skip unsupported types
            valid_types = (int, float, list, tuple, ndarray, Quantity)
            if TORCH_AVAILABLE:
                valid_types += (torch.Tensor,)  # type: ignore

            if not isinstance(quantity, valid_types):
                continue

            default_unit, desired_unit = value

            if (
                TORCH_AVAILABLE
                and torch.is_tensor(quantity)
                and quantity.requires_grad
            ):
                factor = (1 * default_unit).to(desired_unit).to_reduced_units()
                factor = factor.magnitude
                kwargs[key] = quantity * factor
                continue

            if (
                TORCH_AVAILABLE
                and isinstance(quantity, (list, tuple))
                and any(
                    torch.is_tensor(item) and item.requires_grad
                    for item in quantity
                )
            ):
                factor = (1 * default_unit).to(desired_unit).to_reduced_units()
                factor = factor.magnitude
                kwargs[key] = type(quantity)(item * factor for item in quantity)
                continue

            # Convert non-quantities to quantities in default units
            if not isinstance(quantity, Quantity):
                quantity = quantity * default_unit

            # Convert to desired unit, then reduce to base units
            kwargs[key] = quantity.to(desired_unit).to_reduced_units()

        return kwargs
