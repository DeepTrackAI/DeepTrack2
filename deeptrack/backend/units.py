from numpy import ndarray
from pint import Quantity, Unit, Context
from .. import units as u


def get_active_voxel_size():
    """Gets the size of a voxel used for simulation."""
    grid_x = (1 * u.sxpx).to(u.m).magnitude
    grid_y = (1 * u.sypx).to(u.m).magnitude
    grid_z = (1 * u.szpx).to(u.m).magnitude
    return grid_x, grid_y, grid_z


def get_active_scale():
    """Gets the active scale difference between optical units and simulation units."""
    current_xscale = (1 * u.xpx / u.sxpx).to_base_units().magnitude or 1
    current_yscale = (1 * u.ypx / u.sypx).to_base_units().magnitude or 1
    current_zscale = (1 * u.zpx / u.szpx).to_base_units().magnitude or 1
    return (current_xscale, current_yscale, current_zscale)


def create_context(
    xpixel=None,
    ypixel=None,
    zpixel=None,
    xscale=None,
    yscale=None,
    zscale=None,
):
    """Creates a new context for unit conversions.

    If a value is None, the active value is used.
    If any (xyz)scale value is not none, they are multiplied with the active scale.

    Parameters
    ----------
    xpixel, ypixel, zpixel : float
        The size of pixels in each direction in meters
    xscale, yscale, zscale : int
        The upscale factor for internal simulations
    """

    current_xpixel = (1 * u.xpx).to(u.meter).magnitude
    current_ypixel = (1 * u.ypx).to(u.meter).magnitude
    current_zpixel = (1 * u.zpx).to(u.meter).magnitude
    current_xscale, current_yscale, current_zscale = get_active_scale()

    xpixel = xpixel if xpixel else current_xpixel
    ypixel = ypixel if ypixel else current_ypixel
    zpixel = zpixel if zpixel else current_zpixel
    xscale = int(xscale * current_xscale) if xscale else int(current_xscale)
    yscale = int(yscale * current_yscale) if yscale else int(current_yscale)
    zscale = int(zscale * current_zscale) if zscale else int(current_zscale)

    ctx = Context()
    ctx.redefine(f"pixel = {xpixel} meter")
    ctx.redefine(f"xpixel = {xpixel} meter")
    ctx.redefine(f"ypixel = {ypixel} meter")
    ctx.redefine(f"zpixel = {zpixel} meter")
    ctx.redefine(f"simulation_xpixel = {xpixel / xscale} meter")
    ctx.redefine(f"simulation_ypixel = {ypixel / yscale} meter")
    ctx.redefine(f"simulation_zpixel = {zpixel / zscale} meter")
    return ctx


class ConversionTable:
    """Convert a dictionary of values to the desired units.

    The conversions are specified in the constructor. Each key in the dictionary corresponds
    to the name of a property. The value of the key is a tuple of two units. The first unit is
    the default unit, and the second is the desired unit.

    To convert a dictionary of values to the desired units, the `convert` method is called with the
    dictionary as an argument. The dictionary is converted to a dictionary of quantities, and the
    quantities are converted to the desired units. If any value is not a quantity, it is assumed to
    be in the default unit. If a value with the same key is not in `self.conversions`, it is left unchanged.

    Parameters
    ----------
    conversions : dict
        The dictionary of conversions. Each key is the name of a property, and the value is a tuple of two
        units. The first unit is the default unit, and the second is the desired unit.
    """

    def __init__(self, **conversions):

        for value in conversions.values():
            assert isinstance(
                value, tuple
            ), "Each element in the conversion table needs to be a tuple of two units"
            assert (
                len(value) == 2
            ), "Each element in the conversion table needs to be a tuple of two units"
            assert isinstance(value[0], Unit) and isinstance(
                value[1], Unit
            ), "Each element in the conversion table needs to be a tuple of two units"
        self.conversions = conversions

    def convert(self, **kwargs):

        for key, val in self.conversions.items():

            if key not in kwargs:
                continue

            quantity = kwargs[key]

            if not isinstance(quantity, (int, float, list, tuple, ndarray, Quantity)):
                continue

            default_unit, desired_unit = val

            # If not quantity, assume default
            if not isinstance(quantity, Quantity):
                quantity = quantity * default_unit
            quantity = quantity.to(desired_unit)
            quantity = quantity.to_reduced_units()
            kwargs[key] = quantity

        return kwargs
