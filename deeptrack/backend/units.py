from numpy import ndarray
from pint import Quantity, Unit, Context
from .. import units as u


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
