from pint import Quantity, Unit, Context
from .. import units as u


class ConversionTable:
    __default_context = Context()

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

    def convert(self, context=None, **kwargs):

        for key, val in self.conversions.items():

            if key not in kwargs:
                continue

            quantity = kwargs[key]

            default_unit, desired_unit = val

            # If not quantity, assume default
            if not isinstance(quantity, Quantity):
                quantity = quantity * default_unit
            if context:
                quantity = quantity.to(desired_unit, context)
            else:
                quantity = quantity.to(desired_unit)
            quantity = quantity.to_reduced_units()

            kwargs[key] = quantity

        return kwargs
