'''Features modelling light scattering object

Instances of implementations of the class Scatterers need to be
wrapped by an instance of the Optics class. This provides the feature
access to the optical properties. 

Scatterers should generate the complex field at each pixel. 

Contains
--------

abstract class Scatterer
    Base class for scatterers

class PointParticle
    Generates point particles

'''


import numpy as np

from deeptrack.features import Feature

class Scatterer(Feature):
    '''Base class for scatterers.

    A scatterer defines the scattered complex field at each pixel.

    '''

    def __init__(self, position_unit="meter", **kwargs):
        super().__init__(position_unit=position_unit, **kwargs)


    def _process_properties(self, properties):
        if "position" in properties:
            if properties["position_unit"] == "meter":
                properties["position"] = np.array(properties["position"]) / properties["pixel_size"]

        return properties






class PointParticle(Scatterer):
    '''Generates a point particle

    A point particle is approximated by the size of a pixel. For subpixel
    positioning, the intensity is interpolated linearly.

    Parameters
    ----------
    intensity
        The magnitude of the complex field scattered by the point particle.
        Mathematically the integral over the delta distribution.
    position
        The pixel position of the point particle. Defined as (0,0) in the
        upper left corner.

    '''

    def get(self,
            image,
            position=None,
            intensity=None,
            **kwargs):

        y_0 = int(np.floor(position[0]))
        y_p = position[0] - y_0
        x_0 = int(np.floor(position[1]))
        x_p = position[1] - x_0

        try:
            image[x_0, y_0] = np.sqrt(intensity * ((1 - x_p) * (1 - y_p)))
            image[x_0 + 1, y_0] = np.sqrt(intensity * (x_p * (1 - y_p)))
            image[x_0, y_0 + 1] = np.sqrt(intensity * ((1 - x_p) * y_p))
            image[x_0 + 1, y_0 + 1] = np.sqrt(intensity * (x_p * y_p))
        except IndexError:
            pass

        return image
