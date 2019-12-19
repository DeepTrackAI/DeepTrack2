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

from deeptrack.features import Feature, MERGE_STRATEGY_APPEND
from deeptrack.image import Image

class Scatterer(Feature):
    '''Base class for scatterers.

    A scatterer defines the scattered complex field at each pixel.

    '''

    __list_merge_strategy__ = MERGE_STRATEGY_APPEND
    __distributed__ = False

    def __init__(self, *args, position_unit="meter", **kwargs):
        super().__init__(*args, position_unit=position_unit, **kwargs)


    def _process_properties(self, properties):
        if "position" in properties:
            if properties["position_unit"] == "meter":
                properties["position"] = np.array(properties["position"]) / np.array(properties["voxel_size"])[:len(properties["position"])]

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
            intensity=None,
            **kwargs):



        return np.ones((1, 1, 1))*intensity

import matplotlib.pyplot as plt

class Ellipse(Scatterer):
    ''' Generates ellipsoidal scatterers

    Parameters
    ----------
    position               
        The position of the point particle. Defined as (0,0) in the
        upper left corner.
    intensity               
        The magnitude of the complex field scattered by the point particle. 
        Mathematically the integral over the delta distribution. 
    radius
        If number, the radius of a circle. If a list or tuple, the x and y radius of the particle.
    rotation
        If defined, rotates the ellipsoid by this amount in radians
    '''

    def get(
            self, 
            image,
            intensity=None,
            radius=None,
            rotation=0,
            pixel_size=None,
            voxel_size=None,
            **kwargs):

        if not isinstance(radius, (tuple, list, np.ndarray)):
            radius = (radius, radius)
        
        x_rad = radius[0] / voxel_size[0]
        y_rad = radius[1] / voxel_size[1]

        x_ceil = int(np.ceil(x_rad))
        y_ceil = int(np.ceil(y_rad))

        X, Y = np.meshgrid(np.arange(-x_ceil, x_ceil), np.arange(-y_ceil, y_rad))
        if rotation != 0:
            Xt =  (X * np.cos(rotation) + Y * np.sin(rotation))
            Yt = (-X * np.sin(rotation) + Y * np.cos(rotation))
            X = Xt
            Y = Yt 


        mask = ((X * X) / (x_rad * x_rad) + (Y * Y) / (y_rad * y_rad) < 1) * 1.0 * intensity

        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[:, ~np.all(mask == 0, axis=0)]

        mask = np.expand_dims(mask, axis=-1)

        return mask