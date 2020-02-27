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
            **kwargs):

        return np.ones((1, 1, 1)) * 1.0

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
            radius=None,
            rotation=0,
            voxel_size=None,
            upsample=4,
            **kwargs):

        if not isinstance(radius, (tuple, list, np.ndarray)):
            radius = (radius, radius)
        
        x_rad = radius[0] / voxel_size[0] * upsample
        y_rad = radius[1] / voxel_size[1] * upsample

        x_ceil = int(np.ceil(x_rad))
        y_ceil = int(np.ceil(y_rad))

        x_ceil = np.max((x_ceil, y_ceil))
        y_ceil = np.max((x_ceil, y_ceil))

        to_add = (upsample - ((x_ceil * 2) % upsample)) % upsample


        X, Y = np.meshgrid(np.arange(-x_ceil, x_ceil + to_add), np.arange(-y_ceil, y_ceil + to_add))

        if rotation != 0:
            Xt =  (X * np.cos(-rotation) + Y * np.sin(-rotation))
            Yt = (-X * np.sin(-rotation) + Y * np.cos(-rotation))
            X = Xt
            Y = Yt 


        mask = ((X * X) / (x_rad * x_rad) + (Y * Y) / (y_rad * y_rad) < 1)

        if upsample != 1:
            mask = np.reshape(mask, (mask.shape[0] // upsample, upsample, mask.shape[1] // upsample, upsample)).mean(axis=(3, 1))

        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[:, ~np.all(mask == 0, axis=0)]

        mask = np.expand_dims(mask, axis=-1)

        return mask


class Sphere(Scatterer):
    ''' Generates spherical scatterers

    Parameters
    ----------
    position               
        The position of the point particle. Defined as (0,0) in the
        upper left corner.
    radius
        The radius of the sphere, in meters
    '''

    def get(
            self, 
            image,
            radius=None,
            voxel_size=None,
            upsample=4,
            **kwargs):



        rad = radius / voxel_size * upsample

        rad_ceil = np.ceil(rad)

        to_add = (upsample - ((rad_ceil * 2) % upsample)) % upsample

        x = np.arange(-rad_ceil[0], rad_ceil[0] + to_add[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1] + to_add[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2] + to_add[2])

        X, Y, Z = np.meshgrid((x / rad[0])**2, (y / rad[1])**2, (z / rad[2])**2)



        mask = X + Y + Z < 1

        if upsample != 1:
            mask = np.reshape(mask, 
                                (mask.shape[0] // upsample, upsample, 
                                 mask.shape[1] // upsample, upsample,
                                 mask.shape[2] // upsample, upsample)).mean(axis=(5, 3, 1))

        mask = mask[~np.all(mask == 0, axis=(1, 2))]
        mask = mask[:, ~np.all(mask == 0, axis=(0, 2))]
        mask = mask[:, :, ~np.all(mask == 0, axis=(0, 1))]

        return mask


class Ellipsoid(Scatterer):
    ''' Generates ellipsoidal scatterer

    Parameters
    ----------
    position               
        The position of the point particle. Defined as (0,0) in the
        upper left corner.
    radius
        The radius of the ellipsoid along the principal axes in meters. Can be a single value to a vector 
        of length 2 or 3.
    rotation
        The rotation about the three axes in radians. Can be a single value to a vector of length 2 or 3.
    upsample
        During the calculation of the scatterer, the pixelation is increased by this factor. The result is 
        then downsampled by averaging.
    '''
    
    def _process_properties(self, propertydict):
        '''Preprocess the input to the method .get()

        Ensures that the radius and the rotation properties both are arrays of
        length 3.

        If the radius is a single value, the particle is made a sphere
        If the radius are two values, the smallest value is appended as the third value

        The rotation vector is padded with zeros until it is of length 3

        '''


        # Ensure radius has three values
        radius = np.array(propertydict["radius"])
        if radius.ndim == 0:
            radius = np.array([radius])
        if radius.size == 1:
            # If only one value, assume sphere
            radius = (*radius,) * 3
        elif radius.size == 2:
            # If two values, duplicate the minor axis
            radius = (*radius, np.min(radius[-1]))
        elif radius.size == 3:
            # If three values, convert to tuple for consistency
            radius = (*radius, )
        propertydict["radius"] = radius

        # Ensure rotation has three values
        rotation = np.array(propertydict["rotation"])
        if rotation.ndim == 0:
            rotation = np.array([rotation])

        if rotation.size == 1:
            # If only one value, pad with two zeros
            rotation = (*rotation, 0, 0) 
        elif rotation.size == 2:
            # If two values, pad with one zero
            rotation = (*rotation, 0)
        elif rotation.size == 3:
            # If three values, convert to tuple for consistency
            rotation = (*rotation, )
        propertydict["rotation"] = rotation

        return propertydict

    def get(self,
            image,
            radius=None,
            rotation=None,
            voxel_size=None,
            upsample=4,
            **kwargs):


        radius_in_pixels = radius / voxel_size * upsample
        max_rad = np.max(radius) / voxel_size * upsample
        rad_ceil = np.ceil(max_rad)
        to_add = (upsample - ((rad_ceil * 2) % upsample)) % upsample

        # Create a rotated grid of points
        x = np.arange(-rad_ceil[0], rad_ceil[0] + to_add[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1] + to_add[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2] + to_add[2])

        X, Y, Z = np.meshgrid(x, y, z)

        mask = X + Y + Z < 1
        cos = np.cos(rotation)
        sin = np.sin(rotation)

        XR = cos[0] * cos[1] * X + (cos[0] * sin[1] * sin[2] - sin[0] * cos[2]) * Y + (cos[0] * sin[1] * cos[2] + sin[0] * sin[2]) * Z
        YR = sin[0] * cos[1] * X + (sin[0] * sin[1] * sin[2] + cos[0] * cos[2]) * Y + (sin[0] * sin[1] * cos[2] - cos[0] * sin[2]) * Z
        ZR = -sin[1] * X + cos[1] * sin[2] * Y + cos[1] * cos[2] * Z 

        mask = (XR / radius_in_pixels[0])**2 + (YR / radius_in_pixels[1])**2 + (ZR / radius_in_pixels[2])**2 < 1

        # Downsample
        if upsample != 1:
            mask = np.reshape(mask, 
                                (mask.shape[0] // upsample, upsample, 
                                 mask.shape[1] // upsample, upsample,
                                 mask.shape[2] // upsample, upsample)).mean(axis=(5, 3, 1))

        # Crop all rows/columns that contain no non-zero entry
        mask = mask[~np.all(mask == 0, axis=(1, 2))]
        mask = mask[:, ~np.all(mask == 0, axis=(0, 2))]
        mask = mask[:, :, ~np.all(mask == 0, axis=(0, 1))]

        return mask
