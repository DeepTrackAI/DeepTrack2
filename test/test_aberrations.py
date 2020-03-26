import sys
sys.path.append("..") # Adds the module to path

import unittest

import deeptrack.aberrations as aberrations

from deeptrack.scatterers import PointParticle
from deeptrack.optics import Fluorescence
from deeptrack.image import Image



class TestAberrations(unittest.TestCase):
    
    particle = PointParticle(
        position=(32, 32),
        position_unit="pixel",
        intensity=1
    )

    
    def testGaussianApodization(self):
        aberrated_optics = Fluorescence(
            NA = 0.3,
            resolution=1e-6,
            magnification=10,
            wavelength=530e-9,
            output_region=(0, 0, 64, 64),
            padding=(64, 64, 64, 64),
            aberration=aberrations.GaussianApodization(sigma=0.5)
        )
        aberrated_particle = aberrated_optics(self.particle)
        for z in (-100, 0, 100):
            im = aberrated_particle.resolve(z=z)
            self.assertIsInstance(im, Image)        
            self.assertEqual(im.shape, (64, 64, 1))        



if __name__ == '__main__':
    unittest.main()