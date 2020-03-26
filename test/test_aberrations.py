import sys
sys.path.append("..") # Adds the module to path

import unittest

import deeptrack.aberrations as aberrations

from deeptrack.optics import Fluorescence


class TestAberrations(unittest.TestCase):
    
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
        aberrated_particle = aberrated_optics(particle)
        



if __name__ == '__main__':
    unittest.main()