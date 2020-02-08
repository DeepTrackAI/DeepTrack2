import numpy as np
from deeptrack.features import Feature
from deeptrack.image import Image
from deeptrack.utils import as_list



class Aberration(Feature):

    __distributed__ = True

    # Adds rho and theta of the pupil to the input.
    def _process_and_get(self, image_list, **kwargs):
        new_list = []
        for image in image_list:
            x = np.arange(image.shape[0]) - image.shape[0] / 2
            y = np.arange(image.shape[1]) - image.shape[1] / 2

            X, Y = np.meshgrid(x, y)
            rho = np.sqrt(X**2 + Y**2) 
            rho /= np.max(rho[image != 0])
            theta = np.arctan2(Y, X)

            
            new_list += super()._process_and_get([image], rho=rho, theta=theta, **kwargs)
        return new_list

# AMPLITUDE ABERRATIONS

class GaussianApodization(Aberration):
    
    # Flips the input image.
    def get(self, pupil, sigma=1, rho=None, **kwargs):
        return pupil * np.exp(-(rho / sigma) ** 2) 

# PHASE ABERRATIONS

class Zernike(Aberration):
    ''' Zernike phase aberration
    
    Multiplies the input by the phase mask as calculated by

    .. math :: exp(i \cdot (\sum c_iZ_i))

    '''
    def get(self, pupil, rho=None, theta=None, n=None, m=None, coefficient=None, **kwargs):
        m_list = as_list(m)
        n_list = as_list(n)
        coefficients = as_list(coefficient)

        assert len(m_list) == len(n_list), "The number of indices need to match"
        assert len(m_list) == len(coefficients), "The number of indices need to match the number of coefficients"

        pupil_bool = pupil != 0

        rho = rho[pupil_bool]
        theta = theta[pupil_bool]

        Z = 0

        for n, m, coefficient in zip(n_list, m_list, coefficients):
            if (n - m) % 2 or coefficient == 0:
                continue

            R = 0
            for k in range((n - np.abs(m)) // 2):
                R += ((-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * 
                     np.math.factorial((n - m) // 2 - k) * np.math.factorial((n + m) // 2 - k)) *
                     rho**(n - 2*k))
            
            if m > 0:
                R = R * np.cos(m * theta) * (np.sqrt(2*n + 2) * coefficient)
            elif m < 0:
                R = R * np.sin(-m * theta) * (np.sqrt(2*n + 2) * coefficient)
            else:
                R = R * (np.sqrt(n + 1) * coefficient)
            
            Z += R

        phase = np.exp(1j * Z)

        pupil[pupil_bool] *= phase
        
        return pupil

# COMMON ABERRATIONS

class Piston(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=0, m=0, coefficient=coefficient)

class VerticalTilt(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=1, m=-1, coefficient=coefficient)

class HorizontalTilt(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=1, m=1, coefficient=coefficient)

class ObliqueAstigmatism(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=2, m=-2, coefficient=coefficient)

class Defocus(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=2, m=0, coefficient=coefficient)

class Astigmatism(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=2, m=2, coefficient=coefficient)

class ObliqueTrefoil(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=3, m=-3, coefficient=coefficient)

class VerticalComa(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=3, m=-3, coefficient=coefficient)

class HorizontalComa(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=3, m=-3, coefficient=coefficient)

class Trefoil(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=3, m=-3, coefficient=coefficient)

class SphericalAberration(Zernike):
    def __init__(self, *args, coefficient=0):
        super().__init__(*args, n=4, m=0, coefficient=coefficient)




