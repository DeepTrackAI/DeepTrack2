import numpy as np
from DeepTrack.features import Feature
'''
    An optical device that images a fourier representation of the field. 

    Main responsibility is storing parameters of the optical system, as well
    as imaging an intensity map.
'''
class Optics(Feature):


    def __call__(self, feature):
        return feature + self

        
    def get(self, image, **kwargs):
        mode = kwargs.get("mode", "incoherent")
        
        pupil = self.pupil(image.shape, **kwargs)

        # If incoherent, get squared pupil
        if mode == "incoherent":
            psf = np.square(np.abs(np.fft.ifft2(pupil)))
            pupil = np.fft.fft2(psf)

        convolved_fourier_field = image * pupil
        field = np.fft.ifft2(convolved_fourier_field)

        # TODO: FFT does not propagate properties correctly
        field.properties = convolved_fourier_field.properties

        # If coherent, get squared field
        if mode == "coherent":
             field = np.square(np.abs(field))

        return field


class BaseOpticalDevice2D(Optics):
    def __init__(self,
                    NA=0.7,
                    wavelength=0.66e-6,
                    refractive_index_medium=1.33,
                    pixel_size=0.1e-6,
                    defocus=0,
                    upscale=2,
                    mode="incoherent",
                    ROI=(0,0)):
        
        super().__init__(
            NA=NA,
            mode = mode,
            upscale=upscale,
            wavelength=wavelength,
            refractive_index_medium=refractive_index_medium,
            pixel_size=pixel_size,
            ROI=ROI
        )


    # TODO: Split into smaller functions
    def pupil(self, shape, 
                NA=None,
                wavelength=None,
                refractive_index_medium=None,
                pixel_size=None,
                defocus=None,
                upscale=None,
                mode=None,
                ROI=None):

        shape = np.array(shape) 
        upscaled_shape = shape*upscale

        # Pupil radius
        R = NA / wavelength * pixel_size
        x_radius = R * upscaled_shape[0]
        y_radius = R * upscaled_shape[1]

        x = (np.linspace(-(upscaled_shape[0] / 2), upscaled_shape[0] / 2 - 1, upscaled_shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(upscaled_shape[1] / 2), upscaled_shape[1] / 2 - 1, upscaled_shape[1])) / y_radius + 1e-8
        
        W, H = np.meshgrid(x, y)
        RHO = W**2 + H**2

        # Pupil magnitude. TODO: Throws warning
        pupil = (RHO < 1) / ((1 - NA**2 / refractive_index_medium**2 * RHO)**0.25) 


        # Defocus
        if defocus != 0:
            z_shift = 2 * np.pi * refractive_index_medium/wavelength * (1 - NA**2 / refractive_index_medium**2 * RHO)**0.5 * pixel_size * defocus
        else:
            z_shift = 0

        # Downsample the upsampled pupil
        if upscale > 1:
            pupil = np.reshape(pupil, (shape[0], upscale, shape[1], upscale)).mean(axis=(3,1))
            z_shift = np.reshape(z_shift, (shape[0], upscale, shape[1], upscale)).mean(axis=(3,1))
        

        pupil = pupil*np.exp(1j * z_shift)
        
        pupil[np.isnan(pupil)] = 0
    
        return np.fft.fftshift(pupil)

class ZernikeAberration:
    def __init__(self,
                    Z_index,
                    Z_coefficient):
        
        assert len(Z_index) == len(Z_coefficient), "Z_index and Z_coefficient must have same length"
