import numpy as np
class BaseOpticalDevice2D:
    
    def getPupil(self,
                    shape,
                    NA=0.7,
                    wavelength=0.66,
                    pixel_size=0.1):

        X = np.linspace(
            -pixel_size * shape[0] / 2,
            pixel_size * shape[0] / 2,
            num=shape[0],
            endpoint=True)
        
        Y = np.linspace(
            -pixel_size * shape[1] / 2,
            pixel_size * shape[1] / 2,
            num=shape[1],
            endpoint=True)

        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        
        sampling_frequency_x = 1/dx
        sampling_frequency_y = 1/dy

        x_radius = NA / (wavelength * sampling_frequency_x / shape[0])
        y_radius = NA / (wavelength * sampling_frequency_y / shape[1])

        W, H = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))

        pupilMask = ((W - shape[0] / 2) / x_radius) ** 2  + ((H - shape[1] / 2) / (y_radius) ) **2 <= 1
        
        pupil = pupilMask * (1 + 0j)
        return pupil

        

