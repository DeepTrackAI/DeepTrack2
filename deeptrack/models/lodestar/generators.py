from ...generators import ContinuousGenerator
from ...augmentations import Affine
from ...features import Value
from .equivariances import (
    ScaleEquivariance,
    TranslationalEquivariance,
    Rotational2DEquivariance,
)
import numpy as np

# Define default equivariances
a = Affine(translate=lambda: np.random.randn(2) * 2)
b = Affine(
    rotate=lambda: np.random.rand() * 2 * np.pi * 2,
)

DEFAULT_TRANSFORMATION_FUNCTION = b >> a

DEFAULT_EQUIVARIANCE = Rotational2DEquivariance(b.rotate) >> TranslationalEquivariance(
    a.translate
)


class LodeSTARGenerator(ContinuousGenerator):
    """Data generator for use with an LodeSTARer.

    Parameters
    ----------
    feature : Feature
        DeepTrack feature returning crops of single objects.
    num_outputs : int
        Number of values the model is expected to predict (not including the weight-map)
    transformation_function : Feature, Feature, optional
        Tuple of features defining transformations applied to each crop as well as the corresponding equivariance.


    """

    def __init__(
        self,
        feature,
        num_outputs=2,
        transformation_function=(DEFAULT_TRANSFORMATION_FUNCTION, DEFAULT_EQUIVARIANCE),
        **kwargs
    ):

        transformation_input = Value(
            lambda: (np.eye(num_outputs), np.zeros((num_outputs, 1)))
        )

        self.transformation_function = transformation_function[0] & (
            transformation_input >> transformation_function[1]
        )
        self.num_outputs = num_outputs
        super().__init__(feature, **kwargs)

    def construct_datapoint(self, image):
        sample = np.array(image)
        batch, matvec = zip(
            *[
                (*self.transformation_function.update().resolve(sample),)
                for _ in range(self.batch_size)
            ]
        )
        return super().construct_datapoint((batch, matvec))

    def __getitem__(self, idx):
        batch, matvec = self.current_data[idx]["data"]
        A, b = zip(*[self.get_transform_matrix(matvec[0], b) for b in matvec])
        return np.array(batch), (np.array(b), np.array(A))

    def __len__(self):
        return len(self.current_data)

    def get_transform_matrix(self, matvec_1, matvec_2):
        A_2, b_2 = matvec_2
        return A_2, -b_2