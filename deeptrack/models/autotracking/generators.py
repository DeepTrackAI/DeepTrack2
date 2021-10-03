from ...generators import ContinuousGenerator
from ...augmentations import Affine
from ...features import Value
from .equivariances import (
    ScaleEquivariance,
    TranslationalEquivariance,
    Rotational2DEquivariance,
)
import numpy as np

DEFAULT_TRANSFORMATION_FUNCTION = Affine(
    translate=lambda: np.random.randn(2) * 2,
    rotate=lambda: np.random.rand() * 2 * np.pi * 2,
)

DEFAULT_EQUIVARIANCE = Rotational2DEquivariance(
    DEFAULT_TRANSFORMATION_FUNCTION.rotate
) >> TranslationalEquivariance(DEFAULT_TRANSFORMATION_FUNCTION.translate)


class AutoTrackGenerator(ContinuousGenerator):
    def __init__(
        self,
        data_feature,
        transformation_function=(DEFAULT_TRANSFORMATION_FUNCTION, DEFAULT_EQUIVARIANCE),
        num_outputs=2,
        **kwargs
    ):

        transformation_input = Value(
            lambda: (np.eye(num_outputs), np.zeros((num_outputs, 1)))
        )

        self.transformation_function = transformation_function[0] & (
            transformation_input >> transformation_function[1]
        )

        self.num_outputs = num_outputs
        super().__init__(data_feature, **kwargs)

    def construct_datapoint(self, image):
        sample = np.array(image)
        batch, matvec = zip(
            *[
                self.transformation_function.update().resolve(sample)
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

        A_1, b_1 = matvec_1
        A_2, b_2 = matvec_2

        A = np.linalg.inv(A_2) @ A_1
        b = np.linalg.inv(A_2) @ (b_2 - b_1)

        return A, b