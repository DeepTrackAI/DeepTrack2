import numpy as np
from deeptrack.features import Feature
from deeptrack.properties import SequentialProperty
from deeptrack.image import Image
from deeptrack.utils import as_list


class Sequence(Feature):

    __distributed__ = False

    def __init__(self, feature, *args, sequence_length=1, **kwargs):
        self.feature = feature
        super().__init__(*args, sequence_length=sequence_length, **kwargs)

        # Require update
        self.update()

    def get(self, input_list, sequence_length=None, **kwargs):
        return [self.feature.resolve(input_list,
                                sequence_length=sequence_length, 
                                sequence_step=sequence_step,
                                **kwargs)
                for sequence_step in range(sequence_length)]

    def update(self, **kwargs):
        super().update(**kwargs)

        sequence_length = self.properties["sequence_length"].current_value
        if "sequence_length" not in kwargs:
            kwargs["sequence_length"] = sequence_length

        self.feature.update(**kwargs)

        return self

        


def Sequential(feature, *args, **kwargs):
    
    properties = (kwargs, *args)

    for property_dict in properties:
        for property_name, sampling_rule in property_dict.items():

            if property_name in feature.properties:
                initializer = feature.properties[property_name].sampling_rule
            else:
                initializer = sampling_rule
                
            feature.properties[property_name] = SequentialProperty(initializer, sampling_rule)

    return feature


def BrownianMotion(feature, *args, **kwargs):

    def update_position(previous_value=(0, 0, 0), dt=1/30, diffusion_constant=0):
        return previous_value + diffusion_constant * np.random.randn(len(previous_value)) * dt

    return Sequential(feature, *args, position=update_position, **kwargs)