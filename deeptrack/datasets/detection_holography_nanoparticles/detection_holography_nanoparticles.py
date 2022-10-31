"""detection_holography_nanoparticles dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# TODO(detection_holography_nanoparticles): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(detection_holography_nanoparticles): BibTeX citation
_CITATION = """
"""


class DetectionHolographyNanoparticles(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for detection_holography_nanoparticles dataset."""

    VERSION = tfds.core.Version("1.0.2")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(detection_holography_nanoparticles): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Tensor(
                        shape=(972, 729, 2), dtype=tf.float64
                    ),
                    "label": tfds.features.Tensor(shape=(None, 7), dtype=tf.float64),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
            disable_shuffling=True,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(detection_holography_nanoparticles): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=1uAZVr9bldhZhxuXAXvdd1-Ks4m9HPRtM&export=download"
        )

        # TODO(detection_holography_nanoparticles): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(detection_holography_nanoparticles): Yields (key, example) tuples from the dataset

        fields = path.glob("f*.npy")
        labels = path.glob("d*.npy")

        # sort the files
        fields = sorted(fields, key=lambda x: int(x.stem[1:]))
        labels = sorted(labels, key=lambda x: int(x.stem[1:]))

        for field, label in zip(fields, labels):
            field_data = np.load(field)
            field_data = np.stack((field_data.real, field_data.imag), axis=-1)
            yield field.stem, {
                "image": field_data,
                "label": np.load(label),
            }
