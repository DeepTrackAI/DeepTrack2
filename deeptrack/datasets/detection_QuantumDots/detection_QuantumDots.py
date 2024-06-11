"""detection_QuantumDots dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# TODO(detection_QuantumDots): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Sequential images of quantum dots in a fluorescent microscope. The dataset is unlabeled. 
"""

# TODO(detection_QuantumDots): BibTeX citation
_CITATION = """
"""


class DetectionQuantumdots(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for detection_QuantumDots dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(detection_QuantumDots): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(
                        shape=(1200, 1200, 1),
                        dtype=tf.uint16,
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(detection_QuantumDots): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=1naaoxIaAU1F_rBaI-I1pB1K4Sp6pq_Jv&export=download"
        )

        # TODO(detection_QuantumDots): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "QuantumDots"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        tifpath = path / "Qdots.tif"

        image_stack = tfds.core.lazy_imports.tifffile.imread(tifpath)
        image_stack = np.expand_dims(image_stack, axis=-1)
        for i, image in enumerate(image_stack):
            yield str(i), {
                "image": image,
            }
