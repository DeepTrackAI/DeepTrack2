"""regression_holography_nanoparticles dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# TODO(regression_holography_nanoparticles): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(regression_holography_nanoparticles): BibTeX citation
_CITATION = """
"""


class RegressionHolographyNanoparticles(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for regression_holography_nanoparticles dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(regression_holography_nanoparticles): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Tensor(shape=(64, 64, 2), dtype=tf.float64),
                    "radius": tfds.features.Scalar(tf.float64),
                    "refractive_index": tfds.features.Scalar(tf.float64),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(
                "image",
                "radius",
                "refractive_index",
            ),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(regression_holography_nanoparticles): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=1LJqWYmLj93WYLKaLm_yQFmiR1FZHhf1r&export=download"
        )

        # TODO(regression_holography_nanoparticles): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path, "train"),
            "test": self._generate_examples(path, "test"),
        }

    def _generate_examples(self, path, split):
        """Yields examples."""
        # TODO(regression_holography_nanoparticles): Yields (key, example) tuples from the dataset

        if split == "train":
            data = np.load(path / "training_set.npy")
            radius = np.load(path / "training_radius.npy")
            refractive_index = np.load(path / "training_n.npy")
        elif split == "test":
            data = np.load(path / "validation_set.npy")
            radius = np.load(path / "validation_radius.npy")
            refractive_index = np.load(path / "validation_n.npy")
        else:
            raise ValueError("Split not recognized:", split)

        for idx in range(data.shape[0]):
            yield str(idx), {
                "image": data[idx],
                "radius": radius[idx],
                "refractive_index": refractive_index[idx],
            }
