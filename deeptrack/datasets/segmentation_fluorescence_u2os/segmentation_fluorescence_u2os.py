"""segmentation_fluorescence_u2os dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

# TODO(segmentation_fluorescence_u2os): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(segmentation_fluorescence_u2os): BibTeX citation
_CITATION = """
"""


class SegmentationFluorescenceU2os(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for segmentation_fluorescence_u2os dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(segmentation_fluorescence_u2os): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(
                        shape=(None, None, 1), dtype=tf.uint16
                    ),
                    "label": tfds.features.Image(
                        shape=(None, None, 4),
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path_to_images = (
            dl_manager.download_and_extract(
                "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"
            )
            / "images"
        )

        path_to_masks = (
            dl_manager.download_and_extract(
                "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip"
            )
            / "masks"
        )

        path_to_metadata = (
            dl_manager.download_and_extract(
                "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip"
            )
            / "metadata"
        )

        return {
            "train": self._generate_examples(
                path_to_metadata / "training.txt", path_to_images, path_to_masks
            ),
            "test": self._generate_examples(
                path_to_metadata / "test.txt", path_to_images, path_to_masks
            ),
            "validation": self._generate_examples(
                path_to_metadata / "validation.txt", path_to_images, path_to_masks
            ),
        }

    def _generate_examples(self, path, images_path, masks_path):
        """Yields examples."""
        with open(path, "r") as f:
            for line in f:
                filename = line.strip()

                if filename == "":
                    continue

                path_to_image = images_path / filename.replace(".png", ".tif")
                path_to_label = masks_path / filename

                image = tfds.core.lazy_imports.tifffile.imread(path_to_image)[..., None]

                yield filename, {
                    "image": image,
                    "label": path_to_label,
                }
