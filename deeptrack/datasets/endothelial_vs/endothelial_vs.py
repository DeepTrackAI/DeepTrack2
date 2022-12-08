"""endothelial_vs dataset."""

import tensorflow_datasets as tfds
import numpy as np

_DESCRIPTION = """
"""

_CITATION = """
@article{korczak2022dynamic,
  title={Dynamic live/apoptotic cell assay using phase-contrast imaging and deep learning},
  author={Korczak, Zofia and Pineda, Jesus and Helgadottir, Saga and Midtvedt, Benjamin and Goks{\"o}r, Mattias and Volpe, Giovanni and Adiels, Caroline B},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
"""


class EndothelialVs(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for endothelial_vs dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(None, None, 1), dtype="uint16"
                    ),
                    "label": tfds.features.Image(
                        shape=(None, None, 1), dtype="uint16"
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=10gqn0MwuxgyfJVWDZ6a8xduQuGAke4K3&export=download"
        )

        return {
            "train": self._generate_examples(
                path / "LiveDeadDataset" / "training"
            ),
            "test": self._generate_examples(
                path / "LiveDeadDataset" / "validation"
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        images_path = list(path.glob("*ch00*.tif"))

        for p in images_path:
            image = tfds.core.lazy_imports.tifffile.imread(p)
            image = np.expand_dims(image, axis=-1)

            label = tfds.core.lazy_imports.tifffile.imread(
                str(p).replace("ch00", "ch01")
            )
            label = np.expand_dims(label, axis=-1)

            yield p.name, {
                "image": image,
                "label": label,
            }
