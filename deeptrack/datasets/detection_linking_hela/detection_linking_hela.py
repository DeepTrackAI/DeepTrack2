"""detection_linking_Hela dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd

_DESCRIPTION = """
This dataset includes tracking data from DIC-C2DH-HELA (provided by the sixth edition of the Cell Tracking Challenge).
It consists of two dataframes: ``nodes`` and ``parenthood``. ``nodes`` contains information about the individual 
cells, while "parenthood" includes information on the lineage of the cells.
"""

_CITATION = """
@article{pineda2022geometric,
  title={Geometric deep learning reveals the spatiotemporal fingerprint of microscopic motion},
  author={Pineda, Jes{\'u}s and Midtvedt, Benjamin and Bachimanchi, Harshith and No{\'e}, Sergio and Midtvedt, Daniel and Volpe, Giovanni and Manzo, Carlo},
  journal={arXiv preprint arXiv:2202.06355},
  year={2022}
}
"""


class DetectionLinkingHela(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for detection_linking_Hela dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        NODE_FEATURES = self.get_node_features()
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "nodes": tfds.features.FeaturesDict(
                        {
                            **{
                                key: tfds.features.Tensor(
                                    shape=(None,), dtype=NODE_FEATURES[key]
                                )
                                for key in NODE_FEATURES.keys()
                            },
                        }
                    ),
                    "parenthood": tfds.features.FeaturesDict(
                        {
                            "child": tfds.features.Tensor(
                                shape=(None,), dtype=tf.int32
                            ),
                            "parent": tfds.features.Tensor(
                                shape=(None,), dtype=tf.int32
                            ),
                        }
                    ),
                    "images": tfds.features.Tensor(
                        shape=(84, 512, 512, 1), dtype=tf.float64
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=1tMKjRPutjKGf7YVs5aPrQ625CxcvnC9C&export=download"
        )

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(
                path / "detection_linking_hela", "train"
            ),
            "test": self._generate_examples(
                path / "detection_linking_hela", "test"
            ),
        }

    def _generate_examples(self, path, split):
        """Yields examples."""

        # Load data
        nodes, parenthood, images = (
            pd.read_csv(path / split / "nodesdf.csv"),
            pd.read_csv(path / split / "parenthood.csv"),
            np.load(path / split / "images.npy"),
        )

        yield "_", {
            "nodes": {**nodes.to_dict("list")},
            "parenthood": {**parenthood.to_dict("list")},
            "images": images * 1.0,
        }

    def get_node_features(self):
        return {
            "frame": tf.int32,
            "label": tf.int32,
            "centroid-0": tf.float32,
            "centroid-1": tf.float32,
            "area": tf.float32,
            "mean_intensity": tf.float32,
            "perimeter": tf.float32,
            "eccentricity": tf.float32,
            "solidity": tf.float32,
            "set": tf.float32,
            "parent": tf.int32,
            "solution": tf.float32,
        }
