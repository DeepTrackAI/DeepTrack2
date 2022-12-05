"""dmdataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import scipy

_DESCRIPTION = """
"""

_CITATION = """
"""


class Dmdataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dmdataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        NODE_FEATURES = self.get_features()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "graph": tfds.features.FeaturesDict(
                        {
                            **{
                                key: tfds.features.Tensor(
                                    shape=feature[0],
                                    dtype=feature[1],
                                )
                                for key, feature in NODE_FEATURES[
                                    "graph"
                                ].items()
                            },
                        }
                    ),
                    "labels": tfds.features.FeaturesDict(
                        {
                            **{
                                key: tfds.features.Tensor(
                                    shape=feature[0],
                                    dtype=feature[1],
                                )
                                for key, feature in NODE_FEATURES[
                                    "labels"
                                ].items()
                            },
                        }
                    ),
                    "sets": tfds.features.FeaturesDict(
                        {
                            **{
                                key: tfds.features.Tensor(
                                    shape=feature[0],
                                    dtype=feature[1],
                                )
                                for key, feature in NODE_FEATURES[
                                    "sets"
                                ].items()
                            },
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://drive.google.com/u/1/uc?id=19vplN2lbKo4KAmv4NRU2qr3NSlzxFzrx&export=download"
        )

        return {
            "train": self._generate_examples(
                os.path.join(path, "dmdataset", "training")
            ),
            "test": self._generate_examples(
                os.path.join(path, "dmdataset", "validation")
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        data = [{}, {}, {}]
        for i, subdict in enumerate(self.get_features().values()):
            files = (*subdict.keys(),)

            for file in files:
                data_elem = scipy.sparse.load_npz(
                    os.path.join(path, file + ".npz")
                ).toarray()
                data_elem = (
                    data_elem[0] if data_elem.shape[0] == 1 else data_elem
                )

                data[i][file] = data_elem

        yield "key", {
            "graph": data[0],
            "labels": data[1],
            "sets": data[2],
        }

    def get_features(self):
        return {
            "graph": {
                "frame": [(None, 1), tf.float64],
                "node_features": [(None, 3), tf.float64],
                "edge_features": [(None, 1), tf.float64],
                "edge_indices": [(None, 2), tf.int64],
                "edge_dropouts": [(None, 2), tf.float64],
            },
            "labels": {
                "node_labels": [(None,), tf.float64],
                "edge_labels": [(None,), tf.float64],
                "global_labels": [(None, 3), tf.float64],
            },
            "sets": {
                "node_sets": [(None, 2), tf.int64],
                "edge_sets": [(None, 3), tf.int64],
            },
        }
