"""segmentation_ssTEM_drosophila dataset."""

import tensorflow_datasets as tfds
import numpy as np

_DESCRIPTION = """
We provide two image stacks where each contains 20 sections from serial section Transmission Electron Microscopy (ssTEM) 
of the Drosophila melanogaster third instar larva ventral nerve cord. 
Both stacks measure approx. 4.7 x 4.7 x 1 microns with a resolution of 4.6 x 4.6 nm/pixel and section 
thickness of 45-50 nm.

In addition to the raw image data, 
we provide for the first stack a dense labeling of neuron membranes (including orientation and junction),
mitochondria, synapses and glia/extracellular space. 
The first stack serves as a training dataset, and a second stack of the same dimension can be used as a test dataset.

labels: Series of merged labels including oriented membranes, membrane junctions,
mitochondria and synapses. The pixels are labeled as follows:
    0   -> membrane | (0°)
    32  -> membrane / (45°)
    64  -> membrane - (90°)
    96  -> membrane \ (135°)
    128 -> membrane "junction"
    159 -> glia/extracellular
    191 -> mitochondria
    223 -> synapse
    255 -> intracellular
"""

_CITATION = """
@article{Gerhard2013,
author = "Stephan Gerhard and Jan Funke and Julien Martel and Albert Cardona and Richard Fetter",
title = "{Segmented anisotropic ssTEM dataset of neural tissue}",
year = "2013",
month = "11",
url = "https://figshare.com/articles/dataset/Segmented_anisotropic_ssTEM_dataset_of_neural_tissue/856713",
doi = "10.6084/m9.figshare.856713.v1"
}
"""


class SegmentationSstemDrosophila(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for segmentation_ssTEM_drosophila dataset."""

    VERSION = tfds.core.Version("1.0.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.1": "Fix loading of tif images.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(segmentation_ssTEM_drosophila): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 1)),
                    "label": tfds.features.Image(shape=(None, None, 1)),
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

        path = dl_manager.download_and_extract(
            "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip"
        )
        return {
            "train": self._generate_examples(
                path / "groundtruth-drosophila-vnc-master" / "stack1"
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        raws = path / "raw"
        labels = path / "labels"

        raw_paths = list(raws.glob("*.tif"))
        label_paths = list(labels.glob("*.png"))

        for r, l in zip(raw_paths, label_paths):
            assert r.stem[-2:] == l.stem[-2:], "Mismatched raw and label files"

            image = tfds.core.lazy_imports.tifffile.imread(r)
            image = np.expand_dims(image, axis=-1)
            yield int(r.stem), {
                "image": image,
                "label": l,
            }
