"""segmentation_fluorescence_u2os dataset."""

import tensorflow_datasets as tfds
from . import segmentation_fluorescence_u2os


class SegmentationFluorescenceU2osTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for segmentation_fluorescence_u2os dataset."""
  # TODO(segmentation_fluorescence_u2os):
  DATASET_CLASS = segmentation_fluorescence_u2os.SegmentationFluorescenceU2os
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
