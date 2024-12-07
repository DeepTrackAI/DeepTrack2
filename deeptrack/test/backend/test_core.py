import unittest

from deeptrack.backend.core import DeepTrackDataObject
from deeptrack.backend.core import DeepTrackDataDict
from deeptrack.backend.core import DeepTrackNode


class TestCore(unittest.TestCase):
    
    def test_trial(self):
        self.assetTrue(1 == 1)


if __name__ == "__main__":
    unittest.main()