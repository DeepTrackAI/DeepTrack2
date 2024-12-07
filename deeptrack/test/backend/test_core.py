# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack.backend import core


class TestCore(unittest.TestCase):

    def test_DeepTrackDataObject(self):
        dataobj = core.DeepTrackDataObject()
        dataobj.store(1)
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)

        dataobj.invalidate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), False)

        dataobj.validate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)


if __name__ == "__main__":
    unittest.main()
