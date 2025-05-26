from pathlib import Path
import unittest

from deeptrack.backend import config  # adjust to real import path


class BackendTestBase(unittest.TestCase):
    BACKEND = None

    @classmethod
    def setUpClass(cls):
        if cls.BACKEND is None:
            raise ValueError("BACKEND not set")
        config.set_backend(cls.BACKEND)
