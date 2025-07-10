# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import unittest
import os
import shutil

from deeptrack.sources.folder import ImageFolder


class TestFolder(unittest.TestCase):

    def setUp(self):
        self.root = "temp_test_data"
        self.splits = ["train", "val"]
        self.classes = ["cat", "dog", "bird"]

        for split in self.splits:
            for cls in self.classes:
                path = os.path.join(self.root, split, cls)
                os.makedirs(path, exist_ok=True)
                for i in range(2):
                    file_path = os.path.join(path, f"img{i}.jpg")
                    with open(file_path, "w") as f:
                        f.write("")

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_ImageFolder_basic(self):
        dataset = ImageFolder(os.path.join(self.root, "train"))
        self.assertEqual(len(dataset), 6)
        self.assertSetEqual(set(dataset.classes), set(self.classes))
        self.assertTrue(all(os.path.isfile(item["path"]) for item in dataset))

    def test_ImageFolder_split(self):
        combined = ImageFolder(self.root)
        train, val = combined.split("train", "val")

        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 6)

        train_paths = [item["path"] for item in train]
        val_paths = [item["path"] for item in val]

        self.assertTrue(all("train" in path for path in train_paths))
        self.assertTrue(all("val" in path for path in val_paths))

    def test_ImageFolder_label_mapping(self):
        dataset = ImageFolder(os.path.join(self.root, "train"))

        for label in dataset["label"]:
            name = dataset.label_to_name(label)
            self.assertIsInstance(name, str)
            self.assertIn(name, dataset.classes)
            back_label = dataset.name_to_label(name)
            self.assertEqual(label, back_label)


if __name__ == "__main__":
    unittest.main()
