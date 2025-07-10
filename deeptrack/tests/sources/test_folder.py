# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import os
import shutil
import unittest

from deeptrack.sources import folder


class TestFolder(unittest.TestCase):

    def setUp(self):
        self.root_dir = "temp_test_dir"
        self.classes = ["cat", "dog", "bird"]

        os.makedirs(self.root_dir, exist_ok=True)

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(3):
                file_path = os.path.join(class_dir, f"image_{i}.jpg")
                with open(file_path, "w") as f:
                    f.write("")

    def tearDown(self):
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_ImageFolder_basic(self):
        dataset = folder.ImageFolder(self.root_dir)

        self.assertEqual(len(dataset), 9)
        self.assertCountEqual(dataset.classes, self.classes)

    def test_ImageFolder_get_category_name(self):
        dataset = folder.ImageFolder(self.root_dir)
        for item in dataset:
            path = item["path"]
            name = dataset.get_category_name(path, 0)
            self.assertIn(name, self.classes)

    def test_ImageFolder_label_mapping(self):
        dataset = folder.ImageFolder(self.root_dir)

        for item in dataset:
            label = item["label"]
            name = dataset.label_to_name(label)
            idx = dataset.name_to_label(name)
            self.assertEqual(idx, label)

    def test_ImageFolder_split(self):
        dataset = folder.ImageFolder(self.root_dir)

        cat_ds, dog_ds = dataset.split("cat", "dog")

        cat_names = set([item["label_name"] for item in cat_ds])
        dog_names = set([item["label_name"] for item in dog_ds])

        self.assertEqual(cat_names,
                         {"image_0.jpg", "image_1.jpg", "image_2.jpg"})
        self.assertEqual(dog_names,
                         {"image_0.jpg", "image_1.jpg", "image_2.jpg"})
        self.assertEqual(len(cat_ds), 3)
        self.assertEqual(len(dog_ds), 3)


if __name__ == "__main__":
    unittest.main()
