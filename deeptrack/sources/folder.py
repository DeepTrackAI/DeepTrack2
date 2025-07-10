"""Data sources from images organized in a directory structure.

This module provides the `ImageFolder` class, which enables structured access
to images stored in a hierarchical folder layout, such as:

    root/train/cat/image1.jpg
    root/train/dog/image2.jpg
    root/test/bird/image3.jpg

The class supports automatic labeling based on directory names, integration
with DeepTrack data pipelines, and flexible splitting of datasets by folder.

Key Features
------------
- **Attribute Access**

    Provides access to common attributes such as image paths, label indices,
    and category names. Each entry is returned as a `SourceItem` with fields
    `path`, `label`, and `label_name`.

- **Automatic Labeling**

    Converts directory names into integer labels, supporting direct use in
    training pipelines or models that expect categorical inputs.

- **Flexible Dataset Splitting**

    Supports splitting datasets based on the top-level folder structure.
    This enables separating data into training, validation, and test sets
    using directory naming conventions.

Module Structure
----------------
Classes:

- `ImageFolder`: Source of image paths and labels from a structured folder.

    Wraps a directory of image files into a DeepTrack `Source`, supporting
    standard methods such as iteration, indexing, and filtering.

Attributes:

- `known_extensions: list[str]`

    List of recognized file extensions used when scanning directories for
    valid image files: `["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]`

Examples
--------
TODO

"""

from __future__ import annotations

import glob
import os

from deeptrack.sources.base import Source, SourceItem


__all__ = [
    "ImageFolder",
    "known_extensions",
]


known_extensions = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]


class ImageFolder(Source):
    """Data source for images organized in a directory structure.

    `ImageFolder` scans a directory tree where images are stored under
    subdirectories that represent their categorical labels. It automatically
    assigns integer labels, stores paths and names, and supports operations
    such as splitting by folder and category lookup.

    It behaves like a standard `Source`, returning `SourceItem` objects that
    include image file paths, label indices, and label names. This allows
    seamless integration with feature pipelines in DeepTrack2.

    Parameters
    ----------
    root: str
        Path to the root directory that contains subfolders of images. The
        first-level subfolder names are interpreted as categories.

    Attributes
    ----------
    _category_to_int: dict[str, int]
        Mapping from category name to numeric label.
    _int_to_category: dict[int, str]
        Mapping from numeric label to category name.
    _paths: list[str]
        Internal list of all file paths.
    _length: int
        Total number of images discovered.
    _root: str
        Root directory provided by the user.

    Methods
    -------
    __len__() -> int
        Return the number of image files found.
    classes -> list[str]
        Return a list of unique class names found in the directory.
    get_category_name(path: str, directory_level: int) -> str
        Return the category name for a given image path.
    label_to_name(label: int) -> str
        Convert an integer label back to its string category name.
    name_to_label(name: str) -> int
        Convert a string category name to its integer label.
    split(*splits: str) -> tuple[ImageFolder, ...]
        Return one or more subsets of the data based on top-level folder names.

    Examples
    --------
    **Create a dummy dataset structure with train/test subfolders**

import os
import shutil

# Temporary root directory
root = "tmp_data"

# Remove existing directory if needed
if os.path.exists(root):
    shutil.rmtree(root)

# Define splits and classes
splits = ["train", "test"]
classes = ["cat", "dog", "bird"]

# Create directories and dummy files
for split in splits:
    for cls in classes:
        folder_path = os.path.join(root, split, cls)
        os.makedirs(folder_path)
        for i in range(2):
            file_path = os.path.join(folder_path, f"image_{i}.jpg")
            with open(file_path, "w") as f:
                f.write("dummy")

    **Load a split of the dataset**

from deeptrack.sources.folder import ImageFolder

# Load the training set
train_data = ImageFolder(os.path.join(root, "train"))

print(len(train_data))
print(train_data.classes)
print(train_data.path())

    **Access a source item**

item = train_data[0]
print(item["path"])
print(item["label"])
print(item["label_name"])

    **Convert between label names and indices**

train_data.name_to_label("cat")
train_data.label_to_name(0)

    **Split the dataset across top-level folders**

all_data = ImageFolder(root)
train, test = all_data.split("train", "test")

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

    **Print paths in each split**

print("Train files:")
for item in train:
    print(item["path"])

print("Test files:")
for item in test:
    print(item["path"])

    """

    _root: str
    _paths: list[str]
    _length: int
    _category_to_int: dict[str, int]
    _int_to_category: dict[int, str]

    @property
    def classes(
        self: ImageFolder,
    ) -> list[str]:
        """List of category names in the dataset.

        Returns
        -------
        list[str]
            A list of unique category names corresponding to the top-level
            directories found under the root folder.

        """

        return list(self._category_to_int.keys())

    def __init__(
        self: ImageFolder,
        root: str,
    ):
        """Initialize an `ImageFolder` from a directory structure.

        This constructor scans a given root directory recursively for image
        files, assigns labels based on their immediate subfolder names, and
        initializes the `Source` base class with the image paths and
        associated metadata.

        The directory structure is expected to follow the format:

            root/category_name/image_001.png
            root/category_name/image_002.png
            ...

        All recognized files must have an extension in `known_extensions`.

        Parameters
        ----------
        root: str
            Path to the root directory containing the categorized images.

        Raises
        ------
        ValueError
            If no valid image files are found or directory is malformed.

        """

        # Store the root directory path
        self._root = root

        # Recursively collect all file paths under root
        self._paths = glob.glob(f"{root}/**/*", recursive=True)

        # Filter for valid image files using known extensions
        self._paths = [
            path for path in self._paths
            if os.path.isfile(path) and path.split(".")[-1] in known_extensions
        ]
        # Ensure consistent order across runs
        self._paths.sort()
        # Store total number of valid image paths
        self._length = len(self._paths)

        # Extract category name from path (1 level down from root)
        category_per_path = [
            self.get_category_name(path, 0) for path in self._paths
        ]
        # Compute the set of unique category names
        unique_categories = set(category_per_path)

        # Create mapping: category name -> integer label
        self._category_to_int = {
            category: i for i, category in enumerate(unique_categories)
        }
        # Create inverse mapping: label index -> category name
        self._int_to_category = {
            i: category for category, i in self._category_to_int.items()
        }

        # Map each image path to its integer label
        categories = [
            self._category_to_int[category] for category in category_per_path
        ]

        # Initialize the base Source with path, label index, and label name
        super().__init__(
            path=self._paths,
            label=categories,
            label_name=category_per_path,
        )

    def __len__(
        self: ImageFolder,
    ) -> int:
        """Return the total number of images in the dataset.

        Returns
        -------
        int
            The number of image paths found in the directory structure.

        """

        return self._length

    def get_category_name(
        self: ImageFolder,
        path: str,
        directory_level: int,
    ) -> str:
        """Extract the category name from file path at given directory level.

        This method determines the category name (i.e., the name of the
        directory at the specified `directory_level` relative to the root)
        associated with the given file path.

        Parameters
        ----------
        path: str
            The absolute path to the image file.
        directory_level: int
            The index of the directory component to extract, relative to the
            root.

        Returns
        -------
        str
            The name of the folder at the given level in the path.

        """

        relative_path = path.replace(self._root, "", 1).lstrip(os.sep)
        folder = (
            relative_path.split(os.sep)[directory_level]
            if relative_path
            else ""
        )
        return folder

    def label_to_name(
        self: ImageFolder,
        label: int,
    ) -> str:
        """Convert an integer label to its corresponding category name.

        Given a numeric label (e.g., 0, 1, 2), return the associated category
        name (e.g., "cat", "dog") that was assigned during initialization.

        Parameters
        ----------
        label: int
            The integer label representing a category.

        Returns
        -------
        str
            The name of the category corresponding to the label.

        """

        return self._int_to_category[label]

    def name_to_label(
        self: ImageFolder,
        name: str,
    ) -> int:
        """Convert a category name to its corresponding integer label.

        Given a category name (e.g., "cat", "dog"), return the integer label
        (e.g., 0, 1) assigned to it during initialization.

        Parameters
        ----------
        name: str
            The name of the category.

        Returns
        -------
        int
            The integer label corresponding to the category name.

        """

        return self._category_to_int[name]

    def split(
        self: ImageFolder,
        *splits: str,
    ) -> tuple[str]:
        """Split the dataset into subsets by folder name.

        This method splits the dataset into subsets based on the first folder
        name in the path of each image. It is useful when datasets are stored
        in separate directories (e.g., `train`, `test`, `val`), and you want to
        retrieve subsets accordingly.

        If no arguments are given, it returns one `ImageFolder` per top-level
        directory found under the root. If specific names are provided, only
        those subsets are returned.

        Parameters
        ----------
        *splits: str
            Names of the subfolders (relative to the root) to split into.

        Returns
        -------
        tuple[ImageFolder, ...]
            A tuple of `ImageFolder` instances, one per requested split.

        Raises
        ------
        ValueError
            If an unknown split name is provided or no categories are found.

        """

        # Get top-level folder names present in image paths
        all_splits = set([self.get_category_name(path, 0)
                          for path in self._paths])

        # If no specific splits provided, return all available
        if len(splits) == 0:

            if len(all_splits) == 0:
                raise ValueError("No categories to split into")
            return self.split(*all_splits)

        # Validate requested splits
        if not all(split in all_splits for split in splits):
            raise ValueError(
                f"Unknown split. Available splits are {all_splits}"
                )

        output = []

        def update_root_source(
            item: SourceItem,
        ) -> None:
            """Inner function which updates attributes of root source."""
            for key in item:
                getattr(self, key).invalidate()
                getattr(self, key).set_value(item[key])

        for split in splits:
            # Create ImageFolder pointing to subdirectory
            subfolder = ImageFolder(os.path.join(self._root, split))
            # Attach update callback to propagate selected item to parent
            subfolder.on_activate(update_root_source)
            output.append(subfolder)

        return tuple(output)
