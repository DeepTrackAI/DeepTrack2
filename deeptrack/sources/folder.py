import glob
import os 
from deeptrack.sources.base import Source

known_extensions = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]

class ImageFolder(Source):
    """Data source for images organized in a directory structure.

    This class assumes that the images are organized in a
    directory structure where:

    ```bash
        root/dog/xxx.png
        root/dog/xxy.png
        root/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/[...]/asd932_.png
    ```

    The first level of directories (e.g., `dog`, `cat`) is used as labels
    for the images, and the images are expected to have file extensions 
    included in `known_extensions`.

    Parameters
    ----------
    path : list
        List of paths to the image files.
    label : list
        List of corresponding labels for each image.
    label_name : list
        List of category names corresponding to each label.

    Methods
    -------
    classes : list
        Returns a list of unique class names (category names).
    __init__(root: str)
        Initializes the `ImageFolder` instance by scanning
        the directory structure.
    __len__()
        Returns the total number of images in the dataset.
    get_category_name(path: str, directory_level: int)
        Retrieves the category name (directory name) for the given image path
        at a specific directory level.
    label_to_name(label: int)
        Converts a label index to the corresponding category name.
    name_to_label(name: str)
        Converts a category name to the corresponding label index.
    split(*splits: str)
        Splits the dataset into subsets based on the folder structure.
        The first folder name in the path will be used to define the split.
    """

    path: str
    label: int
    label_name: str

    @property
    def classes(self) -> list:
        return list(self._category_to_int.keys())

    def __init__(self, root):
        self._root = root

        self._paths = glob.glob(f"{root}/**/*", recursive=True)
        self._paths = [
            path for path in self._paths if os.path.isfile(path) 
            and path.split(".")[-1] in known_extensions
            ]
        self._paths.sort()
        self._length = len(self._paths)

        # get category name as 1 directory down from root
        category_per_path = [self.get_category_name(path, 0) 
                             for path in self._paths]
        unique_categories = set(category_per_path)

        # create a dictionary mapping category name to integer
        self._category_to_int = {category: i for i, category
                                  in enumerate(unique_categories)}
        self._int_to_category = {i: category for category, i 
                                 in self._category_to_int.items()}

        # create a list of integers corresponding to the category of each path
        categories = [self._category_to_int[category] 
                      for category in category_per_path]

        super().__init__(path=self._paths,
                         label=categories,
                         label_name=category_per_path
                        )

    def __len__(self):
        return self._length

    def get_category_name(self, path, directory_level):
        relative_path = path.replace(self._root, "", 1).lstrip(os.sep)
        folder = relative_path.split(os.sep)[directory_level] \
            if relative_path else ""
        return folder
    
    def label_to_name(self, label):
        return self._int_to_category[label]
    
    def name_to_label(self, name):
        return self._category_to_int[name]
    
    def split(self, *splits: str):
        """Split the dataset into subsets.
        
        The splits are defined by the names of the first folder
        in the path of each image. For example, if the dataset
        contains images in the following structure:
        
        ```bash
        root/A/dog/xxx.png
        root/A/dog/xxy.png
        root/A/[...]/xxz.png

        root/B/cat/123.png
        root/B/cat/nsdf3.png
        root/B/[...]/asd932_.png
        ```
        
        Then the dataset can be split into two subsets, one containing
        all images in the `A` folder and one containing all images 
        in the `B` folder.

        Parameters
        ----------

        splits : str
            The names of the categories to split into.
        """
        
        all_splits = set([self.get_category_name(path, 0) 
                          for path in self._paths])

        if len(splits) == 0:
            
            if len(all_splits) == 0:
                raise ValueError("No categories to split into")
            return self.split(*all_splits)

        if not all(split in all_splits for split in splits):
            raise ValueError(
                f"Unknown split. Available splits are {all_splits}"
                )

        output = []

        def update_root_source(item):
            for key in item:
                getattr(self, key).invalidate()
                getattr(self, key).set_value(item[key])
    

        for split in splits:
            subfolder = ImageFolder(os.path.join(self._root, split))
            subfolder.on_activate(update_root_source)
            output.append(subfolder)

        return tuple(output)

