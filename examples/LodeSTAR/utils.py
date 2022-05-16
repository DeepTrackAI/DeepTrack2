import os
import json
import skimage
import numpy as np
import pycocotools.mask as mask_util
import skimage.io
import skimage.morphology


def convert_dataset_to_coco(image_paths, mask_paths, output_dir, dataset_name):
    """Convert dataset to COCO format.

    Args:
        image_paths (str): Paths to images.
        mask_paths (str): Paths to masks.
        output_dir (str): Directory of output.
        dataset_name (str): Name of dataset.
    """

    categories = [{"id": 1, "name": "object"}]
    annotations = []
    images = []
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        image = skimage.io.imread(image_path)
        mask = skimage.io.imread(mask_path)
        height, width = image.shape[:2]

        images.append(
            {
                "file_name": os.path.basename(image_path),
                "height": height,
                "width": width,
                "id": i,
            }
        )

        mask = np.squeeze(mask)
        props = skimage.measure.regionprops(mask)
        for prop in props:

            bbox = prop.bbox
            bbox = list(bbox)
            x, y, w, h = bbox

            rle = mask_to_coco_segmentation(mask == prop.label)

            annotations.append(
                {
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "segmentation": rle,
                    "area": w * h,
                    "iscrowd": 0,
                    "id": len(annotations),
                }
            )

    output_file = os.path.join(output_dir, dataset_name + ".json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "images": images,
                "categories": categories,
                "annotations": annotations,
            },
            f,
            indent=2,
        )


def mask_to_coco_segmentation(mask):
    """Convert mask to COCO segmentation.

    Args:
        mask (ndarray): Mask.

    Returns:
        list: COCO segmentation.
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle