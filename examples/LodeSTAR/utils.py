import os
import json
import skimage
import numpy as np
import pycocotools.mask as mask_util
import skimage.io
import skimage.morphology
import tqdm


def convert_dataset_to_coco(
    image_paths,
    mask_paths,
    output_dir,
    dataset_name,
    crop_slice=(slice(None), slice(None)),
):
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
    for i, (image_path, mask_path) in tqdm.tqdm(
        enumerate(zip(image_paths, mask_paths)), total=len(image_paths)
    ):
        image_tmp = skimage.io.imread(image_path)
        mask_tmp = skimage.io.imread(mask_path)

        image = np.zeros((1010, 1010), dtype=np.uint8) + np.mean(image_tmp, dtype=int)
        image[crop_slice] = image_tmp[crop_slice]

        mask = np.zeros((1010, 1010), dtype=np.uint8)
        mask[crop_slice] = mask_tmp[crop_slice]

        # change extension to .png
        image_path = os.path.splitext(image_path)[0] + ".png"

        new_path = os.path.join(output_dir, os.path.basename(image_path))
        # resave image
        skimage.io.imsave(new_path, image)

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
            w = w - x
            h = h - y

            rle = mask_to_coco_segmentation(mask == prop.label)

            annotations.append(
                {
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [y, x, h, w],
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
    rle["counts"] = rle["counts"].decode("ascii")

    return rle