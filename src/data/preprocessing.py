from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from PIL.Image import Image as Image_type


def cast_pdf_to_image(pdf_path: str) -> Image_type:
    """The function casts the input pdf to a jpg image

    Args:
        image_input_path (str): Path to the pdf to be converted to images

    Returns:
        Image: Correspondent jpg image
    """
    image = convert_from_path(pdf_path=pdf_path)
    return image[0]


def resize(
    image_name: str,
    image: Image,
    annotations: pd.DataFrame,
    size: tuple = (9600, 7200),
) -> Tuple[Image_type, pd.DataFrame]:
    """The function aims to resize a given image to a given size and update respected annotation

    Args:
        image_name (str): Image (that is to be resized) name
        image (Image): Image to be resized
        annotation (pd.Series): Image annotations
        size (tuple, optional): Final size. Defaults to (7200, 9600).

    Returns:
        (Image, pd.DataFrame): (Resized Image, Calibrated annotations)
    """
    if image.size == size:
        return image, annotations

    image_annotation_idx = annotations.loc[
        annotations["image_filename"] == image_name
    ].index.to_list()
    image_annotations = annotations.loc[image_annotation_idx]
    for this_idx, this_annotation in image_annotations.iterrows():
        segmentation_x_points = [
            float(this_annotation["segmentation"][i])
            for i in range(0, len(this_annotation["segmentation"]), 2)
        ]
        segmentation_y_points = [
            float(this_annotation["segmentation"][i])
            for i in range(1, len(this_annotation["segmentation"]), 2)
        ]

        width_scaling = size[1] / image.width
        height_scaling = size[0] / image.height

        scaled_x_points = list(map(lambda x: x * width_scaling, segmentation_x_points))
        scaled_y_points = list(map(lambda y: y * height_scaling, segmentation_y_points))

        segmentations = [None] * (len(scaled_x_points) + len(scaled_y_points))
        segmentations[0::2] = scaled_x_points
        segmentations[1::2] = scaled_y_points

        image_annotations.loc[this_idx, "segmentation"] = segmentations

    annotations.loc[image_annotation_idx] = image_annotations
    resized_image = cv2.resize(np.array(image), size, cv2.INTER_CUBIC)
    return resized_image, annotations
