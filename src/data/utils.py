import json
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as T


def get_vertices(segmentation: np.ndarray) -> List[dict]:
    """The function computes vertices from segmentation masks

    The functions aims to, given a segmentation mask linked to a
    given prediction, compute the vertices that outline the prediction shape.


    Args:
        segmentation (np.ndarray): Binary segmentation mask

    Returns:
        List[dict]: Segmentation map (x, y) vertices
    """
    vertices, _ = cv2.findContours(
        segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    vertices = np.concatenate(vertices, axis=0)
    result = []
    for this_vertice in vertices:
        this_vertice = np.squeeze(this_vertice)
        result.append({"x": this_vertice[0], "y": this_vertice[1]})

    return result


def parse_json_annotations(annotations_path: str) -> pd.DataFrame:
    """The functions parses the json annotation to a dataframe format

    Each row in the dataframe is link to a given annotation in a given images.
    Each annotation is characterized by the following features:
        1.Annotation_id
        2.Image_id
        3. Image_filename
        4. Boubding box coordinates
        5. Segmentation vertices that outline the polygon containing the object
        6. Bouding Box area

    Args:
        annotations_path (str): Annotations json path

    Returns:
        pd.DataFrame: Image target annotations
    """
    annotations = json.load(open(annotations_path, "r"))
    image_annotations = annotations["images"]
    target_annotations = annotations["annotations"]

    processed_annotations = pd.DataFrame(
        index=list(range(1 + len(target_annotations))),
        columns=["image_id", "image_filename", "segmentation"],
    )

    for this_annotation in target_annotations:
        this_annotation_image_id = this_annotation["image_id"]
        this_image_filename = list(
            filter(
                lambda image: image["id"] == this_annotation_image_id, image_annotations
            )
        )[0]["file_name"]
        this_annotation_id = this_annotation["id"]
        this_annotation_segmentation = this_annotation["segmentation"][0]

        processed_annotations.loc[this_annotation_id] = [
            this_annotation_image_id,
            this_image_filename,
            this_annotation_segmentation,
        ]

    return processed_annotations


def generate_segmentation_masks(
    image_name: str, image: Image, annotations: pd.DataFrame, output_path: str
) -> None:
    """The function aims to generate segmentation masks as jpg files

    Args:
        images (Image): Image whose mask is going to be generated
        annotations (pd.DataFrame): Image annotations data
        output_path (str): Path to where segmentations will be saved

    Please note that the generated masks are automatically saved to the output_path
    """
    this_image_annotations = annotations.loc[
        annotations["image_filename"] == image_name
    ].index.to_list()
    this_image_mask = np.zeros((image.shape[0], image.shape[1], 1))
    this_image_annotations = annotations.loc[this_image_annotations]
    for this_idx, this_annotation in this_image_annotations.iterrows():
        this_annotation_segmentation = this_annotation["segmentation"]
        this_annotation_x_points = this_annotation_segmentation[0::2]
        this_annotation_y_points = this_annotation_segmentation[1::2]
        vertices = np.array(
            [
                [x, y]
                for (x, y) in zip(this_annotation_x_points, this_annotation_y_points)
            ],
            np.int32,
        )
        this_image_mask = cv2.fillPoly(
            this_image_mask,
            pts=[vertices],
            color=(this_idx + 1,),
        )
    cv2.imwrite(output_path, this_image_mask)


def get_transformations(train: bool) -> T.Compose:
    """The functions returns a Compose object holding data augmentations approaches

    Args:
        train (bool): Flag to sinalize train images

    Returns:
        T.Compose: Data Augmentations to be applied while training
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def collate_fn(batch) -> Tuple[torch.tensor]:
    """This functions aims to stack inputs and targets for training/validation

    The functions stacks all input images as one multidimensional tensor,
    and the same happens with the target annotations.

    Args:
        batch : Batch of data for training/validation

    Returns:
        Tuple[torch.tensor]: Batch of stacked inputs and stacked targets
    """
    return tuple(zip(*batch))
