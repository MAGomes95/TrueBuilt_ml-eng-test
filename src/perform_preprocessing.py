import os

import cv2

from data.preprocessing import cast_pdf_to_image, resize
from data.utils import generate_segmentation_masks, parse_json_annotations

unpreprocessed_images_path = "../datasets/Rooms"
processed_images_path = "../datasets/silver/images/"
segmentation_masks_path = "../datasets/silver/masks/"

os.makedirs(processed_images_path, exist_ok=True)
os.makedirs(segmentation_masks_path, exist_ok=True)

annotations = parse_json_annotations(annotations_path="../datasets/annotations.json")

for this_image in os.listdir(unpreprocessed_images_path):
    processed_image_name = this_image[:-4]
    image = cast_pdf_to_image(
        pdf_path=os.path.join(unpreprocessed_images_path, this_image)
    )
    resized_image, annotations = resize(
        f"{processed_image_name}.jpg", image, annotations, size=(950, 950)
    )
    cv2.imwrite(
        os.path.join(processed_images_path, f"{processed_image_name}.png"),
        resized_image,
    )

    generate_segmentation_masks(
        f"{processed_image_name}.jpg",
        resized_image,
        annotations,
        os.path.join(segmentation_masks_path, f"{processed_image_name}_mask.png"),
    )
