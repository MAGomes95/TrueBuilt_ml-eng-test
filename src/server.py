import json

import numpy as np
import torch
from flask import Flask, request
from torchvision.io import read_image

from data.utils import get_vertices

server = Flask(__name__)


@server.route("/run-inference", methods=["POST"])
def generate_predictions(mask_confidence: float = 0.8, device: str = "cuda") -> json:
    """Prediction generation endpoint

    This endpoints aims to generate the predictions for a given image,
    passed in the URL, and format the predictions so that they are returned
    in the expected format.

    Args:
        mask_confidence (float): Probability confidence value aims to extract each prediction binary mask. Default 0.8
        device (str): Device in which the prediction will be generated.

        Please note that the device for training and this one must be the same.

    Returns:
        json: Response body containing predictions
    """
    model_path = "models/roomDetector.pth"

    to_detect = request.args.get("type")
    image_path = request.args.get("image")

    model = torch.load(model_path)
    model.eval()

    input_image = read_image(image_path)
    input_image = input_image[None, :, :, :]
    input_image = (input_image - input_image.min()) / (
        input_image.max() - input_image.min()
    )
    model = model.to(device)
    input_image = input_image.to(device)
    predictions = model(input_image)[0]

    response_body = {"type": to_detect, "imageId": 0}

    for this_idx, (this_mask, this_score) in enumerate(
        zip(predictions["masks"], predictions["scores"])
    ):
        this_mask = this_mask.cpu().detach().numpy()
        this_mask = (this_mask >= mask_confidence).astype(np.uint8)
        this_mask = np.swapaxes(this_mask, 0, -1)
        this_confidence = this_score.cpu().detach().numpy()
        this_prediction = {
            "roomId": this_idx,
            "vertices": get_vertices(this_mask),
            "confidence": float(this_confidence),
        }
        response_body["detectionResults"]["rooms"].append(this_prediction)

    return response_body


if __name__ == "__main__":
    server.run(debug=True)
