from typing import Union

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(
    n_classes: int = 2,
    mask_layer_dimension: int = 64,
    to_freeze: Union[int, float] = 20,
) -> torchvision.models:
    """Creates Mask R-CNN instance segmentation models with Fast RCNN for bounding box prediction.

    Args:
        n_classes (int, optional): Number of classes. Defaults to 2 (Rooms + Background)
        mask_layer_dimension (int, optional): Dimension of mask prediction layer. Defaults to 64.
        to_freeze (Union[int, float], optional): Freezing Controler. Defaults to 20.

        Please note that the to_freeze aims to freeze the optimization of a given set of layers.
        The parameter if "int" is interpreted as being the number of initial layers to freeze.
        If the parameter is "float" is interpreted the percentage of initial layers to freeze.

    Returns:
       torchvision.models: Mask RCNN Model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, mask_layer_dimension, n_classes
    )

    if isinstance(to_freeze, float):
        to_freeze = int(to_freeze * 84)  # 84 is total number of layers in the model

    for this_layer_id, this_layer in enumerate(model.parameters()):
        if this_layer_id <= to_freeze:
            this_layer.requires_grad = False

    return model
