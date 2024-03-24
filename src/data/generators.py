import os

import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F

from data.utils import collate_fn, get_transformations


class RoomsDataset(torch.utils.data.Dataset):
    """Custom Pytorch datataset for training and evaluation"""

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.images = sorted(os.listdir(os.path.join(root, "images")))
        self.masks = sorted(os.listdir(os.path.join(root, "masks")))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, "images", self.images[idx])
        target_path = os.path.join(self.root, "masks", self.masks[idx])
        image = read_image(image_path)
        mask = read_image(target_path)

        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)
        areas = (boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3])

        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        inputs = tv_tensors.Image(image)

        targets = {}
        targets["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(inputs)
        )
        targets["masks"] = tv_tensors.Mask(masks)
        targets["labels"] = labels
        targets["image_id"] = torch.tensor([idx])
        targets["area"] = areas
        targets["iscrowd"] = iscrowd

        if self.transforms is not None:
            inputs, targets = self.transforms(inputs, targets)

        return inputs, targets

    def __len__(self):
        return len(self.images)


def get_dataloader(is_train: bool, indexes: list[int], data_root: str, batchsize: int):
    """Retrieves data loader for training or validation purposes

    Please note that given the small dimension of the Rooms dataset,
    a special selection of images was made for both training and validation.
    This explains the need for the indexes' function parameter.

    Args:
        indexes (list[int]): Index of images to be considered in this loader
        is_train (bool, optional): Train flag for augmentation purposes.
    """
    dataset = RoomsDataset(
        root=data_root, transforms=get_transformations(train=is_train)
    )
    dataset = torch.utils.data.Subset(dataset, indexes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return data_loader
