import torch

from data.generators import get_dataloader
from models.train import evaluate, train_one_epoch
from models.utils import get_instance_segmentation_model

device = torch.device("cuda")
n_epochs = 50
batch_size = 1
learning_rate = 0.0003
momentum = 0.9
weight_decay = 0.0005

train_dataloader = get_dataloader(
    is_train=True,
    indexes=[0, 1, 3, 4, 5, 8],
    data_root="../datasets/silver",
    batchsize=batch_size,
)
validation_dataloader = get_dataloader(
    is_train=False,
    indexes=[2, 6, 7],
    data_root="../datasets/silver/",
    batchsize=batch_size,
)

model = get_instance_segmentation_model(mask_layer_dimension=64, to_freeze=0.7)
model = model.to(device)

parameters = [param for param in model.parameters() if param.requires_grad]

optimizer = torch.optim.SGD(
    params=parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum
)

best_validation_loss = None
for this_epoch in range(n_epochs):
    torch.cuda.empty_cache()
    this_epoch_train_loss = train_one_epoch(
        model, train_dataloader, optimizer, this_epoch, device
    )
    this_epoch_validation_loss = evaluate(
        model, validation_dataloader, this_epoch, device
    )
    if (best_validation_loss is None) or (
        best_validation_loss > this_epoch_validation_loss
    ):
        best_validation_loss = this_epoch_validation_loss
        torch.save(model, "../models/roomDetector.pth")
