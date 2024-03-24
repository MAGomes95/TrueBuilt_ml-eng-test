import torch


def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    epoch,
    device="cuda",
) -> float:
    model.to(device)
    model.train()
    for inputs, targets in train_dataloader:
        inputs = list(input.to(device) for input in inputs)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        losses = model(inputs, targets)
        total_loss = sum(losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}, Total Train Loss: {total_loss}")
    return total_loss


def evaluate(model, validation_dataloader, epoch, device="cuda") -> float:
    model.to(device)
    for inputs, targets in validation_dataloader:
        inputs = list(input.to(device) for input in inputs)
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        with torch.no_grad():
            losses = model(inputs, targets)
        total_loss = sum(losses.values())
    print(f"Epoch {epoch}, Total Evaluation Loss: {total_loss}")
    return total_loss
