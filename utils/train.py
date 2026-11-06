import torch
from tqdm import tqdm
import torch.nn as nn
from coral_pytorch.dataset import (
    levels_from_labelbatch,
    proba_to_label,
    corn_label_from_logits
)
from coral_pytorch.losses import coral_loss, corn_loss

from utils.evaluate import evaluate_model


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    method=None,
    num_classes=5,
    epochs=10,
    tensor_weights=None,
    logger=None
):
    model.to(device)

    acc_history = []
    mae_history = []
    loss_history = []

    eval_acc_history = []
    eval_mae_history = []
    eval_loss_history = []

    if method not in (None, "coral", "corn"):
        raise ValueError(f"Invalid method: {method}")
    
    if method is None:
        criterion = nn.CrossEntropyLoss(weight=tensor_weights)

    for epoch in range(epochs):
        print(f"Starting Epoch: {epoch + 1}")

        model.train()

        running_loss = 0.0
        running_mae = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)

            if method == "coral":
                levels = levels_from_labelbatch(
                    labels,
                    num_classes=num_classes
                ).to(device)

                loss = coral_loss(outputs, levels)
                predicted = proba_to_label(torch.sigmoid(outputs))

            elif method == "corn":
                loss = corn_loss(outputs, labels, num_classes=num_classes)
                predicted = corn_label_from_logits(outputs)

            elif method is None:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            running_mae += torch.abs(predicted - labels).sum().item()

            correct += torch.sum(predicted == labels).item()
            total += batch_size

        epoch_loss = running_loss / total
        acc = 100.0 * correct / total
        mae = (running_mae / total) * 0.5

        acc_history.append(acc)
        mae_history.append(mae)
        loss_history.append(epoch_loss)

        eval_acc, eval_mae, cm, avg_eval_loss = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            method=method,
            num_classes=num_classes
        )

        eval_acc_history.append(eval_acc)
        eval_mae_history.append(eval_mae)
        eval_loss_history.append(avg_eval_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Loss: {epoch_loss:.4f} - "
            f"Acc: {acc:.2f}% - "
            f"MAE: {mae:.4f}"
        )

        print(
            f"Eval Acc: {eval_acc:.2f}% - "
            f"Eval MAE: {eval_mae:.4f} - "
            f"Loss: {avg_eval_loss:.4f}"
        )

    return (
        acc_history,
        mae_history,
        loss_history,
        eval_acc_history,
        eval_mae_history,
        eval_loss_history
    )