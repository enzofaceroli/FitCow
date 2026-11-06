import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from coral_pytorch.dataset import (
    proba_to_label,
    levels_from_labelbatch,
    corn_label_from_logits
)
from coral_pytorch.losses import coral_loss, corn_loss

def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    method=None,
    num_classes=5,
    mae_step=0.5
):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0
    total = 0

    test_bar = tqdm(test_loader, desc="Evaluating", leave=False)

    all_probas = []

    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            if method == "coral":
                levels = levels_from_labelbatch(
                    labels,
                    num_classes=num_classes
                ).to(device)

                loss = coral_loss(outputs, levels)

                probas = torch.sigmoid(outputs)
                all_probas.extend(probas.cpu().numpy())

                preds = proba_to_label(probas)

            elif method == "corn":
                loss = corn_loss(
                    outputs,
                    labels,
                    num_classes=num_classes
                )

                preds = corn_label_from_logits(outputs)

            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if method == "coral":
        all_probas = np.array(all_probas)
        print("Mean probabilities:", all_probas.mean(axis=0))
        print("Positive predictions:", (all_probas >= 0.5).mean(axis=0))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean() * 100.0
    mae = np.abs(all_preds - all_labels).mean() * mae_step

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(num_classes)
    )

    avg_loss = total_loss / total

    return acc, mae, cm, avg_loss