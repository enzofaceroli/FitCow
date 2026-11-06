import torch
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss

from models.models import build_resnet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_resnet18(num_classes=5).to(device)

images = torch.randn(4, 3, 224, 224).to(device)
labels = torch.tensor([0, 1, 3, 4]).to(device)

outputs = model(images)

levels = levels_from_labelbatch(
    labels,
    num_classes=5
).to(device)

loss = coral_loss(outputs, levels)

probas = torch.sigmoid(outputs)
predictions = proba_to_label(probas)

print("Output shape:", outputs.shape)
print("Levels shape:", levels.shape)
print("Loss:", loss.item())
print("Predictions:", predictions)
print("Labels:", labels)

assert outputs.shape == (4, 4)
assert levels.shape == (4, 4)
assert torch.isfinite(loss)
assert predictions.shape == labels.shape

print("CORAL test passed!")