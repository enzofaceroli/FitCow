import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes=5, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model