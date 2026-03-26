import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def build_densenet121(num_classes = 10, freeze_backbone = True):
    model = densenet121(weights = DenseNet121_Weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model