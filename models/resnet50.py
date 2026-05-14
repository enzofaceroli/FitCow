import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet50(num_classes = 5, freeze_backbone = True):
    model = resnet50(weights =  "IMAGENET1K_V1")
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model

# def build_resnet50(num_classes=5, freeze_backbone=True):
#     model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
#     for param in model.parameters():
#         param.requires_grad = False

#     for param in model.layer4.parameters():
#         param.requires_grad = True

#     in_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(in_features, num_classes)
#     )

#     for param in model.fc.parameters():
#         param.requires_grad = True

#     return model