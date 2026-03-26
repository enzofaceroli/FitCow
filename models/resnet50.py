import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet50(num_classes = 10, freeze_backbone = True):
    model = resnet50(weights =  ResNet50_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model