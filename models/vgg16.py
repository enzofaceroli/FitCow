import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def build_vgg16_transfer(num_classes = 10, freeze_backbone = True):
    model = vgg16(weights = VGG16_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model