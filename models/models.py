
import torch
import torch.nn as nn

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
    vgg16,
    VGG16_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)

from coral_pytorch.layers import CoralLayer

def get_output_layer(in_features, num_classes, method=None):
    if method is None:
        return nn.Linear(in_features, num_classes)

    elif method == "coral":
        return CoralLayer(in_features, num_classes)

    elif method == "corn":
        return nn.Linear(in_features, num_classes - 1)

    else:
        raise ValueError(
            f"Invalid method: {method}"
        )

def build_densenet121(num_classes=5, freeze_backbone=True, method=None):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = get_output_layer(
        in_features, num_classes, method
    )

    return model

def build_dinov2(num_classes=5, freeze_backbone=True, method=None):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.embed_dim
    model.head = get_output_layer(
        in_features, num_classes, method
    )

    return model

def build_resnet18(num_classes=5, freeze_backbone=True, method=None):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = get_output_layer(
        in_features, num_classes, method
    )

    return model

def build_resnet50(num_classes=5, freeze_backbone=True, method=None):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = get_output_layer(
        in_features, num_classes, method
    )

    return model

def build_vgg16(num_classes=5, freeze_backbone=True, method=None):
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = get_output_layer(
        in_features, num_classes, method
    )

    return model

def build_convnext_tiny(num_classes=5, freeze_backbone=True, method=None):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = get_output_layer(
        in_features, num_classes, method
    )

    return model
