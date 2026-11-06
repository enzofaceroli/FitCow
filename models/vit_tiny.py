import timm

def build_vit_tiny(num_classes, freeze_backbone=False):

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

    return model