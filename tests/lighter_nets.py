LIGHT_TESTS = [
    {
        "name": "vgg16_bs16_lr1e-4_30epochs_NOWEIGHTS",
        "model": "vgg16",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 30,
        "freeze_backbone": True,
        "augmentation": True
    }
]