ALL_TESTS = [
    {
        "name": "resnet18_bs32_lr1e-3_40epochs",
        "model": "resnet18",
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 40,
        "freeze_backbone": True,
        "augmentation": True
    }
]