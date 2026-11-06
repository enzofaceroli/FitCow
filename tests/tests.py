TESTS = [
    {
        "model": "convnext_tiny",
        "batch_size": 4,
        "learning_rate": 3e-5,
        "epochs": 1,
        "freeze_backbone": True,
        "augmentation": True,
        "method": "coral"
    },
    {
        "model": "convnext_tiny",
        "batch_size": 4,
        "learning_rate": 3e-5,
        "epochs": 1,
        "freeze_backbone": True,
        "augmentation": True,
        "method": "corn"
    },
    {
        "model": "convnext_tiny",
        "batch_size": 4,
        "learning_rate": 3e-5,
        "epochs": 1,
        "freeze_backbone": True,
        "augmentation": True,
        "method": None
    }
]