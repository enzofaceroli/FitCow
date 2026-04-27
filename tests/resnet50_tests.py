RESNET50_TESTS = [
    # {
    #     "name": "resnet50_bs16_lr1e-3_15epochs",
    #     "model": "resnet50",
    #     "batch_size": 16,
    #     "learning_rate": 1e-3,
    #     "epochs": 15,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },
    
    # {
    #     "name": "resnet50_bs16_lr1e-4_15epochs",
    #     "model": "resnet50",
    #     "batch_size": 16,
    #     "learning_rate": 1e-4,
    #     "epochs": 15,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },
    
    {
        "name": "resnet50_bs16_lr1e-3_30epochs",
        "model": "resnet50",
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 30,
        "freeze_backbone": True,
        "augmentation": True
    },
    
    {
        "name": "resnet50_bs16_lr1e-4_30epochs",
        "model": "resnet50",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 30,
        "freeze_backbone": True,
        "augmentation": True
    },
    
    {
        "name": "resnet50_bs32_lr1e-4_20epochs",
        "model": "resnet50",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 20,
        "freeze_backbone": True,
        "augmentation": True
    },    
]