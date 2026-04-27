DENSENET121_TESTS = [
    {
        "name": "densenet121_bs16_lr1e-3_20epochs",
        "model": "densenet121",
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 20,
        "freeze_backbone": True,
        "augmentation": True
    },
    
    {
        "name": "densenet121_bs16_lr1e-4_20epochs",
        "model": "densenet121",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 20,
        "freeze_backbone": True,
        "augmentation": True
    },
    
    # {
    #     "name": "densenet121_bs32_lr1e-3_20epochs",
    #     "model": "densenet121",
    #     "batch_size": 32,
    #     "learning_rate": 1e-3,
    #     "epochs": 20,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },    
        
    # {
    #     "name": "densenet121_bs32_lr1e-4_20epochs",
    #     "model": "densenet121",
    #     "batch_size": 32,
    #     "learning_rate": 1e-4,
    #     "epochs": 20,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },    

    # {
    #     "name": "densenet121_bs16_lr1e-3_30epochs",
    #     "model": "densenet121",
    #     "batch_size": 16,
    #     "learning_rate": 1e-3,
    #     "epochs": 30,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },
    
    # {
    #     "name": "densenet121_bs16_lr1e-4_30epochs",
    #     "model": "densenet121",
    #     "batch_size": 16,
    #     "learning_rate": 1e-4,
    #     "epochs": 30,
    #     "freeze_backbone": True,
    #     "augmentation": True
    # },
]