from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader.dataset import fitcow_dataset
from torchvision import transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_fitcow_loaders(train_df, test_df, dataset_path, batch_size, augment=False):
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    
    # base_transforms = [
    #     transforms.Resize((224, 224)),
    # ]
    
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    
    test_transforms = transforms.Compose(base_transforms + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if augment:
        train_transforms = transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transforms = test_transforms
    
    train_dataset = fitcow_dataset(train_df, dataset_path, transform=train_transforms)
    test_dataset = fitcow_dataset(test_df, dataset_path, transform=test_transforms)
    
    # train_labels = train_df['class'].astype(str).map({
    #     '2.5': 0,
    #     '3.0': 1,
    #     '3.5': 2,
    #     '4.0': 3,
    #     '4.5': 4
    # }).values

    # class_counts = np.bincount(train_labels)
    # class_weights = 1. / class_counts

    # sample_weights = class_weights[train_labels]

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader