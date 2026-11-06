
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.dataset import fitcow_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_fitcow_loaders(
    train_df,
    test_df,
    dataset_path,
    batch_size,
    augment=False
):
    train_transforms = [
        transforms.Resize((224, 224))
    ]

    if augment:
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5)
        ])

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_transforms = transforms.Compose(train_transforms)

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_dataset = fitcow_dataset(
        train_df,
        dataset_path,
        transform=train_transforms
    )

    test_dataset = fitcow_dataset(
        test_df,
        dataset_path,
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader
