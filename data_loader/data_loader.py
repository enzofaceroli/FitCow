from torch.utils.data import DataLoader
from data_loader.dataset import fitcow_dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_fitcow_loaders(train_df, test_df, dataset_path, batch_size, augment=False):
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    
    base_transforms = [
        transforms.Resize((224, 224)),
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader