import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from pathlib import Path

class fitcow_dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        self.label_map = {
            '2.5': 0,
            '3.0': 1,
            '3.5': 2,
            '4': 3,
            '4.5': 4
        }
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        fold = f'fold_{self.img_labels.iloc[idx, 'fold']}'
        img_class = str(self.img_labels.iloc[idx, 'class'])
        file = self.img_labels.iloc[idx, 'file']

        img_path = os.path.join(self.img_dir, fold, img_class, file) 
        
        image = decode_image(img_path)
        label = self.label_map[self.img_labels.iloc[idx, 'class']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label