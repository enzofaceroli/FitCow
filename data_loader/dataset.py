import os
from torch.utils.data import Dataset
from PIL import Image

class fitcow_dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.label_map = {
            '2.5': 0,
            '3.0': 1,
            '3.5': 2,
            '4.0': 3,
            '4.5': 4
        }
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        fold = f"fold_{self.img_labels['fold'].iloc[idx]}"
        img_class = str(self.img_labels['class'].iloc[idx])
        file = self.img_labels['file'].iloc[idx]

        img_path = os.path.join(
            self.img_dir, fold, "images", file
        )

        image = Image.open(img_path).convert('RGB')
        label = self.label_map[img_class]

        if self.transform:
            image = self.transform(image)

        return image, label