import os
from PIL import Image
from pathlib import Path

dataset_folder = 'assets/Dataset'
dataset_path = Path(dataset_folder)

for file in dataset_path.rglob('*.jpg'):
    img = Image.open(file)
    w, h = img.size
    
    if (h > w):
        img = img.transpose(Image.ROTATE_90)
        print(f"Image {file} transposed")
        img.save(file)
    