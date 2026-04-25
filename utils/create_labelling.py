import os
import pandas as pd
from pathlib import Path

base_csv = 'assets/fitcow.csv'
destination_folder = 'assets/'
destinantion_file_name = 'fitcow_label.csv'

dataset_folder = 'assets/Dataset'
dataset_path = Path(dataset_folder)

def main():
    df = pd.read_csv(base_csv, sep=';', decimal=',')
    df = df.drop(['mode_g2'], axis=1)
    
    animal_class_dict = dict(zip(df['ID'].astype(str), df['mode_g1'].astype(str)))
    
    count = 0
    labels = []
    
    for file in dataset_path.rglob('*.jpg'):
        img_path = str(file.relative_to(dataset_path))
        animal_name = file.name.split('_')[0]
        animal_class = animal_class_dict[animal_name]
        
        label_dict = {
            'file':file.name,
            'class':animal_class,
            'fold': file.parent.parent.name[-1:]
            }
                
        labels.append(label_dict)
        
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(destination_folder + destinantion_file_name, index=False)
        
        
if __name__ == '__main__':
    main()