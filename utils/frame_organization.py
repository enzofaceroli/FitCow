import pandas as pd 
import os
import shutil

source_folder = 'assets/Frames/selecionados'
destination_folder = 'assets/Dataset'
df_file = 'assets/fitcow.csv'

def main ():
    df = pd.read_csv(df_file, decimal=',', sep=';')
    map = dict(zip(df['ID'].astype(str), df['mode_g1'].astype(str)))
    classes = df['mode_g1'].drop_duplicates(ignore_index=True)
        
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for c in classes:
        path = destination_folder + f'/{str(c)}'
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        unique_individuals = set()
        
        for f in os.listdir(path):
            animal_name = f.split('_')[0]
            unique_individuals.add(animal_name)
            
        print (f"Class {c}: Number of images: {len(os.listdir(path))}")        
        print(f"Class {c}: Number of animals: {len(unique_individuals)}\n")
        
    for f in os.listdir(source_folder):
        animal_name = f.split('_')[0]
        
        if animal_name in map:
            animal_class = map[animal_name]
            
            dest = os.path.join(destination_folder, animal_class)
            
            src_file = os.path.join(source_folder, f)
            destination_file = os.path.join(dest, f)
            
            if not os.path.exists(destination_file):
                shutil.copy2(src_file, destination_file)    
                print(f"Animal {animal_name} copied to folder {animal_class}")
            
        else:
            print (f"{animal_name} not found")
    
if __name__ == '__main__':
    main()