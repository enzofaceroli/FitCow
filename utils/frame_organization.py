import pandas as pd 
import os
import shutil

source_folder = 'assets/Frames/selecionados'
destination_folder = 'assets/Dataset'
df_file = 'assets/fitcow.csv'

def create_dataset_folders(classes, folds):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    for i in range (1, folds+1):
        for c in classes:
            path = destination_folder + f'/fold_{i}/{str(c)}'
            
            if not os.path.exists(path):
                os.makedirs(path)

def main ():
    df = pd.read_csv(df_file, decimal=',', sep=';')
    df = df.drop(['mode_g2'], axis=1)
            
    animal_image_count = {}
    
    for f in os.listdir(source_folder):
        animal_name = f.split('_')[0]
        
        if animal_name in animal_image_count:
            animal_image_count[animal_name] += 1
        else:
            animal_image_count[animal_name] = 1
        
    df['image_count'] = df['ID'].map(animal_image_count).fillna(0)
    df = df[df['image_count'] != 0].sort_values(by='mode_g1')
    
    df.to_csv('assets/image_count.csv', index=None)

    map = dict(zip(df['ID'].astype(str), df['mode_g1'].astype(str)))
    classes = sorted(df['mode_g1'].drop_duplicates(ignore_index=True))
    folds = 5
        
    create_dataset_folders(classes, folds)

    folds_total = {i: 0 for i in range(1, folds + 1)}
    folds_classes = {i: {c: 0 for c in classes} for i in range(1, folds + 1)}

    for c in classes:
        class_df = df[df['mode_g1'] == c].sort_values(by='image_count', ascending=False)
        print(f"Processando classe {c}...")
        
        for _, row in class_df.iterrows():
            animal = str(row['ID'])
            qtd = int(row['image_count'])

            # best_fold = min(range(1, folds + 1), key=lambda i: (folds_classes[i][c], folds_total[i]))
            best_fold = min(range(1, folds + 1), key=lambda i: (folds_total[i], folds_classes[i][c]))

            folds_classes[best_fold][c] += qtd
            folds_total[best_fold] += qtd

            dest_path = os.path.join(destination_folder, f'fold_{best_fold}', str(c))
            
            for f in os.listdir(source_folder):
                if f.startswith(animal + '_'):
                    src_file = os.path.join(source_folder, f)
                    dst_file = os.path.join(dest_path, f)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        
    print("\n Image count per fold:")
    for i in range(1, folds + 1):
        total_fold = 0
        for c in classes:
            path = os.path.join(destination_folder, f'fold_{i}', str(c))
            total_fold += len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"Fold_{i}: {total_fold} images")

    print("\n Image count per class:")
    for c in classes:
        print(f"\nClass {c}:")
        for i in range(1, folds + 1):
            path = os.path.join(destination_folder, f'fold_{i}', str(c))
            qtd_classe_fold = len(os.listdir(path)) if os.path.exists(path) else 0
            print(f"  Fold {i}: {qtd_classe_fold} images")
        
                    
if __name__ == '__main__':
    main()