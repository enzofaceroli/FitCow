import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from data_loader.data_loader import get_fitcow_loaders
import argparse
from tests.resnet50_tests import RESNET50_TESTS
import os

TESTS_MAP = {
    "resnet50": RESNET50_TESTS,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=TESTS_MAP.keys()
    )
    
    return parser.parse_args()


from models.resnet50 import build_resnet50

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.plots import plot_confusion_matrix
from utils.logger import log_test

def get_model(exp_config, num_classes):
    model_name = exp_config['model']
    
    if model_name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
def main():
    os.makedirs("results", exist_ok=True)
    dataset_path = "assets/Dataset"
    args = parse_args()
    TESTS = TESTS_MAP[args.arch]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    class_map = {
        0: '2.5',
        1: '3.0',
        2: '3.5',
        3: '4.0',
        4: '4.5'
    }
    
    df = pd.read_csv('assets/fitcow_label.csv')
    
    for exp in TESTS:
        print(f"Experiment: {exp['name']}")
        
        fold_accs = []
        fold_maes = []
        
        global_cm = np.zeros((5,5), dtype=int)
        
        for fold in range(1, 6):
            train_df = df[df['fold'] != fold]
            test_df = df[df['fold'] == fold]

            train_loader, test_loader = get_fitcow_loaders(
                train_df=train_df,
                test_df=test_df,
                dataset_path=dataset_path,
                batch_size=exp['batch_size'],
                augment=exp['augmentation']
            )
            
            model = get_model(exp, num_classes=5)
            model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=exp["learning_rate"]
            )
            
            train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=exp['epochs']
            )
            
            acc, cm, mae = evaluate_model(model, test_loader, device)
            print(f"Fold {fold} results:\nAcc = {acc:.2f}%\nMAE = {mae:.3f}")
            
            fold_accs.append(acc)
            fold_maes.append(mae)
            
            global_cm += cm
            
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
            
        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        
        print(f"Experiment {exp['name']} results:")
        print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.3f}")
        print(f"Mean Accuracy Error: {mean_mae:.2f} ± {std_mae:.3f}\n")

        class_names = list(class_map.values())
        cm_save_path = f"results/confusion_matrix_{exp['name']}.png"
        
        log_test(
            csv_path=f"results/result_table_{exp['name']}.csv",
            data={
                "experiment_name": exp["name"],
                "model": exp["model"],
                "batch_size": exp["batch_size"],
                "learning_rate": exp["learning_rate"],
                "epochs": exp["epochs"],
                "augmentation": exp["augmentation"],
                "input_size": 224,
                "accuracy": mean_acc
            }
        )
        
        plot_confusion_matrix(global_cm, class_names, cm_save_path)
        
if __name__ == '__main__':
    main()