import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from data_loader.data_loader import get_fitcow_loaders
import argparse
from tests.resnet50_tests import RESNET50_TESTS
from tests.densenet121_tests import DENSENET121_TESTS
from utils.focal_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight
import os

TESTS_MAP = {
    "resnet50": RESNET50_TESTS,
    "densenet121": DENSENET121_TESTS
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
from models.densenet121 import build_densenet121

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.plots import plot_confusion_matrix, plot_training_curves
from utils.logger import log_test

def get_model(exp_config, num_classes):
    model_name = exp_config['model']
    
    if model_name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
    elif model_name == "densenet121":
        return build_densenet121(
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
    
    class_to_idx = {
        '2.5': 0,
        '3.0': 1,
        '3.5': 2,
        '4.0': 3,
        '4.5': 4
    }
    
    df = pd.read_csv('assets/fitcow_label.csv')
    
    for exp in TESTS:
        print(f"Experiment: {exp['name']}")
        
        fold_accs = []
        fold_maes = []
        
        global_acc_history = {}
        global_loss_history = {}
        
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
            
            train_labels = df['class'].astype(str).map(class_to_idx)

            calc_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            
            tensor_weights = torch.tensor(calc_weights, dtype=torch.float).to(device)
    
            # criterion = nn.CrossEntropyLoss()
            criterion = FocalLoss(gamma=2.0, alpha=tensor_weights, num_classes=5)
            
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=exp["learning_rate"]
            )
            
            acc_history, loss_history = train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=exp['epochs']
            )
            
            global_acc_history[fold] = acc_history
            global_loss_history[fold] = loss_history
                        
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
        
        cm_path = f"results/confusion_matrix/cm_{exp['name']}.png"
        graph_path = f"results/plots/plot_{exp['name']}.png"
        csv_path = f"results/csv/result_table_{exp['name']}.csv"
        
        log_test(
            csv_path=csv_path,
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
        
        plot_confusion_matrix(global_cm, class_names, cm_path)
        plot_training_curves(global_acc_history, global_loss_history, graph_path)
        
if __name__ == '__main__':
    main()