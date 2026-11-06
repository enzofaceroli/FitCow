import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from data_loader.data_loader import get_fitcow_loaders
import argparse
from tests.resnet50_tests import RESNET50_TESTS
from tests.densenet121_tests import DENSENET121_TESTS
from tests.all_tests import ALL_TESTS
from utils.focal_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight
import os

TESTS_MAP = {
    "resnet50": RESNET50_TESTS,
    "densenet121": DENSENET121_TESTS,
    "all": ALL_TESTS
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

# from models.resnet50 import build_resnet50
# from models.densenet121 import build_densenet121
# from models.vgg16 import build_vgg16
# from models.resnet18 import build_resnet18
# from models.vit_tiny import build_vit_tiny
# from models.dinov2 import build_dinov2

from models.coral import build_densenet121, build_dinov2, build_resnet18, build_vgg16, build_resnet50, build_convnext_tiny

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
        
    elif model_name == "dinov2":
        return build_dinov2(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
    elif model_name == "densenet121":
        return build_densenet121(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
    elif model_name == "vgg16":
        return build_vgg16(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
    elif model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            freeze_backbone=exp_config["freeze_backbone"]
        )
        
    # elif model_name == "vit_tiny":
    #     return build_vit_tiny(
    #         num_classes=num_classes,
    #         freeze_backbone=exp_config["freeze_backbone"]
    #     )
    
    elif model_name == "convnext_tiny":
        return build_convnext_tiny(
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
        exp_name = f"model_{exp['model']}_bs{exp['batch_size']}_lr{exp['learning_rate']}_epochs{exp['epochs']}" 
        print(f"Experiment: {exp_name}")
                
        fold_accs = []
        fold_maes = []
        fold_losses = []

        global_acc_history = {}
        global_mae_history = {}
        global_loss_history = {}

        global_eval_acc_history = {}
        global_eval_mae_history = {}
        global_eval_loss_history = {}
        
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
            
            train_labels = train_df['class'].astype(str).map(class_to_idx)

            calc_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            
            tensor_weights = torch.tensor(calc_weights, dtype=torch.float).to(device)
            tensor_weights = tensor_weights / tensor_weights.sum()
            tensor_weights *= len(tensor_weights)

            # criterion = nn.CrossEntropyLoss(weight=tensor_weights)
            
            
            optimizer = optim.AdamW(
                model.parameters(), lr=exp["learning_rate"], weight_decay=1e-4
            )
            
            acc_history, mae_history, loss_history, eval_acc_history, eval_mae_history, eval_loss_history = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    epochs=exp['epochs']
                )
            
            global_acc_history[fold] = acc_history
            global_mae_history[fold] = mae_history
            global_loss_history[fold] = loss_history

            global_eval_acc_history[fold] = eval_acc_history
            global_eval_mae_history[fold] = eval_mae_history
            global_eval_loss_history[fold] = eval_loss_history   

            acc, mae, cm, final_eval_loss = evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )
            print(f"Fold {fold} results:\nAcc = {acc:.2f}%\nLoss: {final_eval_loss:.4f}")
            
            fold_accs.append(acc)
            fold_maes.append(mae)
            fold_losses.append(final_eval_loss)
            global_cm += cm
            
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)

        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)

        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        
        print(f"Experiment {exp_name} results:")

        print(f"Accuracy: {mean_acc:.2f} ± {std_acc:.3f}")
        print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"Loss: {mean_loss:.4f} ± {std_loss:.4f}")

        class_names = list(class_map.values())
        
        base_path = f"{exp['model']}/{exp_name}"
        os.makedirs(base_path, exist_ok=True)

        cm_path = f"{base_path}/confusion_matrix.pdf"
        graph_path = f"{base_path}/plot.pdf"
        csv_path = f"{base_path}/result_table.csv"
                
        log_data = {
            "experiment_name": exp_name,
            "model": exp["model"],
            "batch_size": exp["batch_size"],
            "learning_rate": exp["learning_rate"],
            "epochs": exp["epochs"],
            "augmentation": exp["augmentation"],
            # "input_size": 224,
            "mean_accuracy": f"{mean_acc:.4f}",
            "std_accuracy": f"{std_acc:.4f}",
            "mean_mae": f"{mean_mae:.4f}",
            "std_mae": f"{std_mae:.4f}",
            "mean_loss": f"{mean_loss:.4f}",
            "std_loss": f"{std_loss:.4f}"
        }
                
        for i in range(5):
            log_data[f"fold_{i+1}_acc"] = f"{fold_accs[i]:.4f}"
            log_data[f"fold_{i+1}_mae"] = f"{fold_maes[i]:.4f}"
            log_data[f"fold_{i+1}_loss"] = f"{fold_losses[i]:.4f}"
        
        log_test(
            csv_path=csv_path,
            data=log_data
        )
        
        plot_confusion_matrix(global_cm, class_names, cm_path)
        plot_training_curves(
            global_acc_history,
            global_mae_history,
            global_loss_history,
            global_eval_acc_history,
            global_eval_mae_history,
            global_eval_loss_history,
            graph_path
        )
        
if __name__ == '__main__':
    main()