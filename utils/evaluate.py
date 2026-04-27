import torch 
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    test_bar = tqdm(test_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for images, labels in test_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = (all_preds == all_labels).mean()
        cm = confusion_matrix(all_labels, all_preds)
        mae_classes = np.abs(all_preds - all_labels).mean()
        real_mae = mae_classes * 0.5
        
        return acc, cm, real_mae