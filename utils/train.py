import torch
from tqdm import tqdm 
from data_loader.data_loader import get_fitcow_loaders
from utils.evaluate import evaluate_model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    
    acc_history = []
    loss_history = []
    eval_acc_history = []
    
    for epoch in range(epochs):
        print(f"Starting Epoch: {epoch+1}")
    
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            correct += torch.sum(predicted == labels.data).item()
            total += images.size(0)
            
        epoch_loss = running_loss / total
        acc = 100.0 * correct / total
        
        acc_history.append(acc)
        loss_history.append(epoch_loss)

        eval_acc, cm, mae = evaluate_model(model, test_loader, device)
        
        eval_acc_history.append(eval_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {acc:.2f}%")
        print(f"Eval Acc: {eval_acc:.2f}%\n")
    
    return acc_history, loss_history, eval_acc_history