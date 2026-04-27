import torch
from tqdm import tqdm 
from data_loader.data_loader import get_fitcow_loaders

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Starting Epoch: {epoch}")
    
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
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {acc:.2f}%")
        