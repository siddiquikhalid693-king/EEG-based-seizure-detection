"""
EEG Deep Learning Model Training Script
Trains EEGNet and CNN-LSTM PyTorch models on multi-channel EEG epochs and evaluates classification metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset.generator import EEGDataGenerator, CLASS_NAMES
from models.eegnet import EEGNet
from models.cnn_lstm import CNN_LSTM

def train_model(model_type: str = 'eegnet', epochs: int = 15, batch_size: int = 16, lr: float = 0.001) -> str:
    print(f"=== Starting Training for {model_type.upper()} Deep Learning Model ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Generate Synthetic Training & Validation Dataset
    gen = EEGDataGenerator(fs=256)
    print("Generating synthetic EEG training dataset...")
    X, y = gen.generate_dataset(n_samples_per_class=120, epoch_sec=4.0)  # Shape: X=(N, 14, 1024), y=(N,)
    
    # Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    n_channels = X.shape[1]
    n_samples = X.shape[2]
    n_classes = len(CLASS_NAMES)

    if model_type.lower() == 'eegnet':
        model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples)
    else:
        model = CNN_LSTM(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop
    best_acc = 0.0
    saved_model_path = f"{model_type.lower()}_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                y_true_list.extend(targets.cpu().numpy())
                y_pred_list.extend(predicted.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), saved_model_path)
            # Also save general model.pth
            torch.save(model.state_dict(), 'model.pth')

    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to '{saved_model_path}' and 'model.pth'")
    
    # 4. Detailed Classification Report
    target_names = [CLASS_NAMES[i] for i in range(n_classes)]
    print("\nValidation Classification Report:")
    print(classification_report(y_true_list, y_pred_list, target_names=target_names))

    return saved_model_path

if __name__ == '__main__':
    train_model(model_type='eegnet', epochs=15)
