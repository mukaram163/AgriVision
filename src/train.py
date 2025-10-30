"""
train.py
--------
Training loop for AgriVision model using PyTorch.
Includes metrics logging and checkpoint saving.

Author: Mukaram Ali
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.model import create_model, save_checkpoint
from src.preprocessing import create_dataloaders


# ----------------------------------------------------------
# 🧠 Function: Train and Validate
# ----------------------------------------------------------
def train_model(num_epochs=5, batch_size=32, learning_rate=0.001, img_size=224, device=None):
    """
    Full training loop for the AgriVision model.

    Args:
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size for DataLoader
        learning_rate (float): Learning rate for optimizer
        img_size (int): Image resize dimension
        device: 'cuda' or 'cpu'
    """

    # 1️⃣ Device setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # 2️⃣ Create DataLoaders
    data_dir = os.path.join("data", "processed")
    train_loader, val_loader, _, class_names = create_dataloaders(data_dir, batch_size, img_size)
    num_classes = len(class_names)

    # 3️⃣ Create Model
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    # 4️⃣ Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5️⃣ Ensure models/ folder exists
    os.makedirs("models", exist_ok=True)

    best_val_acc = 0.0

    # 6️⃣ Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / total

        # 7️⃣ Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / total_val

        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}\n")

        # 8️⃣ Save checkpoint if model improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, filepath="models/best_model.pth")

    print(f"✅ Training complete. Best Val Acc: {best_val_acc:.4f}")


# ----------------------------------------------------------
# 🧩 Run from terminal
# ----------------------------------------------------------
if __name__ == "__main__":
    train_model(num_epochs=3)