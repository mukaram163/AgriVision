"""
model.py
--------
Defines and prepares CNN model architectures for AgriVision.
We'll use a pretrained ResNet18 from torchvision and fine-tune it
for the PlantVillage dataset.

Author: Mukaram Ali
"""

import os
import torch
import torch.nn as nn
from torchvision import models


# ----------------------------------------------------------
# 🧠 Function: Create Model
# ----------------------------------------------------------
def create_model(num_classes, pretrained=True):
    """
    Creates and returns a modified ResNet18 model.

    Args:
        num_classes (int): Number of output classes (diseases)
        pretrained (bool): Whether to use ImageNet pretrained weights

    Returns:
        model (torch.nn.Module): The fine-tunable ResNet18 model
    """

    # Load pretrained ResNet18 from torchvision
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # Get the number of input features to the final fully-connected layer
    in_features = model.fc.in_features

    # Replace the classifier head with a new one for your dataset
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# ----------------------------------------------------------
# 💾 Function: Save and Load Model Checkpoints
# ----------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, val_acc,
                    filepath="models/best_model.pth"):
    """
    Saves model and optimizer states to a checkpoint file.

    Args:
        model (torch.nn.Module): Trained model
        optimizer (torch.optim.Optimizer): Optimizer used
        epoch (int): Current epoch
        val_acc (float): Best validation accuracy
        filepath (str): Path to save checkpoint file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved at: {filepath}")


def load_checkpoint(filepath):
    """
    Loads a saved model checkpoint.

    Args:
        filepath (str): Path to checkpoint file

    Returns:
        checkpoint (dict)
    """
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    print(f"✅ Loaded checkpoint from {filepath}")
    return checkpoint
