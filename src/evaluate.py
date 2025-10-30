"""
evaluate.py
------------
Evaluates the trained AgriVision model on the test dataset
and performs inference on new images.

Author: Mukaram Ali
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.model import create_model, load_checkpoint


# ----------------------------------------------------------
# ⚙️ Load Test Data
# ----------------------------------------------------------
def load_test_data(data_dir="data/processed", img_size=224, batch_size=32):
    """
    Loads the test dataset using the same preprocessing pipeline.
    """
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_dataset.classes


# ----------------------------------------------------------
# 🧠 Evaluate Model
# ----------------------------------------------------------
def evaluate_model(model, test_loader, class_names, device, save_csv=True):
    """
    Evaluates model performance on test data and optionally saves metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{(correct/total):.4f}"
            })

    avg_loss = test_loss / total
    accuracy = correct / total

    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"📉 Test Loss: {avg_loss:.4f}")
    print("\n📋 Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ✅ Save metrics
    if save_csv:
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame(report).transpose()
        df.loc["overall", "accuracy"] = accuracy
        df.loc["overall", "loss"] = avg_loss
        csv_path = os.path.join("results", "eval_metrics.csv")
        df.to_csv(csv_path, index=True)
        print(f"📁 Metrics saved to: {csv_path}")

# ----------------------------------------------------------
# 🔍 Inference on Single Image
# ----------------------------------------------------------
def predict_image(model, image_path, class_names, device, img_size=224):
    """
    Predicts class for a single input image.
    """
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]

    plt.imshow(image)
    plt.title(f"Prediction: {pred_class}")
    plt.axis("off")
    plt.show()

    return pred_class


# ----------------------------------------------------------
# 🧩 Run from Terminal
# ----------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    # Load test data
    test_loader, class_names = load_test_data()

    # Load trained model
    model = create_model(num_classes=len(class_names))
    checkpoint = load_checkpoint("models/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Evaluate
    evaluate_model(model, test_loader, class_names, device)

    # Example inference (replace with your own test image)
    # predict_image(model, "data/raw/PlantVillage/Tomato___Late_blight/0a1b2c.jpg", class_names, device)