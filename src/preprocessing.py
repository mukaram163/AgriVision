"""
preprocessing.py
----------------
This file handles all preprocessing tasks for the AgriVision project:
1. Splitting the dataset into train/val/test sets
2. Applying image transformations (resize, flip, rotate, normalize)
3. Creating PyTorch Datasets and DataLoaders

Author: Mukaram Ali
"""

import os
import shutil
import random
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ----------------------------------------------------------
# 🧩 1. Function: Split Dataset into Train / Val / Test
# ----------------------------------------------------------
def split_dataset(
    source_dir,
    dest_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
):
    """
    Splits the PlantVillage dataset into train, validation, and test folders.

    Args:
        source_dir (str): Path to raw dataset (e.g., ../data/raw/PlantVillage)
        dest_dir (str): Destination path for processed data
            (e.g., ../data/processed)
        train_ratio (float): Fraction of data for training
        val_ratio (float): Fraction of data for validation
        test_ratio (float): Fraction of data for testing

    Example:
        split_dataset("../data/raw/PlantVillage", "../data/processed")
    """

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    classes = [
        d
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        images = os.listdir(class_dir)
        random.shuffle(images)  # randomize image order

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Divide image names into splits
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        # Copy files into respective folders
        for split_name, split_files in splits.items():
            split_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for img in tqdm(split_files, desc=f"{cls} → {split_name}"):
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(split_dir, img)
                shutil.copy2(src_path, dst_path)

    print("✅ Dataset successfully split into train, val, and test folders.")


# ----------------------------------------------------------
# 🧩 2. Function: Define Image Transformations
# ----------------------------------------------------------
def get_transforms(img_size=224):
    """
    Defines image preprocessing and augmentation steps.

    Args:
        img_size (int): Size to resize all images (default 224x224)

    Returns:
        (train_transforms, val_test_transforms)
    """

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # randomly flip horizontally
        transforms.RandomRotation(15),  # rotate image by ±15°
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),  # convert to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # normalize (ImageNet mean/std)
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transforms, val_test_transforms


# ----------------------------------------------------------
# 🧩 3. Function: Create Datasets and DataLoaders
# ----------------------------------------------------------
def create_dataloaders(
    data_dir="../data/processed",
    batch_size=32,
    img_size=224,
):
    """
    Creates train, validation, and test dataloaders for PyTorch.

    Args:
        data_dir (str): Path to processed dataset folder
        batch_size (int): Number of images per batch
        img_size (int): Image resize dimension (default 224)

    Returns:
        train_loader, val_loader, test_loader, class_names
    """

    train_transforms, val_test_transforms = get_transforms(img_size)

    # Create PyTorch datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_test_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=val_test_transforms
    )

    # Create DataLoaders (PyTorch’s way to feed batches efficiently)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    class_names = train_dataset.classes
    print(
        f"✅ Datasets ready: {len(train_dataset)} train, "
        f"{len(val_dataset)} val, {len(test_dataset)} test"
    )
    print(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


# ----------------------------------------------------------
# 🧩 Example (Run once to split dataset)
# ----------------------------------------------------------
if __name__ == "__main__":
    # This block only runs when you execute: python src/preprocessing.py
    source = "../data/raw/PlantVillage"
    destination = "../data/processed"
    split_dataset(source, destination)
