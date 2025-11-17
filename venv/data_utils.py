# data_utils.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random
from collections import Counter
import time

from config import *  # Import constants from config.py


# --- A. Setup and Seed ---

def set_seed(seed):
    """Sets the random seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# --- B. Custom Dataset Class (Extracting Image Paths) ---

class RealWasteDataset(Dataset):
    """A custom PyTorch Dataset to handle the RealWaste images."""

    def __init__(self, data_df, transforms=None):
        """
        Args:
            data_df (pd.DataFrame): DataFrame with 'image_path' and 'label'.
            transforms (callable, optional): Optional transform to be applied.
        """
        self.data_paths = data_df['image_path'].to_list()
        self.labels = data_df['label'].to_list()
        self.transforms = transforms

        self.classes = {
            0: 'Cardboard', 1: 'Food Organics', 2: 'Glass', 3: 'Metal',
            4: 'Miscellaneous Trash', 5: 'Paper', 6: 'Plastic',
            7: 'Textile Trash', 8: 'Vegetation'
        }

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]

        # Load the image and ensure it has 3 channels (RGB)
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label


# --- C. Data Splitting (Stratified Split) ---

def train_val_test_split(root_dir, test_size, val_size, random_seed):
    """
    Loads all data paths and performs a stratified split into three DataFrames.
    Stratification ensures class distribution is preserved in each set.
    """
    print("Collecting dataset information...")
    full_data = []
    full_labels = []

    # Map class names to numeric labels
    # class_map = {name: label for label, name in enumerate(RealWasteDataset(pd.DataFrame()).classes.values())}
    # NEW, corrected code (pass a dummy dataframe with the required column names):
    dummy_df = pd.DataFrame({'image_path': [], 'label': []})
    class_map = {name: label for label, name in enumerate(RealWasteDataset(dummy_df).classes.values())}

    # Iterate through class directories to collect all paths and labels
    for class_name, label in class_map.items():
        class_dir = os.path.join(root_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                full_data.append(img_path)
                full_labels.append(label)

    full_dataframe = pd.DataFrame({'image_path': full_data, 'label': full_labels})

    # 1. Train/Validation/Test Split (Stratified)

    # Split the dataset into Train/Val and Test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_val_idx, test_idx in sss.split(full_dataframe, full_dataframe['label']):
        train_val_df = full_dataframe.iloc[train_val_idx]
        test_df = full_dataframe.iloc[test_idx]

    # Calculate the adjusted validation size relative to the remaining data
    val_adjusted_size = val_size / (1 - test_size)

    # Split Train/Val into final Train and Validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_adjusted_size, random_state=random_seed)
    for train_idx, val_idx in sss.split(train_val_df, train_val_df['label']):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    return train_df, val_df, test_df


# --- D. Transformation and Augmentation Pipelines ---

def get_transforms(img_size):
    """Defines the necessary data transformations."""

    # Transformations applied ONLY to the training data (to prevent overfitting)
    # Note: These are standard transformations found in the notebook.
    train_augment_transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.RandomResizedCrop(size=img_size[0], scale=(0.6, 1.0)),  # Randomly crop and resize
        transforms.ToTensor(),  # Convert image to a PyTorch Tensor
    ])

    # Transformations applied to Validation and Test data (Standardization only)
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    return train_augment_transform, val_test_transform


# --- E. Create DataLoaders and Class Weights ---

def create_dataloaders():
    """Combines all functions to produce the final DataLoaders and weights."""

    # 1. Split Data
    train_df, val_df, test_df = train_val_test_split(DATA_DIR, TEST_SPLIT, VAL_SPLIT, RANDOM_SEED)

    # 2. Get Transforms
    train_transforms, val_test_transforms = get_transforms(IMG_SIZE)

    # 3. Create Datasets (using augmentation only on the training set)
    train_dataset = RealWasteDataset(train_df, transforms=train_transforms)
    val_dataset = RealWasteDataset(val_df, transforms=val_test_transforms)
    test_dataset = RealWasteDataset(test_df, transforms=val_test_transforms)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Calculate Class Weights (Crucial for Imbalanced Data)
    # Calculate how many images belong to each class in the training set
    class_counts = np.bincount(train_df['label'])
    # Calculate weights inversely proportional to class frequency
    class_weights = 1.0 / class_counts
    # Normalize weights so they sum to the number of classes
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    print(f"\n--- Data Load Complete ---")
    print(f"Total Images: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Train/Val/Test Split: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Calculated Class Weights: {class_weights}")

    return train_loader, val_loader, test_loader, class_weights


# --- F. Initial Test ---
# You can run this file directly to test the data loading:
if __name__ == '__main__':
    start_time = time.time()
    train_dl, val_dl, test_dl, weights = create_dataloaders()
    end_time = time.time()
    print(f"Time taken to load data: {round(end_time - start_time, 2)} seconds")