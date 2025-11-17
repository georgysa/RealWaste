# train_utils.py

import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import NUM_CLASSES
from data_utils import set_seed  # Re-import set_seed for consistency


# --- A. Training Function ---

'''
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device=None, model_name="Model"):
    """
    Trains a given PyTorch model and evaluates it on a validation set per epoch.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss with weights).
        optimizer (optim): Optimizer used for training (e.g., Adam).
        num_epochs (int): Number of training epochs.
        device (torch.device): 'cuda' or 'cpu'.
        model_name (str): Name of the model for logging.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):

        # --- TRAINING PHASE ---
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Use tqdm for a professional progress bar
        with tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item() * images.size(0)

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                # Update progress bar
                tepoch.set_postfix(loss=running_loss / total_train, accuracy=100 * correct_train / total_train)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # --- VALIDATION PHASE ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient calculations (saving memory and speeding up)
            with tqdm(val_loader, desc=f"Val   Epoch {epoch + 1}/{num_epochs}", unit="batch") as vepoch:
                for images, labels in vepoch:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)

                    # Calculate validation accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    vepoch.set_postfix(loss=val_loss / total_val, accuracy=100 * correct_val / total_val)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    return model, train_losses, val_losses, train_accuracies, val_accuracies
'''


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=None, model_name="Model"):
    """
    Trains a PyTorch model with automated Early Stopping based on validation loss.

    Args:
        model (nn.Module): The neural network model to train.
        # ... (other arguments)
        num_epochs (int): Max number of training epochs (will stop early if performance plateaus).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Initialize Early Stopper: patience=5 means training stops if Val Loss doesn't improve for 5 consecutive epochs
    early_stopper = EarlyStopper(patience=5)

    for epoch in range(num_epochs):

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # ... (Training loop logic remains the same)
        with tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / total_train, accuracy=100 * correct_train / total_train)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Val   Epoch {epoch + 1}/{num_epochs}", unit="batch") as vepoch:
                for images, labels in vepoch:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    vepoch.set_postfix(loss=val_loss / total_val, accuracy=100 * correct_val / total_val)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Print the detailed epoch summary (with loss!)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # --- EARLY STOPPING CHECK ---
        if early_stopper.early_stop(epoch_val_loss):
            print(
                f"Early stopping triggered at epoch {epoch + 1}. Validation loss has not improved for {early_stopper.patience} epochs.")
            break  # Exit the training loop

    return model, train_losses, val_losses, train_accuracies, val_accuracies

# --- Early Stopping Class ---
class EarlyStopper:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
# --- B. Evaluation and Metrics Function ---

def evaluate_and_report(model, test_loader, device, class_names, model_name):
    """Evaluates the model on the test set and prints/plots detailed metrics."""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Classification Report (Precision, Recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n======== Classification Report for {model_name} ========")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Confusion Matrix (Visualizing misclassification)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    cm_percentage = cm * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Normalized Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"CM_{model_name.replace(' ', '_')}.png")
    plt.show()

    # 3. Plotting Training History (Accuracy and Loss)
    # The training history (losses, accuracies) will be passed separately by the main script

    return report


def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    """Plots the loss and accuracy history over epochs."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'Loss per Epoch ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy per Epoch ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"History_{model_name.replace(' ', '_')}.png")
    plt.show()