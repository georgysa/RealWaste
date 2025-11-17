# main_runner.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import numpy as np
import pandas as pd

# --- 1. Import Project Utilities ---
from config import NUM_CLASSES, RANDOM_SEED
from data_utils import create_dataloaders, set_seed, RealWasteDataset
from train_utils import train_model, evaluate_and_report, plot_history

# Set seed for reproducibility
set_seed(RANDOM_SEED)

# --- 2. Setup Device and Data Load ---
# New device setup for Apple Silicon (M1/M2/M3)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load all data utilities and class weights
train_loader, val_loader, test_loader, class_weights = create_dataloaders()

# Get class names from the Dataset object for reporting
# Note: We create a dummy instance just to access the class names map.
class_names = list(RealWasteDataset(pd.DataFrame({'image_path': [], 'label': []})).classes.values())

# --- A. MODEL 1: ResNet50 - The High-Accuracy Benchmark ---
MODEL_NAME_1 = "ResNet50_Run1_Baseline"

print(f"\n======== Starting Training for {MODEL_NAME_1} ========")

# 3. Model Definition and Layer Freezing (Transfer Learning Strategy)
# Load ResNet50 with pre-trained weights (ImageNet)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# --- FREEZE THE BASE ---
# CRITICAL STEP 1: Freeze all layers in the base feature extractor.
for param in model_resnet.parameters():
    param.requires_grad = False

# --- MODIFY CLASSIFICATION HEAD (UNFREEZE) ---
# CRITICAL STEP 2: Replace the final Fully Connected (FC) layer to fit our 9 classes.
# The weights in this new layer will be trained from scratch.
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# --- 4. Optimization Setup (Hyperparameters) ---
# TUNING PARAMETER 1: Loss Function (with Class Weights)
# We apply class weights to penalize errors on minority classes (Textile Trash, etc.).
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# TUNING PARAMETER 2: Optimizer and Learning Rate (LR)
LEARNING_RATE = 1e-4  # A small LR is essential when fine-tuning deep models
optimizer = optim.Adam(
    # Only pass parameters that require gradient updates (the new 'fc' layer)
    filter(lambda p: p.requires_grad, model_resnet.parameters()),
    lr=LEARNING_RATE
)

# --- 5. Training ---
NUM_EPOCHS = 10 # Start with 10 epochs for baseline, adjust later

start_time = time.time()
trained_model_resnet, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model=model_resnet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device,
    model_name=MODEL_NAME_1
)
training_time = time.time() - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

# --- 6. Evaluation on Test Set ---
start_time = time.time()
report = evaluate_and_report(
    model=trained_model_resnet,
    test_loader=test_loader,
    device=device,
    class_names=class_names,
    model_name=MODEL_NAME_1
)
testing_time = time.time() - start_time
print(f"Total Testing Time: {testing_time:.2f} seconds")

# --- 7. Plotting Results ---
plot_history(train_losses, val_losses, train_accuracies, val_accuracies, MODEL_NAME_1)

# --- 8. Record Results (For Report Comparison) ---
# You would save these results to a CSV or Excel file manually or via code
print(f"\n--- RESULTS SUMMARY for {MODEL_NAME_1} ---")
print(f"Test Accuracy: {report['accuracy']:.4f}")
print(f"Test Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Total Trainable Parameters: {sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)}")
print(f"Training Time: {training_time:.2f}s")