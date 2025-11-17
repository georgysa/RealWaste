import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import numpy as np
import pandas as pd  # <-- ADDED PANDAS IMPORT

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
# This code is now correct because pandas is imported.
class_names = list(RealWasteDataset(pd.DataFrame({'image_path': [], 'label': []})).classes.values())

# --- A. MODEL 1: ResNet50 - The High-Accuracy Benchmark ---
MODEL_NAME_1 = "ResNet50_Run3_Optimal_FineTuning"

print(f"\n======== Starting Training for {MODEL_NAME_1} ========")

# 3. Model Definition and Layer Freezing (Transfer Learning Strategy)
# VVVVVVVV MOVED MODEL SETUP BLOCK HERE VVVVVVVV
# Load ResNet50 with pre-trained weights (ImageNet)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# --- FREEZE THE BASE ---
# CRITICAL STEP 1: Freeze all layers in the base feature extractor.
for param in model_resnet.parameters():
    param.requires_grad = False

# --- MODIFY CLASSIFICATION HEAD (UNFREEZE) ---
# CRITICAL STEP 2: Replace the final Fully Connected (FC) layer to fit our 9 classes.
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# ^^^^^^^^ END OF MODEL SETUP BLOCK ^^^^^^^^

# --- 4. Optimization Setup (Loss & Base Parameters) ---
# TUNING PARAMETER 1: Loss Function (with Class Weights)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# --- STAGE 1: FAST TRAINING OF THE CLASSIFICATION HEAD (5 Epochs) ---
print("\n--- STAGE 1: Training FC Head (High LR) ---")

STAGE_1_LR = 1e-4  # High LR for fast initial learning
stage1_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_resnet.parameters()),
    lr=STAGE_1_LR
)
NUM_EPOCHS_S1 = 5

start_time_s1 = time.time()
trained_model_resnet, train_losses_s1, val_losses_s1, train_accuracies_s1, val_accuracies_s1 = train_model(
    model=model_resnet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=stage1_optimizer,
    num_epochs=NUM_EPOCHS_S1,
    device=device,
    model_name=MODEL_NAME_1 + " (Stage 1)"
)
training_time_s1 = time.time() - start_time_s1

# --- STAGE 2: DEEP FINE-TUNING OF LAYER 4 (10 Epochs) ---
print("\n--- STAGE 2: Deep Fine-Tuning Layer 4 (Tiny LR) ---")

# 1. UNFREEZE Layer 4 (CRITICAL: this must happen before the optimizer is redefined for Stage 2)
for param in model_resnet.layer4.parameters():
    param.requires_grad = True
print("Unfrozen layers: Layer 4 and the Final FC Head.")

# 2. Redefine the Optimizer with the TINY LR (uses the model state from Stage 1)
STAGE_2_LR = 1e-5
NUM_EPOCHS_S2 = 10
stage2_optimizer = optim.Adam(
    # The filter now correctly includes the newly unfrozen Layer 4 parameters
    filter(lambda p: p.requires_grad, model_resnet.parameters()),
    lr=STAGE_2_LR
)

start_time_s2 = time.time()
trained_model_resnet, train_losses_s2, val_losses_s2, train_accuracies_s2, val_accuracies_s2 = train_model(
    model=trained_model_resnet,  # Continue training the model instance
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=stage2_optimizer,
    num_epochs=NUM_EPOCHS_S2,
    device=device,
    model_name=MODEL_NAME_1 + " (Stage 2)"
)
training_time_s2 = time.time() - start_time_s2

# --- 5. Combine Histories and Update Metrics for Final Output ---
train_losses = train_losses_s1 + train_losses_s2
val_losses = val_losses_s1 + val_losses_s2
train_accuracies = train_accuracies_s1 + train_accuracies_s2
val_accuracies = val_accuracies_s1 + val_accuracies_s2
training_time = training_time_s1 + training_time_s2

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
print(f"\n--- RESULTS SUMMARY for {MODEL_NAME_1} ---")
print(f"Test Accuracy: {report['accuracy']:.4f}")
print(f"Test Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Total Trainable Parameters: {sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)}")
print(f"Training Time: {training_time:.2f}s")