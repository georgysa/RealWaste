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

# Get class names for reporting
class_names = list(RealWasteDataset(pd.DataFrame({'image_path': [], 'label': []})).classes.values())

# =========================================================
# --- A. MODEL 1: ResNet50 - The High-Accuracy Benchmark ---
# (Final Run 4: Optimal Two-Stage Fine-Tuning)
# =========================================================
MODEL_NAME_1 = "ResNet50_Optimal_Benchmark"
print(f"\n======== Starting Training for {MODEL_NAME_1} ========")

# 3. Model Definition and Layer Freezing (Correctly placed before training)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# --- FREEZE THE BASE ---
for param in model_resnet.parameters():
    param.requires_grad = False

# --- MODIFY CLASSIFICATION HEAD (UNFREEZE) ---
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# --- 4. Optimization Setup ---
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# --- STAGE 1: Training FC Head (High LR) ---
print("\n--- STAGE 1: Training FC Head (LR 1e-4) ---")

STAGE_1_LR = 1e-4
stage1_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_resnet.parameters()),
    lr=STAGE_1_LR
)
NUM_EPOCHS_S1 = 5  # Fixed low epoch count for fast head stabilization

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

# --- STAGE 2: Deep Fine-Tuning Layer 4 (50 Max Epochs with Early Stop) ---
print("\n--- STAGE 2: Deep Fine-Tuning Layer 4 (LR 1e-5) ---")

# 1. UNFREEZE Layer 4 (CRITICAL: this must happen before the optimizer is redefined for Stage 2)
for param in model_resnet.layer4.parameters():
    param.requires_grad = True

# 2. Redefine the Optimizer with the TINY LR (uses the model state from Stage 1)
STAGE_2_LR = 1e-5  # Optimal LR determined from tuning runs (Fixes the stall)
NUM_EPOCHS_S2 = 50  # Max epochs set high for Early Stopper to find convergence point

stage2_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, trained_model_resnet.parameters()),
    lr=STAGE_2_LR
)

start_time_s2 = time.time()
trained_model_resnet, train_losses_s2, val_losses_s2, train_accuracies_s2, val_accuracies_s2 = train_model(
    model=trained_model_resnet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=stage2_optimizer,
    num_epochs=NUM_EPOCHS_S2,
    device=device,
    model_name=MODEL_NAME_1 + " (Stage 2)"
)
training_time_s2 = time.time() - start_time_s2

# --- FINAL EVALUATION AND SUMMARY (ResNet) ---
train_losses = train_losses_s1 + train_losses_s2
val_losses = val_losses_s1 + val_losses_s2
train_accuracies = train_accuracies_s1 + train_accuracies_s2
val_accuracies = val_accuracies_s1 + val_accuracies_s2
training_time = training_time_s1 + training_time_s2  # Final total training time

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
print(f"Total ResNet Testing Time: {testing_time:.2f} seconds")

# --- 7. Plotting Results ---
plot_history(train_losses, val_losses, train_accuracies, val_accuracies, MODEL_NAME_1)

# --- 8. Record Results (ResNet Summary) ---
print(f"\n--- RESULTS SUMMARY for {MODEL_NAME_1} ---")
print(f"Test Accuracy: {report['accuracy']:.4f}")
print(f"Test Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Total Trainable Parameters: {sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)}")
print(f"Training Time: {training_time:.2f}s")

'''
# =========================================================
# --- B. MODEL 2: MobileNetV2 - The Efficiency Benchmark ---
# (This block will run immediately after ResNet completes)
# =========================================================
MODEL_NAME_2 = "MobileNetV2_Optimal_Benchmark"
print(f"\n======== Starting Training for {MODEL_NAME_2} (Efficiency Benchmark) ========")

# 3. Model Definition and Layer Freezing
model_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# --- FREEZE THE BASE (All layers) ---
for param in model_mobilenet.parameters():
    param.requires_grad = False
print("Trainable layers: Final FC Head ONLY (Max Efficiency).")

# --- MODIFY CLASSIFICATION HEAD (UNFREEZE) ---
num_ftrs = model_mobilenet.classifier[-1].in_features
model_mobilenet.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)

# --- 4. Optimization Setup ---
# Loss function remains weighted (criterion is already defined)

# --- TRAINING: Single Stage for Speed (Max 50 Epochs, Early Stop will trigger fast) ---
STAGE_LR = 1e-4  # Fast learning rate
NUM_EPOCHS_M2 = 50  # Max epochs set high for Early Stopper

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_mobilenet.parameters()),
    lr=STAGE_LR
)

start_time = time.time()
trained_model_mobilenet, train_losses_m2, val_losses_m2, train_accuracies_m2, val_accuracies_m2 = train_model(
    model=model_mobilenet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS_M2,
    device=device,
    model_name=MODEL_NAME_2
)
training_time_m2 = time.time() - start_time

# --- 6. Evaluation on Test Set (MobileNet) ---
start_time = time.time()
report = evaluate_and_report(
    model=trained_model_mobilenet,
    test_loader=test_loader,
    device=device,
    class_names=class_names,
    model_name=MODEL_NAME_2
)
testing_time_m2 = time.time() - start_time
print(f"Total MobileNetV2 Testing Time: {testing_time_m2:.2f} seconds")

# --- 7. Plotting Results (MobileNet) ---
plot_history(train_losses_m2, val_losses_m2, train_accuracies_m2, val_accuracies_m2, MODEL_NAME_2)

# --- 8. Record Results (MobileNet Summary) ---
print(f"\n--- RESULTS SUMMARY for {MODEL_NAME_2} ---")
print(f"Test Accuracy: {report['accuracy']:.4f}")
print(f"Test Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Total Trainable Parameters: {sum(p.numel() for p in model_mobilenet.parameters() if p.requires_grad)}")
print(f"Training Time: {training_time_m2:.2f}s")
'''