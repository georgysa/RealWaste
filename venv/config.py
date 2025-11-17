# config.py

# --- Data Configuration ---
BATCH_SIZE = 64
IMG_SIZE = (224, 224)
TEST_SPLIT = 0.1  # 10% for final, unbiased evaluation
VAL_SPLIT = 0.1   # 10% for hyperparameter tuning during training
RANDOM_SEED = 42

DATA_DIR = '/Users/georgysalem/Desktop/Real waste classification/realwaste-main/RealWaste'

# --- Model Configuration ---
NUM_CLASSES = 9