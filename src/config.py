# src/config.py

# Image settings
IMG_SIZE = 256
IMG_CHANNELS = 3

# Training settings
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

# Paths
DATASET_DIR = "data"
IMAGES_DIR = "images"
MASKS_DIR = "masks"

# Output
WEIGHTS_DIR = "weights"
MODEL_PATH = "weights/model.h5"

# Model settings
NUM_CLASSES = 1          # binary segmentation
FINAL_ACTIVATION = "sigmoid"