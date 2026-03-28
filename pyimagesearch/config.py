# %%
# import the necessary packages
import torch
import os

# %%
# base path of the dataset
TRAIN_DATASET_PATH = os.path.join("dataset", "train")
TEST_DATASET_PATH = os.path.join("dataset", "test")
# os.path.join('a', 'b') returns a/b

# %%
# define the path to the images and masks dataset
TRAIN_IMAGE_DATASET_PATH = os.path.join(TRAIN_DATASET_PATH, "images")
TRAIN_MASK_DATASET_PATH = os.path.join(TRAIN_DATASET_PATH, "masks")

TEST_IMAGE_DATASET_PATH = os.path.join(TEST_DATASET_PATH, "images")

# define the validation set split
VAL_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# %%
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 60
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# DEFINE THE OUTPUT IMG DIMENSION
OUT_IMAGE_WIDTH = 101
OUT_IMAGE_HEIGHT = 101

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# My modification: for the model saving
BEST_VALID_LOSS = float('inf')

# %%
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
VAL_PATHS = os.path.join(BASE_OUTPUT, "val_paths.txt")
PRED_PATHS = os.path.join(BASE_OUTPUT, "predictions.csv")

KAGGLE_MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt_kaggle.pth")

# %%
