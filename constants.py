import torch
import os

os.environ["IPDB_CONTEXT_LENGTH"] = "5"

CRITERION = torch.nn.BCEWithLogitsLoss()
N_EPOCHS = 1000
IM_DIM = 784
Z_DIM = 64
DISPLAY_STEP = 100
BATCH_SIZE = 128
HIDDEN_DIM = 128
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "."
LOG_PATH = "./logs"
CHECKPOINT_DIR = "./checkpoints"
IMAGE_DIR = "./images"
PRETRAINED_PATH = "./checkpoints/gan_mnist.pth"
