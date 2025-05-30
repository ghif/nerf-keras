import keras
import tensorflow as tf
import numpy as np
import argparse
import json

from data_utils import split_data, create_dataset_pipeline
from models import create_nerf_model, render_rgb_depth

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/tiny_nerf.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

# Initialize global variables.
BATCH_SIZE = conf["BATCH_SIZE"]
NUM_SAMPLES = conf["NUM_SAMPLES"]
POS_ENCODE_DIMS = conf["POS_ENCODE_DIMS"]
EPOCHS = conf["EPOCHS"]
LEARNING_RATE = conf["LEARNING_RATE"]
NUM_LAYERS = conf["NUM_LAYERS"]
HIDDEN_DIM = conf["HIDDEN_DIM"]

AUTO = tf.data.AUTOTUNE
MODEL_DIR = "models"


# Download the dataset if it does not exist.
url = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)
data = keras.utils.get_file(origin=url)

# Load the dataset
data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

# Split the data into training and validation sets
train_images, val_images, train_poses, val_poses = split_data(images, poses, split_ratio=0.8)

# Make the dataset pipelines
train_ds = create_dataset_pipeline(
    train_images,
    train_poses,
    H,
    W,
    focal,
    NUM_SAMPLES,
    POS_ENCODE_DIMS,
    BATCH_SIZE,
    AUTO,
    near=2.0,
    far=6.0,
    shuffle=True,
    rand_sampling=True,
)

# Make the validation pipeline
val_ds = create_dataset_pipeline(
    val_images,
    val_poses,
    H,
    W,
    focal,
    NUM_SAMPLES,
    POS_ENCODE_DIMS,
    BATCH_SIZE,
    AUTO,
    near=2.0,
    far=6.0,
    shuffle=False, # Typically, validation data is not shuffled
    rand_sampling=False, # Or True, depending on if you want stochasticity here
)

num_pos = H * W * NUM_SAMPLES
nerf_model = create_nerf_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    num_pos=num_pos,
    pos_encode_dims=POS_ENCODE_DIMS
)

print(nerf_model.summary(expand_nested=True))

# Load the model weights if they exist

weight_path = f"{MODEL_DIR}/tinynerf-keras-20250530-134332/nerf.weights.h5"
nerf_model.load_weights(weight_path)
print("Model weights loaded successfully.")

train_imgs, train_rays = next(iter(train_ds))
train_rays_flat, train_t_vals = train_rays

val_imgs, val_rays = next(iter(val_ds))
val_rays_flat, val_t_vals = val_rays