import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Setting random seed to obtain reproducible results.
import tensorflow as tf

import keras
from keras import ops

import io
import datetime
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from data_utils import split_data, create_dataset_pipeline, get_rays, render_flat_rays, encode_position, sample_pdf
from models import create_nerf_model, render_predictions, create_nerf_complete_model

# tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/tiny_nerf_complete_h256.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

# Initialize global variables.
BATCH_SIZE = conf["BATCH_SIZE"]
NS_COARSE = conf["NS_COARSE"]
NS_FINE = conf["NS_FINE"]
L_XYZ = conf["L_XYZ"]
L_DIR = conf["L_DIR"]
H = conf["HEIGHT"]
W = conf["WIDTH"]
EPOCHS = conf["EPOCHS"]
LEARNING_RATE = conf["LEARNING_RATE"]
NUM_LAYERS = conf["NUM_LAYERS"]
SKIP_LAYER = conf["SKIP_LAYER"]
HIDDEN_DIM = conf["HIDDEN_DIM"]
WITH_GCS = conf["WITH_GCS"]
NEAR = 2.0
FAR = 6.0
RAND = True

AUTO = tf.data.AUTOTUNE

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# GCS bucket configuration
GCS_BUCKET_NAME = "keras-models"
GCS_MODEL_DIR = f"gs://{GCS_BUCKET_NAME}/nerf/models"
GCS_IMAGE_DIR = f"gs://{GCS_BUCKET_NAME}/nerf/images"

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if WITH_GCS:
    checkpoint_dir = os.path.join(GCS_MODEL_DIR, f"tinynerf-keras-{current_time}")
    visualization_dir = os.path.join(GCS_IMAGE_DIR, f"tinynerf-keras-{current_time}")
else:
    checkpoint_dir = os.path.join(MODEL_DIR, f"tinynerf-keras-{current_time}")

# Create the model directory if it does not exist.


# Download the dataset if it does not exist.
url = (
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
)
data = keras.utils.get_file(origin=url)

# Load the dataset
data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, _, _, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

pose0 = poses[0]
(ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose0)

t_vals_coarse = keras.ops.linspace(NEAR, FAR, NS_COARSE)
shape = list(ray_origins.shape[:-1]) + [NS_COARSE]
if RAND:
    noise = keras.random.uniform(shape=shape) * (FAR - NEAR) / NS_COARSE
    t_vals_coarse = t_vals_coarse + noise
else:
    t_vals_coarse = keras.ops.broadcast_to(t_vals_coarse, shape)

rays_coarse = ray_origins[..., None, :] + (
    ray_directions[..., None, :] * t_vals_coarse[..., None]
)
rays_flat_coarse = keras.ops.reshape(rays_coarse, [-1, 3])
# rays_flat_encoded = encode_position(rays_flat, POS_ENCODE_DIMS)


rays_flat_encoded_coarse = encode_position(rays_flat_coarse, L_XYZ)
dir_coarse_shape = ops.shape(rays_coarse[..., :3])
dirs_coarse = ops.broadcast_to(ray_directions[..., None, :], dir_coarse_shape)

dirs_flat_coarse = keras.ops.reshape(dirs_coarse, [-1, 3])
dirs_flat_encoded_coarse = encode_position(dirs_flat_coarse, L_DIR)

print(f"Shape of rays_flat_encoded: {rays_flat_encoded_coarse.shape}")
print(f"Shape of dirs_flat_encoded: {dirs_flat_encoded_coarse.shape}")

coarse_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR
)
print("Coarse Model Summary:")
print(coarse_model.summary(expand_nested=True))

predictions = coarse_model([rays_flat_encoded_coarse[None, ...], dirs_flat_encoded_coarse[None, ...]], training=False)

print(f"Predictions coarse: ({ops.min(predictions)}, {ops.max(predictions)}): {predictions}")

# Get the predictions from the nerf model and reshape it.
predictions_coarse = ops.reshape(predictions, (-1, H, W, NS_COARSE, 4))

rgb_coarse, depth_coarse, weight_coarse = render_predictions(predictions_coarse, t_vals_coarse[None, ...], rand=True)
t_vals_coarse_mid = (0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))
# t_vals_fine = sample_pdf(t_vals_coarse_mid[None, ...], depth_coarse[..., None], NS_FINE)
t_vals_fine = sample_pdf(t_vals_coarse_mid[None, ...], weight_coarse, NS_FINE)

t_vals_fine_all = ops.sort(ops.concatenate([t_vals_coarse[None, ...], t_vals_fine], axis=-1), axis=-1)

rays_fine = (ray_origins[..., None, :] + ray_directions[..., None, :] * t_vals_fine_all[..., None])
rays_flat_fine = ops.reshape(rays_fine, [-1, 3])
rays_flat_encoded_fine = encode_position(rays_flat_fine, L_XYZ)

dirs_fine_shape = ops.shape(rays_fine[..., :3])
dirs_fine = ops.broadcast_to(ray_directions[..., None, :], dirs_fine_shape)
dirs_flat_fine = keras.ops.reshape(dirs_fine, [-1, 3])
dirs_flat_encoded_fine = encode_position(dirs_flat_fine, L_DIR)
print(f"Shape of rays_flat_encoded_fine: {rays_flat_encoded_fine.shape}")
print(f"Shape of dirs_flat_encoded_fine: {dirs_flat_encoded_fine.shape}")



