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

from data_utils import split_data, create_tiny_dataset_pipeline
from models import create_nerf_complete_model, NeRFTrainer

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
EPOCHS = conf["EPOCHS"]
LEARNING_RATE = conf["LEARNING_RATE"]
NUM_LAYERS = conf["NUM_LAYERS"]
SKIP_LAYER = conf["SKIP_LAYER"]
HIDDEN_DIM = conf["HIDDEN_DIM"]
WITH_GCS = conf["WITH_GCS"]
H = conf["HEIGHT"]
W = conf["WIDTH"]

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
    checkpoint_dir = os.path.join(GCS_MODEL_DIR, f"tinynerf-complete-keras-{current_time}")
    visualization_dir = os.path.join(GCS_IMAGE_DIR, f"tinynerf-complete-keras-{current_time}")
else:
    checkpoint_dir = os.path.join(MODEL_DIR, f"tinynerf-complete-keras-{current_time}")


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

# Split the data into training and validation sets
train_images, val_images, train_poses, val_poses = split_data(images, poses, split_ratio=0.8)

# Make the train dataset pipelines
train_ds = create_tiny_dataset_pipeline(
    train_images,
    train_poses,
    H,
    W,
    focal,
    NS_COARSE,
    L_XYZ,
    L_DIR,
    BATCH_SIZE,
    AUTO,
    near=2.0,
    far=6.0,
    shuffle=True,
    rand_sampling=True,
)

# Make the validation dataset pipeline
val_ds = create_tiny_dataset_pipeline(
    val_images,
    val_poses,
    H,
    W,
    focal,
    NS_COARSE,
    L_XYZ,
    L_DIR,
    BATCH_SIZE,
    AUTO,
    near=2.0,
    far=6.0,
    shuffle=False, # Typically, validation data is not shuffled
    rand_sampling=False, # Or True, depending on if you want stochasticity here
)

train_imgs, train_rays = next(iter(train_ds))
train_ray_origins, train_ray_directions, train_rays_flat, train_dirs_flat, train_t_vals = train_rays

val_imgs, val_rays = next(iter(val_ds))
val_ray_origins, val_ray_directions, val_rays_flat, val_dirs_flat, val_t_vals = val_rays

# Create coarse and fine NeRF models
coarse_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR
)

fine_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR
)

nerf_trainer = NeRFTrainer(
    coarse_model=coarse_model,
    fine_model=fine_model,
    batch_size=BATCH_SIZE,
    ns_coarse=NS_COARSE,
    ns_fine=NS_FINE
)

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)
nerf_trainer.compile(
    optimizer=optimizer,
    loss_fn=keras.losses.MeanSquaredError(),
)


# Build nerf_trainer
images_shape = train_imgs.shape[1:]
ray_origins_shape = train_ray_origins.shape[1:]
ray_directions_shape = train_ray_directions.shape[1:]
rays_flat_shape = train_rays_flat.shape[1:]
dirs_flat_shape = train_dirs_flat.shape[1:]
t_vals_shape = train_t_vals.shape[1:]
rays_tuple_shape = (ray_origins_shape, ray_directions_shape, rays_flat_shape, dirs_flat_shape, t_vals_shape)
input_shape_for_build = (images_shape, rays_tuple_shape)
nerf_trainer.build(input_shape=input_shape_for_build)

class TrainCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Saves images at the end of each epoch."""
        print(f"TrainCallback: Epoch {epoch + 1} ended with logs: {logs}")

        loss_coarse = logs["loss_coarse"]
        loss = logs["loss"]
        psnr = logs["psnr"]

        rgbs, depths, weights, preds = self.model.forward_render(val_ray_origins, val_ray_directions, val_t_vals, H, W, L_XYZ, L_DIR, training=False)

        (rgbs_coarse, rgbs_fine) = rgbs
        (depths_coarse, depths_fine) = depths
        (preds_coarse, preds_fine) = preds

        print(f"preds_coarse ({ops.min(preds_coarse)}, {ops.max(preds_coarse)})")
        print(f"preds_fine ({ops.min(preds_fine)}, {ops.max(preds_fine)})")
        print(f"rgbs_coarse ({ops.min(rgbs_coarse)}, {ops.max(rgbs_coarse)})")
        print(f"rgbs_fine ({ops.min(rgbs_fine)}, {ops.max(rgbs_fine)})")
        print(f"depths_coarse ({ops.min(depths_coarse)}, {ops.max(depths_coarse)})")
        print(f"depths_fine ({ops.min(depths_fine)}, {ops.max(depths_fine)})")
        

nerf_trainer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=TrainCallback()
)

# rgbs, depths, weights = nerf_trainer.forward_render(val_ray_origins, val_ray_directions, val_t_vals, H, W, L_XYZ, L_DIR, training=False)
# rgbs_coarse, rgbs_fine = rgbs
# print(f"rgbs coarse ({ops.min(rgbs_coarse)}, {ops.max(rgbs_coarse)}): {rgbs_coarse[0, :3, :3]}")
# print(f"rgbs fine ({ops.min(rgbs_fine)}, {ops.max(rgbs_fine)}): {rgbs_fine[0, :3, :3]}")