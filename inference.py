import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops
import tensorflow as tf
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio.v2 as imageio

from data_utils import split_data, create_dataset_pipeline, get_rays, render_flat_rays
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

GCS_BUCKET_NAME = "keras-models"
GCS_MODEL_DIR = f"gs://{GCS_BUCKET_NAME}/nerf/models"


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
weight_path = f"{MODEL_DIR}/tinynerf-keras-20250601-043239/nerf_l{NUM_LAYERS}_d{HIDDEN_DIM}.weights.h5"
nerf_model.load_weights(weight_path)
print("Model weights loaded successfully.")

train_imgs, train_rays = next(iter(train_ds))
train_rays_flat, train_t_vals = train_rays

val_imgs, val_rays = next(iter(val_ds))
val_rays_flat, val_t_vals = val_rays

test_recons_images, depth_maps = render_rgb_depth(
    model=nerf_model,
    rays_flat=val_rays_flat,
    t_vals=val_t_vals,
    batch_size=BATCH_SIZE,
    h=H,
    w=W,
    num_samples=NUM_SAMPLES,
    rand=True,
    train=False
)

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

for ax, ori_img, recons_img, depth_map in zip(
    axes, val_imgs, test_recons_images, depth_maps
):
    ax[0].imshow(keras.utils.array_to_img(ori_img))
    ax[0].set_title("Original Image")

    ax[1].imshow(keras.utils.array_to_img(recons_img))
    ax[1].set_title("Reconstructed Image")

    ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]), cmap="inferno")
    ax[2].set_title("Depth Map")

plt.show()


# def get_translation_t(t):
#     """Get the translation matrix for movement in t."""
#     matrix = [
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, t],
#         [0, 0, 0, 1],
#     ]
#     return ops.convert_to_tensor(matrix, dtype="float32")


# def get_rotation_phi(phi):
#     """Get the rotation matrix for movement in phi."""
#     matrix = [
#         [1, 0, 0, 0],
#         [0, ops.cos(phi), -ops.sin(phi), 0],
#         [0, ops.sin(phi), ops.cos(phi), 0],
#         [0, 0, 0, 1],
#     ]
#     return ops.convert_to_tensor(matrix, dtype="float32")


# def get_rotation_theta(theta):
#     """Get the rotation matrix for movement in theta."""
#     matrix = [
#         [ops.cos(theta), 0, -ops.sin(theta), 0],
#         [0, 1, 0, 0],
#         [ops.sin(theta), 0, ops.cos(theta), 0],
#         [0, 0, 0, 1],
#     ]
#     return ops.convert_to_tensor(matrix, dtype="float32")


# def pose_spherical(theta, phi, t):
#     """
#     Get the camera to world matrix for the corresponding theta, phi
#     and t.
#     """
#     c2w = get_translation_t(t)
#     c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
#     c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
#     c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
#     return c2w

# rgb_frames = []
# batch_flat = []
# batch_t = []

# # Iterate over different theta value and generate scenes.
# for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
#     # Get the camera to world matrix.
#     c2w = pose_spherical(theta, -30.0, 4.0)

#     ray_oris, ray_dirs = get_rays(H, W, focal, c2w)

#     rays_flat, t_vals = render_flat_rays(
#         ray_oris, 
#         ray_dirs, 
#         near=2.0, 
#         far=6.0, 
#         num_samples=NUM_SAMPLES, 
#         pos_encode_dims=POS_ENCODE_DIMS,
#         rand=False
#     )

#     if index % BATCH_SIZE == 0 and index > 0:
#         batched_flat = ops.stack(batch_flat, axis=0)
#         batch_flat = [rays_flat]

#         batched_t = ops.stack(batch_t, axis=0)
#         batch_t = [t_vals]

#         rgb, _ = render_rgb_depth(
#             model=nerf_model,
#             rays_flat=batched_flat,
#             t_vals=batched_t,
#             batch_size=BATCH_SIZE,
#             h=H,
#             w=W,
#             num_samples=NUM_SAMPLES,
#             rand=True,
#             train=False
#         )

#         temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb]

#         rgb_frames += temp_rgb
#     else:
#         batch_flat.append(rays_flat)
#         batch_t.append(t_vals)


# rgb_video = "rgb_video.mp4"
# imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)