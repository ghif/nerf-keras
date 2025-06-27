import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Setting random seed to obtain reproducible results.
import tensorflow as tf

import keras
from keras import ops

import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
import imageio.v2 as imageio

from data_utils import create_batched_dataset_pipeline, generate_t_vals, pose_spherical, get_rays
from lego_data_utils import prepare_lego_data
from fern_data_utils import prepare_fern_data
from models import NeRFBatchTrainer, create_nerf_complete_model

# tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/lego_batch_h256.json")
# parser.add_argument("--config", type=str, default="config/fern_batch_h256.json")
args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

# Get config filename
config_filename = os.path.splitext(os.path.basename(args.config))[0]

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
BATCH_NORM = conf["BATCH_NORM"]

AUTO = tf.data.AUTOTUNE

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# Load Lego dataset
(train_data, val_data, bounds, focal) = prepare_lego_data(H, W)
# (train_data, val_data, bounds) = prepare_fern_data(H, W)
(train_images_s, train_ray_oris_s, train_ray_dirs_s) = train_data
(val_images_s, val_ray_oris_s, val_ray_dirs_s) = val_data
(near, far) = bounds

train_ds = create_batched_dataset_pipeline(
    train_images_s, 
    train_ray_oris_s, 
    train_ray_dirs_s,
    NS_COARSE,
    BATCH_SIZE, 
    AUTO, 
    near=near, 
    far=far,
    shuffle=True,
    rand_sampling=True,
)

val_ds = create_batched_dataset_pipeline(
    val_images_s,
    val_ray_oris_s,
    val_ray_dirs_s,
    NS_COARSE,
    BATCH_SIZE,
    AUTO,
    near=near,
    far=far,
    shuffle=False,
    rand_sampling=True,
)
train_imgs, train_rays =  next(iter(train_ds))
train_ray_origins, train_ray_directions, train_t_vals = train_rays

val_imgs, val_rays = next(iter(val_ds))
val_ray_origins, val_ray_directions, val_t_vals = val_rays

# Create coarse and fine NeRF models
coarse_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR,
    bn=BATCH_NORM
)

print(f"Coarse Model Summary:")
print(coarse_model.summary(expand_nested=True))

fine_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR,
    bn=BATCH_NORM
)

print(f"Fine Model Summary:")
print(fine_model.summary(expand_nested=True))

nerf_trainer = NeRFBatchTrainer(
    coarse_model=coarse_model,
    fine_model=fine_model,
    batch_size=BATCH_SIZE,
    ns_coarse=NS_COARSE,
    ns_fine=NS_FINE,
    l_xyz=L_XYZ,
    l_dir=L_DIR,
)

# Build nerf_trainer
images_shape = train_imgs.shape[1:]
ray_origins_shape = train_ray_origins.shape[1:]
ray_directions_shape = train_ray_directions.shape[1:]
t_vals_shape = train_t_vals.shape[1:]
rays_tuple_shape = (ray_origins_shape, ray_directions_shape, t_vals_shape)
input_shape_for_build = (images_shape, rays_tuple_shape)
nerf_trainer.build(input_shape=input_shape_for_build)

# Load the model weights if they exist
weight_path = f"{MODEL_DIR}/{config_filename}-best/nerf_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NS_COARSE + NS_FINE}_ep{EPOCHS}.weights.h5"
if os.path.exists(weight_path):
    nerf_trainer.load_weights(weight_path)
    print("Model weights loaded successfully.")
else:
    print(f"Model weights not found at {weight_path}.")

# Get first 5 samples from the validation dataset
n_images = 3
n_samples = n_images * H * W

val_image_samples = val_images_s[:n_samples]
val_ray_ori_samples = val_ray_oris_s[:n_samples]
val_ray_dir_samples = val_ray_dirs_s[:n_samples]

t_vals = generate_t_vals(near, far, ops.shape(val_ray_ori_samples)[0], NS_COARSE, rand_sampling=False)

val_rgbs, val_depths, val_weights, _ = nerf_trainer.forward_pass(val_ray_ori_samples, val_ray_dir_samples, t_vals, L_XYZ, L_DIR, training=False)

val_rgb_coarse, val_rgb_fine = val_rgbs
val_depth_coarse, val_depth_fine = val_depths
val_weights_coarse, val_weights_fine = val_weights

# Reshape the test_recons_images and depth_maps to (nb, H, W, 3) and (nb, H, W) respectively.
nb = int(ops.shape(val_image_samples)[0] / (H * W))
ori_imgs = ops.reshape(val_image_samples, (nb, H, W, 3))
recons_imgs = ops.reshape(val_rgb_fine, (nb, H, W, 3))
depth_maps = ops.reshape(val_depth_fine, (nb, H, W))

# Create subplots
fig, axes = plt.subplots(nrows=n_images, ncols=3, figsize=(10, 20))

counter = 0
for ax, ori_img, recons_img, depth_map in zip(
    axes, ori_imgs, recons_imgs, depth_maps
):
    ax[0].imshow(keras.utils.array_to_img(ori_img))
    if counter == 0:
        ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(keras.utils.array_to_img(recons_img))
    if counter == 0:
        ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    # ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]), cmap="inferno")
    ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]))
    if counter == 0:
        ax[2].set_title("Depth Map")
    ax[2].axis("off")

    counter += 1
plt.show()

rgb_frames = []
batch_ray_oris = []
batch_ray_dirs = []

# Iterate over different theta value and generate scenes.
for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
    # Get the camera to world matrix.
    c2w = pose_spherical(theta, -30.0, 4.0)

    ray_oris, ray_dirs = get_rays(H, W, focal, c2w)
    # print(f"Shape of ray_oris: {ray_oris.shape}, ray_dirs: {ray_dirs.shape}")

    if index % 5 == 0 and index > 0:
        batched_ray_oris = ops.stack(batch_ray_oris, axis=0)
        batch_ray_oris = [ray_oris]
        
        batched_ray_dirs = ops.stack(batch_ray_dirs, axis=0)
        batch_ray_dirs = [ray_dirs]

        # Render the RGB and depth maps using the nerf model
        ray_ori_s = ops.reshape(batched_ray_oris, (-1, batched_ray_oris.shape[-1]))
        ray_dir_s = ops.reshape(batched_ray_dirs, (-1, batched_ray_dirs.shape[-1]))
        t_vals_s = generate_t_vals(near, far, ops.shape(ray_ori_s)[0], NS_COARSE, rand_sampling=False)

        rgbs, depth_maps, weights, _ = nerf_trainer.forward_pass(ray_ori_s, ray_dir_s, t_vals_s, L_XYZ, L_DIR, training=False)

        # Get the RGB from the fine model
        rgb_fine = rgbs[1]

        # Reshape rgb_fine to (nb, H, W, 3)
        nb = int(ops.shape(rgb_fine)[0] / (H * W))
        rgb_fine = ops.reshape(rgb_fine, (nb, H, W, 3))
        
        # Get the RGB frames from the rendered images
        temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb_fine]
        rgb_frames += temp_rgb
    else:
        # batch_flat.append(rays_flat)
        batch_ray_oris.append(ray_oris)
        batch_ray_dirs.append(ray_dirs)
        
rgb_video = f"{config_filename}_rgb_video_v2.mp4"
imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=9, macro_block_size=None)