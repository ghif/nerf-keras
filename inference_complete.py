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

from data_utils import split_data, create_tiny_dataset_pipeline, get_rays, render_rays
from models import create_nerf_complete_model, NeRFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/tiny_nerf_complete_debug.json")
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

# Define nerf models
coarse_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR
)

print("Coarse Model Summary:")
print(coarse_model.summary(expand_nested=True))

fine_model = create_nerf_complete_model(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    skip_layer=SKIP_LAYER,
    lxyz=L_XYZ,
    ldir=L_DIR
)

print("Fine Model Summary:")
print(fine_model.summary(expand_nested=True))

nerf_trainer = NeRFTrainer(
    coarse_model=coarse_model,
    fine_model=fine_model,
    batch_size=BATCH_SIZE,
    ns_coarse=NS_COARSE,
    ns_fine=NS_FINE
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


# Load the model weights if they exist
weight_path = f"{MODEL_DIR}/tinynerf-complete-keras-20250616-best/nerf_complete_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NS_COARSE + NS_FINE}_ep{EPOCHS}.weights.h5"
if os.path.exists(weight_path):
    nerf_trainer.load_weights(weight_path)
    print("Model weights loaded successfully.")
else:
    print(f"Model weights not found at {weight_path}.")


val_rgbs, val_depths, val_weights, _ = nerf_trainer.forward_render(val_ray_origins, val_ray_directions, val_t_vals, H, W, L_XYZ, L_DIR, training=False)

val_rgb_coarse, val_rgb_fine = val_rgbs
val_depth_coarse, val_depth_fine = val_depths
val_weights_coarse, val_weights_fine = val_weights

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

for ax, ori_img, recons_img, depth_map in zip(
    axes, val_imgs, val_rgb_fine, val_depth_fine
):
    ax[0].imshow(keras.utils.array_to_img(ori_img))
    ax[0].set_title("Original Image")

    ax[1].imshow(keras.utils.array_to_img(recons_img))
    ax[1].set_title("Reconstructed Image")

    ax[2].imshow(keras.utils.array_to_img(depth_map[..., None]), cmap="inferno")
    ax[2].set_title("Depth Map")

plt.show()

def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return ops.convert_to_tensor(matrix, dtype="float32")


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, ops.cos(phi), -ops.sin(phi), 0],
        [0, ops.sin(phi), ops.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return ops.convert_to_tensor(matrix, dtype="float32")


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [ops.cos(theta), 0, -ops.sin(theta), 0],
        [0, 1, 0, 0],
        [ops.sin(theta), 0, ops.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return ops.convert_to_tensor(matrix, dtype="float32")


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

rgb_frames = []
batch_ray_oris = []
batch_ray_dirs = []
batch_t = []


# Iterate over different theta value and generate scenes.
for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
    # Get the camera to world matrix.
    c2w = pose_spherical(theta, -30.0, 4.0)

    ray_oris, ray_dirs = get_rays(H, W, focal, c2w)

    (rays_flat, dirs_flat, t_vals) = render_rays(
        ray_origins=ray_oris,
        ray_directions=ray_dirs,
        near=2.0,
        far=6.0,
        num_samples=NS_COARSE,
        l_xyz=L_XYZ,
        l_dir=L_DIR,
        rand=False,
    )

    if index % BATCH_SIZE == 0 and index > 0:
        batched_ray_oris = ops.stack(batch_ray_oris, axis=0)
        batch_ray_oris = [ray_oris]
        
        batched_ray_dirs = ops.stack(batch_ray_dirs, axis=0)
        batch_ray_dirs = [ray_dirs]

        batched_t = ops.stack(batch_t, axis=0)
        batch_t = [t_vals]

        # Render the RGB and depth maps using the nerf model
        rgbs, depth_maps, weights, _ = nerf_trainer.forward_render(batched_ray_oris, batched_ray_dirs, batched_t, H, W, L_XYZ, L_DIR, training=False)

        # Get the RGB from the fine model
        rgb_fine = rgbs[1]
        
        # Get the RGB frames from the rendered images
        temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb_fine]
        rgb_frames += temp_rgb
    else:
        # batch_flat.append(rays_flat)
        batch_ray_oris.append(ray_oris)
        batch_ray_dirs.append(ray_dirs)
        batch_t.append(t_vals)
        
rgb_video = "rgb_complete_video.mp4"
imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)