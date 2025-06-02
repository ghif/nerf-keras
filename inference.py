import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

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

# # Test ReLU directly
# relu_layer_test = keras.layers.ReLU()
# test_input_neg = tf.constant([-3.0, -0.5, 0.0, 0.5, 3.0])
# test_output_neg = relu_layer_test(test_input_neg)
# print(f"Direct ReLU test input: {test_input_neg.numpy()}")
# print(f"Direct ReLU test output: {test_output_neg.numpy()}") # Expected: [0. 0. 0. 0.5 3.]

# # Test ReLU in a minimal Sequential model
# simple_model = keras.Sequential([
#     keras.Input(shape=(1,)), # Expects input shape (batch_size, 1)
#     # keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros'),
#     keras.layers.Dense(1),
#     keras.layers.ReLU()
# ])
# # Prepare input with correct rank for Dense layer
# test_input_for_model = np.array([[-3.0], [-0.5], [0.0], [0.5], [3.0]], dtype=np.float32)
# simple_output = simple_model.predict(test_input_for_model)
# print(f"Simple model (Dense -> ReLU) input: {test_input_for_model.flatten()}")
# print(f"Simple model (Dense -> ReLU) output: {simple_output.flatten()}") # Expected: [0. 0. 0. 0.5 3.]

# mini_model_prerelu = keras.Model(nerf_model.input, nerf_model.layers[1].output)
# print(mini_model_prerelu.summary(expand_nested=True))
# h1 = mini_model_prerelu(val_rays_flat)
# print(f"h1[0]: {h1[0]}")

# mini_model = keras.Model(nerf_model.input, nerf_model.layers[2].output)
# print(mini_model.summary(expand_nested=True))
# h2 = mini_model(val_rays_flat.numpy())
# print(f"h2[0]: {h2[0]}")

# print(f"h1[0] == h2[0]: {np.allclose(h1[0], h2[0])}") 

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