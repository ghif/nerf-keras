import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Setting random seed to obtain reproducible results.
import tensorflow as tf

import keras

import io
import datetime
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from data_utils import split_data, create_dataset_pipeline
from models import create_nerf_model, NeRF, render_rgb_depth

# tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Add argument parser
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

print("Model Summary:")
print(nerf_model.summary(expand_nested=True))

model = NeRF(nerf_model, BATCH_SIZE, NUM_SAMPLES)
optimizer=keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE
)
model.compile(
    # optimizer=keras.optimizers.Adam(),
    optimizer=optimizer,
    loss_fn=keras.losses.MeanSquaredError(),
)

# Create a directory to save the images during training.
if not os.path.exists("images"):
    os.makedirs("images")

train_imgs, train_rays = next(iter(train_ds))
train_rays_flat, train_t_vals = train_rays

val_imgs, val_rays = next(iter(val_ds))
val_rays_flat, val_t_vals = val_rays

loss_list = []

class TrainCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Saves images at the end of each epoch."""
        print(f"TrainCallback: Epoch {epoch + 1} ended with logs: {logs}")
        loss = logs["loss"]
        loss_list.append(loss)

        test_recons_images, depth_maps = render_rgb_depth(
            self.model.nerf_model,
            val_rays_flat,
            val_t_vals,
            BATCH_SIZE,
            H,
            W,
            NUM_SAMPLES,
            rand=True,
            train=False,
        )

        # Save weights of self.model.nerf_model
        if WITH_GCS:
            if not tf.io.gfile.exists(checkpoint_dir):
                tf.io.gfile.makedirs(checkpoint_dir)

            print(f"Created GCS directory: {checkpoint_dir}")
            weight_path = tf.io.gfile.join(checkpoint_dir, f"nerf_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NUM_SAMPLES}_ep{EPOCHS}.weights.h5")
        else:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            print(f"Created Local directory: {checkpoint_dir}")
            weight_path = os.path.join(checkpoint_dir, f"nerf_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NUM_SAMPLES}_ep{EPOCHS}.weights.h5")

        self.model.nerf_model.save_weights(weight_path)

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        if WITH_GCS:
            if not tf.io.gfile.exists(visualization_dir):
                tf.io.gfile.makedirs(visualization_dir)
            img_path = tf.io.gfile.join(visualization_dir, f"{epoch:03d}.png")

            # Save figure to a BytesIO buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0) # Rewind buffer to the beginning
            # Write buffer to GCS
            with tf.io.gfile.GFile(img_path, 'wb') as f:
                f.write(buf.getvalue())
            print(f"Saved image to GCS: {img_path}")
            buf.close()

        else:
            img_dir = f"images/{checkpoint_dir}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            img_path = os.path.join(img_dir, f"{epoch:03d}.png")

            fig.savefig(img_path)
        # plt.show()
        # plt.close()


model.build(input_shape=(BATCH_SIZE, num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[TrainCallback()],
)