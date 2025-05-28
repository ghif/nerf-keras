import os

os.environ["KERAS_BACKEND"] = "tensorflow"

# Setting random seed to obtain reproducible results.
import tensorflow as tf

import keras
from keras import layers

import os
import datetime
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import split_data, create_dataset_pipeline
from models import create_nerf_model, NeRF, render_rgb_depth

# tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 100
MODEL_DIR = "models"
LEARNING_RATE = 1e-4

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(MODEL_DIR, f"tinynerf-keras-{current_time}")

# Create the model directory if it does not exist.
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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
    num_layers=8,
    hidden_dim=128,
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

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.utils.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.utils.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(loss_list)
        ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        img_dir = f"images/{checkpoint_dir}"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        fig.savefig(f"{img_dir}/{epoch:03d}.png")
        # plt.show()
        # plt.close()


checkpoint_path = os.path.join(checkpoint_dir, "tinynerf_model.weights.h5")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq="epoch",
    save_weights_only=True,
)


model.build(input_shape=(BATCH_SIZE, num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[TrainCallback(), checkpoint_callback],
)