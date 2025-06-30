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

from data_utils import create_batched_dataset_pipeline, generate_t_vals
from fern_data_utils import prepare_fern_data
from models import NeRFTrainer, create_nerf_complete_model

# tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/fern_batch_debug.json")
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

# GCS bucket configuration
GCS_BUCKET_NAME = "keras-models"
GCS_MODEL_DIR = f"gs://{GCS_BUCKET_NAME}/nerf/models"
GCS_IMAGE_DIR = f"gs://{GCS_BUCKET_NAME}/nerf/images"

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if WITH_GCS:
    checkpoint_dir = os.path.join(GCS_MODEL_DIR, f"{config_filename}-{current_time}")
    visualization_dir = os.path.join(GCS_IMAGE_DIR, f"{config_filename}-{current_time}")
else:
    checkpoint_dir = os.path.join(MODEL_DIR, f"{config_filename}-{current_time}")

# Load Fern dataset
(train_data, val_data, bounds) = prepare_fern_data(H, W, from_gcs=WITH_GCS)
(train_images_s, train_ray_oris_s, train_ray_dirs_s) = train_data
(val_images_s, val_ray_oris_s, val_ray_dirs_s) = val_data
(near, far) = bounds

# # Plot a random image
# nb = int(ops.shape(train_images_s)[0] / (H * W))
# train_imgs = ops.reshape(train_images_s, (nb, H, W, 3))


# plt.imshow(train_imgs[np.random.randint(0, nb)])
# plt.show()

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

nerf_trainer = NeRFTrainer(
    coarse_model=coarse_model,
    fine_model=fine_model,
    batch_size=BATCH_SIZE,
    ns_coarse=NS_COARSE,
    ns_fine=NS_FINE,
    l_xyz=L_XYZ,
    l_dir=L_DIR,
)

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)
nerf_trainer.compile(
    optimizer=optimizer,
    loss_fn=keras.losses.MeanSquaredError(),
)

# Create a directory to save the images during training.
if not os.path.exists("images"):
    os.makedirs("images")

loss_coarse_list = []
loss_list = []
psnr_list = []
history = {"losses": [], "psnrs": []}

class TrainCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Saves images at the end of each epoch."""
        print(f"TrainCallback: Epoch {epoch + 1} ended with logs: {logs}")

        loss_coarse = logs["loss_coarse"]
        loss = logs["loss"]
        psnr = logs["psnr"]

        loss_coarse_list.append(loss_coarse)
        loss_list.append(loss)
        psnr_list.append(psnr)

        history["losses_coarse"] = loss_coarse_list
        history["losses"] = loss_list
        history["psnrs"] = psnr_list

        # Predict with volume rendering
        nsample = 1 * H * W
        val_ray_ori_samples = val_ray_oris_s[:nsample]
        val_ray_dir_samples = val_ray_dirs_s[:nsample]

        t_vals = generate_t_vals(near, far, ops.shape(val_ray_ori_samples)[0], NS_COARSE, rand_sampling=True)
        rgbs, depths, _, _ = self.model.forward_pass(val_ray_ori_samples, val_ray_dir_samples, t_vals, L_XYZ, L_DIR, training=False)

        (_, test_recons_images) = rgbs
        (_, depth_maps) = depths

        # Reshape the test_recons_images and depth_maps to (nb, H, W, 3) and (nb, H, W) respectively.
        nb = int(ops.shape(test_recons_images)[0] / (H * W))
        test_recons_images = ops.reshape(test_recons_images, (nb, H, W, 3))
        depth_maps = ops.reshape(depth_maps, (nb, H, W))
        
        # Save weights of self.model.nerf_model
        weight_file_prefix = f"nerf_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NS_COARSE + NS_FINE}_ep{EPOCHS}"
        if WITH_GCS:
            if not tf.io.gfile.exists(checkpoint_dir):
                tf.io.gfile.makedirs(checkpoint_dir)

            print(f"Created GCS directory: {checkpoint_dir}")
            weight_path = tf.io.gfile.join(checkpoint_dir, f"{weight_file_prefix}.weights.h5")
        else:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            print(f"Created Local directory: {checkpoint_dir}")
            weight_path = os.path.join(checkpoint_dir, f"{weight_file_prefix}.weights.h5")

        self.model.save_weights(weight_path)

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

            # Save history to a JSON file
            history_path = tf.io.gfile.join(checkpoint_dir, f"history_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NS_COARSE + NS_FINE}_ep{EPOCHS}.json")
            try:
                history_json_string = json.dumps(history)
                with tf.io.gfile.GFile(history_path, 'w') as f_json:
                    f_json.write(history_json_string)
            except Exception as e:
                print(f"Error saving history to GCS: {e}")
            

        else:
            img_dir = f"images/{checkpoint_dir}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            img_path = os.path.join(img_dir, f"{epoch:03d}.png")

            fig.savefig(img_path)

            # Save history to a JSON file
            history_path = os.path.join(checkpoint_dir, f"history_l{NUM_LAYERS}_d{HIDDEN_DIM}_n{NS_COARSE +NS_FINE}_ep{EPOCHS}.json")
            with open(history_path, 'w') as f:
                json.dump(history, f)

        # plt.show()
        # plt.close()


# Build nerf_trainer
images_shape = train_imgs.shape[1:]
ray_origins_shape = train_ray_origins.shape[1:]
ray_directions_shape = train_ray_directions.shape[1:]
t_vals_shape = train_t_vals.shape[1:]
rays_tuple_shape = (ray_origins_shape, ray_directions_shape, t_vals_shape)
input_shape_for_build = (images_shape, rays_tuple_shape)
nerf_trainer.build(input_shape=input_shape_for_build)

nerf_trainer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[TrainCallback()],
)