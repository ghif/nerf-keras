import numpy as np
import keras
from keras import ops
import tensorflow as tf

from data_utils import get_rays, split_data

def prepare_lego_data(target_height, target_width):
    # Load Lego dataset
    # Download the dataset if it does not exist.
    url = (
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    )
    data = keras.utils.get_file(origin=url)

    # Load the dataset
    data = np.load(data)
    images = data["images"]
    (poses, focal) = (data["poses"], data["focal"])

    # Prepare lego data
    # Resize images to (H, W)
    images_r = tf.image.resize(images, (target_height, target_width))

    # Split the data into training and validation sets
    train_images, val_images, train_poses, val_poses = split_data(images_r, poses, split_ratio=0.8)

    # Convert poses to rays
    train_rays = [get_rays(target_height, target_width, focal, pose) for pose in train_poses]
    train_rays = ops.stack(train_rays, axis=0).numpy()
    train_ray_oris = train_rays[:, 0, ...]
    train_ray_dirs = train_rays[:, 1, ...]

    val_rays = [get_rays(target_height, target_height, focal, pose) for pose in val_poses]
    val_rays = ops.stack(val_rays, axis=0).numpy()
    val_ray_oris = val_rays[:, 0, ...]
    val_ray_dirs = val_rays[:, 1, ...]

    train_images_s = ops.reshape(train_images, [-1, train_images.shape[-1]])
    val_images_s = ops.reshape(val_images, [-1, val_images.shape[-1]])

    train_ray_oris_s = ops.reshape(train_ray_oris, [-1, train_ray_oris.shape[-1]])
    val_ray_oris_s = ops.reshape(val_ray_oris, [-1, val_ray_oris.shape[-1]])

    train_ray_dirs_s = ops.reshape(train_ray_dirs, [-1, train_ray_dirs.shape[-1]])
    val_ray_dirs_s = ops.reshape(val_ray_dirs, [-1, val_ray_dirs.shape[-1]])

    near = 2.0
    far = 6.0

    return (train_images_s, train_ray_oris_s, train_ray_dirs_s), (val_images_s, val_ray_oris_s, val_ray_dirs_s), (near, far), focal
    

