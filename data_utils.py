import keras
import tensorflow as tf
from keras import ops

import numpy as np

def encode_position(x, pos_encode_dims):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.
        pos_encode_dims: The number of dimensions for Fourier encoding.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(pos_encode_dims):
        for fn in [ops.sin, ops.cos]:
            positions.append(fn(2.0**i * x))
    return ops.concatenate(positions, axis=-1)

def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Get the pixel coordinates
    u, v = ops.meshgrid(
        ops.arange(width, dtype="float32"),
        ops.arange(height, dtype="float32"),
        indexing="xy",
    )
    transformed_u = (u - width * 0.5) / focal
    transformed_v = (v - height * 0.5) / focal
    directions = ops.stack(
        [transformed_u, -transformed_v, -ops.ones_like(u)], axis=-1
    )
    camera_matrix = pose[:3, :3]
    translations = pose[:3, -1]
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = ops.sum(camera_dirs, axis=-1)
    ray_origins = ops.broadcast_to(translations, ops.shape(ray_directions))
    return (ray_origins, ray_directions)


def sample_rays(ray_origins, ray_directions, t_vals):
    """Samples rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        t_vals: Sampled points on each ray.
        l_xyz: Dimensions for positional encoding of xyz coordinates.
        l_dir: Dimensions for positional encoding of direction vectors.

    Returns:
        Tuple of flattened rays and direction vectors.
    """
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., :, None]
    )
    dir_shape = ops.shape(rays[..., :3])
    dirs = ops.broadcast_to(ray_directions[..., None, :], dir_shape)
    return rays, dirs

def volume_render(preds, t_vals):
    # Get rgb and sigma from the predictions
    rgb = ops.sigmoid(preds[..., :-1])
    sigma_a = ops.relu(preds[..., -1])

    # Get the distance of adjacent intervals
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    const = ops.broadcast_to([1e10], shape=(delta.shape[0], 1))
    delta = ops.concatenate([delta, const], axis=-1)

    alpha = 1.0 - ops.exp(-sigma_a * delta)
    exp_term = 1.0 - alpha
    epsilon = 1e-10

    # Compute transmittance: Cumulative prod with exclusive mode
    tm = ops.cumprod(exp_term + epsilon, axis=-1)
    tm = ops.roll(tm, shift=1, axis=-1)
    transmittance = ops.concatenate([ops.ones((tm.shape[0], 1)), tm[:, 1:]], axis=-1)

    # Compute weights
    weights = alpha * transmittance
    rgb_w = ops.sum(weights[..., None] * rgb, axis=-2)    
    depth_map = ops.sum(weights * t_vals, axis=-1)
    return (rgb_w, depth_map, weights)

def split_data(images, poses, split_ratio=0.8):
    """Splits images and poses into training and validation sets.

    Args:
        images: Numpy array of images.
        poses: Numpy array of poses.
        split_ratio: Ratio for training data.

    Returns:
        Tuple of (train_images, val_images, train_poses, val_poses).
    """
    num_items = images.shape[0]
    split_index = int(num_items * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]
    return train_images, val_images, train_poses, val_poses

def generate_t_vals(near, far, batch_size, num_samples, rand_sampling=True):
    """Generates t-values for ray sampling.

    Args:
        near: Near plane for ray sampling.
        far: Far plane for ray sampling.
        num_samples: Number of samples per ray.
        rand_sampling: Boolean for randomizing sampling.

    Returns:
        Tensor of t-values.
    """
    t_vals = ops.linspace(near, far, num_samples, dtype="float32")
    if rand_sampling:
        noise = keras.random.uniform(shape=ops.shape(t_vals)) * (far - near) / num_samples
        t_vals = t_vals + noise
    
    t_vals = ops.broadcast_to(t_vals, (batch_size, num_samples))

    return t_vals

def create_batched_dataset_pipeline(
    images_s, 
    ray_oris_s, 
    ray_dirs_s, 
    num_samples,
    batch_size, 
    auto, 
    near=2.0, 
    far=6.0,
    shuffle=True,
    rand_sampling=True,
):
    # Image dataset
    img_ds = tf.data.Dataset.from_tensor_slices(images_s)

    # Ray dataset
    t_vals = generate_t_vals(near, far, ops.shape(ray_oris_s)[0], num_samples, rand_sampling)
    ray_ds = tf.data.Dataset.from_tensor_slices((ray_oris_s, ray_dirs_s, t_vals))

    dataset = tf.data.Dataset.zip((img_ds, ray_ds))
    
    if shuffle:
        # Consider using a larger buffer size for better shuffling if memory allows
        dataset = dataset.shuffle(buffer_size=batch_size * 5 if batch_size else 1024)

    dataset = (
        dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=auto)
        .prefetch(auto)
    )

    return dataset

def sample_pdf(t_vals_mid, weights, ns_fine):
    # Get batch_size, H, W
    batch_size = ops.shape(weights)[0]
    if len(ops.shape(weights)) == 4: # (b, h, w, num_samples)
        image_height, image_width = ops.shape(weights)[1:3]
    
    # add a small value to the weights to prevent it from nan
    weights += 1e-5
    
    # normalize the weights to get the pdf
    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    
    # from pdf to cdf transformation
    cdf = tf.cumsum(pdf, axis=-1)
    
    # start the cdf with 0sa
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)

    # get the sample points
    if len(ops.shape(weights)) == 4:
        u_shape = [batch_size, image_height, image_width, ns_fine]
    else:
        u_shape = [batch_size, ns_fine]

    u = tf.random.uniform(shape=u_shape)
    
    # get the indices of the points of u when u is inserted into cdf in a
    # sorted manner
    indices = tf.searchsorted(cdf, u, side="right")

    # define the boundaries
    below = tf.maximum(0, indices-1)
    above = tf.minimum(cdf.shape[-1]-1, indices)
    indices_g = tf.stack([below, above], axis=-1)
    
    # gather the cdf according to the indices
    cdf_g = tf.gather(cdf, indices_g, axis=-1, batch_dims=len(indices_g.shape)-2)

    # gather the tVals according to the indices
    indices_gt = tf.minimum(t_vals_mid.shape[-1] - 1, indices_g)
    t_vals_mid_g = tf.gather(t_vals_mid, indices_gt, axis=-1,
        batch_dims=len(indices_g.shape)-2)
    
    # create the samples by inverting the cdf
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = (t_vals_mid_g[..., 0] + t * 
        (t_vals_mid_g[..., 1] - t_vals_mid_g[..., 0]))

    # return the samples
    return samples

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