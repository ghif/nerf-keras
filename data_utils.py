import keras
import tensorflow as tf
from keras import ops

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
    i, j = ops.meshgrid(
        ops.arange(width, dtype="float32"),
        ops.arange(height, dtype="float32"),
        indexing="xy",
    )
    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal
    directions = ops.stack(
        [transformed_i, -transformed_j, -ops.ones_like(i)], axis=-1
    )
    camera_matrix = pose[:3, :3]
    translations = pose[:3, -1]
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = ops.sum(camera_dirs, axis=-1)
    ray_origins = ops.broadcast_to(translations, ops.shape(ray_directions))
    return (ray_origins, ray_directions)

def encode_and_flatten(samples, pos_encode_dims):
    samples_enc = encode_position(samples, pos_encode_dims)
    print(f"[encode_and_flatten] samples shape: {ops.shape(samples_enc)}")
    samples_flat = ops.reshape(samples_enc, [-1, samples_enc.shape[-1] * samples_enc.shape[-2]])
    print(f"[encode_and_flatten] samples_flat shape: {ops.shape(samples_flat)}")
    return samples_flat

def flatten_and_encode(samples, pos_encode_dims):
    sample_shape = ops.shape(samples)
    if len(sample_shape) > 4:
        batch_size = sample_shape[0]
        samples_flat = ops.reshape(samples, [batch_size, -1, 3])
    else:
        samples_flat = ops.reshape(samples, [-1, 3])

    samples_flat_encoded = encode_position(samples_flat, pos_encode_dims)
    return samples_flat_encoded

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

def sample_rays_flat(ray_origins, ray_directions, t_vals, l_xyz, l_dir):
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    dir_shape = ops.shape(rays[..., :3])
    dirs = ops.broadcast_to(ray_directions[..., None, :], dir_shape)
    rays_flat = flatten_and_encode(rays, pos_encode_dims=l_xyz)
    dirs_flat = flatten_and_encode(dirs, pos_encode_dims=l_dir)
    return rays_flat, dirs_flat

def render_rays(
    ray_origins, ray_directions, near, far, num_samples, l_xyz, l_dir, rand=False
):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        l_xyz: Dimensions for positional encoding of xyz coordinates.
        l_dir: Dimensions for positional encoding of direction vectors.
        rand: Choice for randomising the sampling strategy.
    Returns:
        Tuple of flattened rays, direction vectors and sample points on each rays.
    """
    t_vals = ops.linspace(near, far, num_samples, dtype="float32")
    shape = list(ray_origins.shape[:-1]) + [num_samples]
    if rand:
        noise = keras.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise
    else:
        t_vals = ops.broadcast_to(t_vals, shape)
    
    rays_flat, dirs_flat = sample_rays_flat(
        ray_origins, 
        ray_directions, 
        t_vals, 
        l_xyz,
        l_dir
    )
    
    return rays_flat, dirs_flat, t_vals


def render_flat_rays(
    ray_origins, ray_directions, near, far, num_samples, pos_encode_dims, rand=False
):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        pos_encode_dims: The number of dimensions for Fourier encoding.
        rand: Choice for randomising the sampling strategy.

    Returns:
       Tuple of flattened rays and sample points on each rays.
    """
    t_vals = ops.linspace(near, far, num_samples)
    shape = list(ray_origins.shape[:-1]) + [num_samples]
    if rand:
        noise = keras.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise
    else:
        t_vals = ops.broadcast_to(t_vals, shape)

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = ops.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat, pos_encode_dims)
    return (rays_flat, t_vals)

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

def render_predictions(predictions, t_vals, rand=True):
    """Volume rendering: Generates the RGB image and depth map from the model predictions.

    Args:
        predictions: Model predictions of shape (batch_size, h, w, num_samples, 4).
        t_vals: Sampled points on each ray of shape (batch_size, num_samples).
        rand: Boolean indicating whether to use random sampling.

    Returns:
        Tuple of rgb image and depth map.
    """
    # Get batch size, h , w
    batch_size, h, w = ops.shape(predictions)[:3]

    # Slice the predictions into rgb and sigma.
    rgb = ops.sigmoid(predictions[..., :-1])
    sigma_a = ops.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    if rand:
        delta = ops.concatenate(
            [delta, ops.broadcast_to([1e10], shape=(batch_size, h, w, 1))], axis=-1
        )
        alpha = 1.0 - ops.exp(-sigma_a * delta)
    else:
        delta = ops.concatenate(
            [delta, ops.broadcast_to([1e10], shape=(batch_size, 1))], axis=-1
        )
        alpha = 1.0 - ops.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    # transmittance = ops.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = ops.sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = ops.sum(weights * t_vals, axis=-1)
    else:
        depth_map = ops.sum(weights * t_vals[:, None, None], axis=-1)

    return (rgb, depth_map, weights)

def create_preprocess_image_fn(target_height, target_width):
    """Creates a preprocessing function for images.

    Args:
        target_height: Target height for resizing images.
        target_width: Target width for resizing images.

    Returns:
        A function that preprocesses an image by resizing it.
    """
    def preprocess_image(image):
        image = tf.image.resize(image, [target_height, target_width])
        return image
    return preprocess_image

# def create_ray_fn(num_samples, l_xyz, l_dir, near=2.0, far=6.0, rand=True):
#     def ray_fn(ray_ori_s ,ray_dir_s):
#         t_vals = ops.linspace(near, far, num_samples, dtype="float32")
#         # ray, dir = sample_rays(ray_ori_s, ray_dir_s, t_vals) 
#         return (ray, dir, t_vals)
#     return ray_fn

def create_preprocess_ray_fn(H, W, focal, num_samples, l_xyz, l_dir, near=2.0, far=6.0, rand=True):
    """Creates a preprocessing function for rays.

    Args:
        H: Height of the image.
        W: Width of the image.
        focal: Focal length.
        num_samples: Number of samples per ray.
        l_xyz: Dimensions for positional encoding of xyz coordinates.
        l_dir: Dimensions for positional encoding of direction vectors.
        near: Near plane for ray sampling. (default: 2.0)
        far: Far plane for ray sampling. (default: 6.0)
        rand: Boolean for randomizing sampling. (default: True)
    Returns:
        A function that processes a camera pose to get rays and t_vals.
    """
    def preprocess_rays_fn(pose):
        (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
        (rays_flat, dirs_flat, t_vals) = render_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            near=near,
            far=far,
            num_samples=num_samples,
            l_xyz=l_xyz,
            l_dir=l_dir,
            rand=rand,
        )
        return (ray_origins, ray_directions, rays_flat, dirs_flat, t_vals)
    return preprocess_rays_fn

def create_map_fn(H, W, focal, num_samples, pos_encode_dims, near, far, rand):
    """Creates a mapping function for dataset processing.

    Args:
        H: Height of the image.
        W: Width of the image.
        focal: Focal length.
        num_samples: Number of samples per ray.
        pos_encode_dims: Dimensions for positional encoding.
        near: Near plane for ray sampling.
        far: Far plane for ray sampling.
        rand: Boolean for randomizing sampling.

    Returns:
        A function that maps a camera pose to rays and t_vals.
    """
    def map_fn(pose):
        (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
        (rays_flat, t_vals) = render_flat_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            near=near,
            far=far,
            num_samples=num_samples,
            pos_encode_dims=pos_encode_dims,
            rand=rand,
        )
        return (rays_flat, t_vals)
    return map_fn

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
    # t_vals = ops.linspace(near, far, num_samples, dtype="float32")
    # if rand_sampling:
    #     noise = keras.random.uniform(shape=ops.shape(t_vals)) * (far - near) / num_samples
    #     t_vals = t_vals + noise
    
    # t_vals = ops.broadcast_to(t_vals, ops.shape(ray_oris_s)[:-1] + (num_samples, ))

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

def create_complete_dataset_pipeline(
    images_data,
    poses_data,
    H,
    W,
    focal,
    num_samples,
    l_xyz,
    l_dir,
    batch_size,
    auto,
    near=2.0,
    far=6.0,
    shuffle=True,
    rand_sampling=True,
):
    """Creates a TensorFlow dataset pipeline for NeRF.

    Args:
        images_data: Numpy array of images for the dataset.
        poses_data: Numpy array of poses for the dataset.
        H: Height of the image.
        W: Width of the image.
        focal: Focal length.
        num_samples: Number of samples per ray.
        pos_encode_dims: Dimensions for positional encoding.
        batch_size: Batch size for the dataset.
        auto: tf.data.AUTOTUNE.
        near: Near plane for ray sampling. (default: 2.0)
        far: Far plane for ray sampling. (default: 6.0)
        shuffle: Boolean to shuffle the dataset. (default: True)
        rand_sampling: Boolean for randomizing sampling in render_flat_rays. (default: True)

    Returns:
        A tf.data.Dataset object.
    """
    img_ds = tf.data.Dataset.from_tensor_slices(images_data)
    pose_ds = tf.data.Dataset.from_tensor_slices(poses_data)

    # Preprocess images
    image_fn_instance = create_preprocess_image_fn(H, W)
    img_ds = img_ds.map(image_fn_instance, num_parallel_calls=auto)

    # Create the mapping function for rays and t_vals
    ray_fn_instance = create_preprocess_ray_fn(
        H, W, focal, num_samples, l_xyz, l_dir, near, far, rand_sampling
    )
    ray_ds = pose_ds.map(ray_fn_instance, num_parallel_calls=auto)

    dataset = tf.data.Dataset.zip((img_ds, ray_ds))

    if shuffle:
        # Consider using a larger buffer size for better shuffling if memory allows
        dataset = dataset.shuffle(buffer_size=batch_size * 5 if batch_size else 1024)

    dataset = (
        dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=auto)
        .prefetch(auto)
    )
    return dataset

def create_dataset_pipeline(
    images_data,
    poses_data,
    H,
    W,
    focal,
    num_samples,
    pos_encode_dims,
    batch_size,
    auto,
    near=2.0,
    far=6.0,
    shuffle=True,
    rand_sampling=True,
):
    """Creates a TensorFlow dataset pipeline for NeRF.

    Args:
        images_data: Numpy array of images for the dataset.
        poses_data: Numpy array of poses for the dataset.
        H: Height of the image.
        W: Width of the image.
        focal: Focal length.
        num_samples: Number of samples per ray.
        pos_encode_dims: Dimensions for positional encoding.
        batch_size: Batch size for the dataset.
        auto: tf.data.AUTOTUNE.
        near: Near plane for ray sampling. (default: 2.0)
        far: Far plane for ray sampling. (default: 6.0)
        shuffle: Boolean to shuffle the dataset. (default: True)
        rand_sampling: Boolean for randomizing sampling in render_flat_rays. (default: True)

    Returns:
        A tf.data.Dataset object.
    """
    img_ds = tf.data.Dataset.from_tensor_slices(images_data)
    pose_ds = tf.data.Dataset.from_tensor_slices(poses_data)

    # Preprocess images
    preprocess_fn_instance = create_preprocess_image_fn(H, W)
    img_ds = img_ds.map(preprocess_fn_instance, num_parallel_calls=auto)

    # Create the mapping function for rays and t_vals
    map_fn_instance = create_map_fn(
        H, W, focal, num_samples, pos_encode_dims, near, far, rand_sampling
    )
    ray_ds = pose_ds.map(map_fn_instance, num_parallel_calls=auto)

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
    # start the cdf with 0s
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
    cdf_g = tf.gather(cdf, indices_g, axis=-1,
        batch_dims=len(indices_g.shape)-2)

    # gather the tVals according to the indices
    t_vals_mid_g = tf.gather(t_vals_mid, indices_g, axis=-1,
        batch_dims=len(indices_g.shape)-2)
    # create the samples by inverting the cdf
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = (t_vals_mid_g[..., 0] + t * 
        (t_vals_mid_g[..., 1] - t_vals_mid_g[..., 0]))

    # return the samples
    return samples