import keras
import tensorflow as tf

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
        for fn in [keras.ops.sin, keras.ops.cos]:
            positions.append(fn(2.0**i * x))
    return keras.ops.concatenate(positions, axis=-1)

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
    i, j = keras.ops.meshgrid(
        keras.ops.arange(width, dtype="float32"),
        keras.ops.arange(height, dtype="float32"),
        indexing="xy",
    )
    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal
    directions = keras.ops.stack(
        [transformed_i, -transformed_j, -keras.ops.ones_like(i)], axis=-1
    )
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = keras.ops.sum(camera_dirs, axis=-1)
    ray_origins = keras.ops.broadcast_to(height_width_focal, keras.ops.shape(ray_directions))
    return (ray_origins, ray_directions)

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
    t_vals = keras.ops.linspace(near, far, num_samples)
    shape = list(ray_origins.shape[:-1]) + [num_samples]
    if rand:
        noise = keras.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise
    else:
        t_vals = keras.ops.broadcast_to(t_vals, shape)

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = keras.ops.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat, pos_encode_dims)
    return (rays_flat, t_vals)

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