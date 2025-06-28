from keras import ops
import keras
from keras import layers

import tensorflow as tf

def get_rays(height, width, focal, pose):
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
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., :, None]
    )
    dir_shape = ops.shape(rays[..., :3])
    dirs = ops.broadcast_to(ray_directions[..., None, :], dir_shape)
    return rays, dirs

def generate_t_vals(near, far, batch_size, num_samples, rand_sampling=True):
    t_vals = ops.linspace(near, far, num_samples, dtype="float32")
    if rand_sampling:
        noise = keras.random.uniform(shape=ops.shape(t_vals)) * (far - near) / num_samples
        t_vals = t_vals + noise
    
    t_vals = ops.broadcast_to(t_vals, (batch_size, num_samples))

    return t_vals


def encode_position(x, pos_encode_dims):
    positions = [x]
    for i in range(pos_encode_dims):
        for fn in [ops.sin, ops.cos]:
            positions.append(fn(2.0**i * x))
    return ops.concatenate(positions, axis=-1)

def create_nerf_complete_model(num_layers, hidden_dim, skip_layer, lxyz, ldir, bn=False):
    ray_input = keras.Input(shape=(None, 2 * 3 * lxyz + 3))
    dir_input = keras.Input(shape=(None, 2 * 3 * ldir + 3))

    x = ray_input
    for i in range(num_layers):
        if bn:
            x = layers.Dense(hidden_dim)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        else:
            x = layers.Dense(hidden_dim, activation="relu")(x)

        # Check if we have to include residual connections
        if i % skip_layer == 0 and i > 0:
            x = layers.concatenate([x, ray_input], axis=-1)
        
    # Get the sigma value
    sigma = layers.Dense(1)(x)

    # Create a feature vector
    feature = layers.Dense(hidden_dim)(x)

    # Concatenate the feature vector with the direction input
    feature = layers.concatenate([feature, dir_input], axis=-1)
    if bn:
        x = layers.Dense(hidden_dim//2)(feature)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    else:
        x = layers.Dense(hidden_dim//2, activation="relu")(feature)

    # Get the rgb value
    rgb = layers.Dense(3)(x)

    outputs = layers.concatenate([rgb, sigma], axis=-1)

    nerf_model = keras.Model(inputs=[ray_input, dir_input], outputs=outputs)
    return nerf_model

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

def forward_pass(self, ray_origins, ray_directions, t_vals, l_xyz, l_dir, training=False):
    rays, dirs = sample_rays(ray_origins, ray_directions, t_vals)
    rays_enc = encode_position(rays, pos_encode_dims=l_xyz)
    dirs_enc = encode_position(dirs, pos_encode_dims=l_dir)

    predictions_coarse = self.coarse_model([rays_enc, dirs_enc], training=training)
    # predictions_coarse = self.coarse_model.predict([rays_enc, dirs_enc], batch_size=128)
    rgb_coarse, depth_coarse, weights_coarse = volume_render(predictions_coarse, t_vals)
    t_vals_coarse_mid = (0.5 * (t_vals[..., 1:] + t_vals[..., :-1]))
    t_vals_fine = sample_pdf(t_vals_coarse_mid, weights_coarse, self.ns_fine)
    t_vals_fine_all = ops.sort(ops.concatenate([t_vals, t_vals_fine], axis=-1), axis=-1)

    rays_fine, dirs_fine = sample_rays(ray_origins, ray_directions, t_vals_fine_all)
    rays_fine_enc = encode_position(rays_fine, pos_encode_dims=l_xyz)
    dirs_fine_enc = encode_position(dirs_fine, pos_encode_dims=l_dir)

    predictions_fine = self.fine_model([rays_fine_enc, dirs_fine_enc], training=training)
    # predictions_fine = self.fine_model.predict([rays_fine_enc, dirs_fine_enc], batch_size=128)
    rgb_fine, depth_fine, weights_fine = volume_render(predictions_fine, t_vals_fine_all)
    return (rgb_coarse, rgb_fine), (depth_coarse, depth_fine), (weights_coarse, weights_fine), (predictions_coarse, predictions_fine)