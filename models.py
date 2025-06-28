import keras
from keras import layers
from keras import ops
import tensorflow as tf

from data_utils import sample_pdf, sample_rays, encode_position, volume_render

# Define NeRF model
def create_nerf_model(num_layers, hidden_dim, num_pos, pos_encode_dims):
    inputs = keras.Input(shape=(num_pos, 2 * 3 * pos_encode_dims + 3))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(hidden_dim)(x)
        x = layers.ReLU()(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection
            x = layers.concatenate([x, inputs], axis=-1)
    
    outputs = layers.Dense(units=4)(x)  # RGB + density
    return keras.Model(inputs=inputs, outputs=outputs)

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

class NeRFTrainer(keras.Model):
    def __init__(self, coarse_model, fine_model, batch_size, ns_coarse, ns_fine, l_xyz, l_dir):
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.batch_size = batch_size
        self.ns_coarse = ns_coarse
        self.ns_fine = ns_fine
        self.l_xyz = l_xyz
        self.l_dir = l_dir

        if not isinstance(self.coarse_model, keras.Model):
            raise TypeError("coarse_model must be a keras.Model instance")
        if not isinstance(self.fine_model, keras.Model):
            raise TypeError("fine_model must be a keras.Model instance")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_coarse_tracker = keras.metrics.Mean(name="loss_coarse")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_tracker = keras.metrics.Mean(name="psnr")
    
    def train_step(self, inputs):
        # Get the image and the rays
        (images, rays) = inputs
        
        (ray_origins, ray_directions, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model
            rgbs, _, _, _ = self.forward_pass(ray_origins, ray_directions, t_vals, self.l_xyz, self.l_dir, training=True)
            rgb_coarse, rgb_fine = rgbs
            loss_coarse = self.loss_fn(images, rgb_coarse)
            loss_fine = self.loss_fn(images, rgb_fine)
            
            # Combine the coarse and fine losses
            loss = loss_coarse + loss_fine

        # Apply gradient updates for the model
        tv_nerf = self.coarse_model.trainable_variables + self.fine_model.trainable_variables
        grads = tape.gradient(loss, tv_nerf)
        self.optimizer.apply_gradients(zip(grads, tv_nerf))

        # Get the PSNR of the reconstructed images and the source images
        psnr = ops.psnr(images, rgb_fine, max_val=1.0)

        # Compute the metrics
        self.loss_coarse_tracker.update_state(loss_coarse)
        self.loss_tracker.update_state(loss_fine)
        self.psnr_tracker.update_state(psnr)
        return {
            "loss_coarse": self.loss_coarse_tracker.result(),
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
        }
    
    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (ray_origins, ray_directions, t_vals) = rays

        # Get the predictions from the model.
        rgbs, _, _, _ = self.forward_pass(ray_origins, ray_directions, t_vals, self.l_xyz, self.l_dir, training=False)
        (rgb_coarse, rgb_fine) = rgbs

        loss_coarse = self.loss_fn(images, rgb_coarse)
        loss_fine = self.loss_fn(images, rgb_fine)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = ops.psnr(images, rgb_fine, max_val=1.0)

        # Compute our own metrics
        self.loss_coarse_tracker.update_state(loss_coarse)
        self.loss_tracker.update_state(loss_fine)
        self.psnr_tracker.update_state(psnr)
        return {
            "loss_coarse": self.loss_coarse_tracker.result(),
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_tracker]
    
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