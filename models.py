import keras
from keras import layers
from keras import ops
import tensorflow as tf

from data_utils import render_predictions, sample_pdf, sample_rays_flat

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

def create_nerf_complete_model(num_layers, hidden_dim, skip_layer, lxyz, ldir):
    ray_input = keras.Input(shape=(None, 2 * 3 * lxyz + 3))
    dir_input = keras.Input(shape=(None, 2 * 3 * ldir + 3))

    x = ray_input
    for i in range(num_layers):
        x = layers.Dense(hidden_dim, activation='relu')(x)

        # Check if we have to include residual connections
        if i % skip_layer == 0 and i > 0:
            x = layers.concatenate([x, ray_input], axis=-1)
        
    # Get the sigma value
    sigma = layers.Dense(1, activation='relu')(x)

    # Create a feature vector
    feature = layers.Dense(hidden_dim)(x)

    # Concatenate the feature vector with the direction input
    feature = layers.concatenate([feature, dir_input], axis=-1)
    x = layers.Dense(hidden_dim//2, activation='relu')(feature)

    # Get the rgb value
    rgb = layers.Dense(3, activation='sigmoid')(x)

    outputs = layers.concatenate([rgb, sigma], axis=-1)

    nerf_model = keras.Model(inputs=[ray_input, dir_input], outputs=outputs)
    return nerf_model

def render_rgb_depth(model, rays_flat, t_vals, batch_size, h, w, num_samples, rand=True, train=True):
    """Generates the RGB image and depth map from model prediction.

    Args:
        model: The MLP model that is trained to predict the rgb and
            volume density of the volumetric scene.
        rays_flat: The flattened rays that serve as the input to
            the NeRF model.
        t_vals: The sample points for the rays.
        rand: Choice to randomise the sampling strategy.
        train: Whether the model is in the training or testing phase.

    Returns:
        Tuple of rgb image and depth map.
    """
    # Get the predictions from the nerf model and reshape it.
    predictions = model(rays_flat, training=train)
    predictions = ops.reshape(predictions, (batch_size, h, w, num_samples, 4))

    # Slice the predictions into rgb and sigma.
    rgb = ops.sigmoid(predictions[..., :-1])
    sigma_a = ops.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    # delta shape = (num_samples)
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

    return (rgb, depth_map)

class NeRFTrainer(keras.Model):
    def __init__(self, coarse_model, fine_model, batch_size, ns_coarse, ns_fine):
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.batch_size = batch_size
        self.ns_coarse = ns_coarse
        self.ns_fine = ns_fine

        if not isinstance(self.coarse_model, keras.Model):
            raise TypeError("coarse_model must be a keras.Model instance")
        if not isinstance(self.fine_model, keras.Model):
            raise TypeError("fine_model must be a keras.Model instance")


    # def compile(self, optimizer_coarse, optimizer_fine, loss_fn):
    def compile(self, optimizer, loss_fn):
        super().compile()
        # self.optimizer_coarse = optimizer_coarse
        # self.optimizer_fine = optimizer_fine
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_coarse_tracker = keras.metrics.Mean(name="loss_coarse")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_tracker = keras.metrics.Mean(name="psnr")
    
    def train_step(self, inputs):
        # Get the image and the rays)
        (images, rays) = inputs
        l_xyz = int((ops.shape(self.coarse_model.inputs[0])[-1] - 3) / 6)
        l_dir = int((ops.shape(self.coarse_model.inputs[1])[-1] - 3) / 6)
        (ray_origins, ray_directions, rays_flat, dirs_flat, t_vals) = rays

        # Get image dimensions
        h, w = ops.shape(images)[1:3]

        with tf.GradientTape() as tape:
            # Get the predictions from the model
            rgbs, _, _ = self.forward_render(ray_origins, ray_directions, t_vals, h, w, l_xyz, l_dir, training=True)
            rgb_coarse, rgb_fine = rgbs

            loss_coarse = self.loss_fn(images, rgb_coarse)
            loss_fine = self.loss_fn(images, rgb_fine)

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
    
    # def train_step(self, inputs):
    #     # Get the image and the rays)
    #     (images, rays) = inputs
    #     l_xyz = int((ops.shape(self.coarse_model.inputs[0])[-1] - 3) / 6)
    #     l_dir = int((ops.shape(self.coarse_model.inputs[1])[-1] - 3) / 6)
    #     (ray_origins, ray_directions, rays_flat, dirs_flat, t_vals) = rays

    #     # Get image dimensions
    #     h, w = ops.shape(images)[1:3]

    #     with tf.GradientTape() as coarse_tape:
    #         # Get the predictions from the model
    #         predictions = self.coarse_model([rays_flat, dirs_flat], training=True)
    #         predictions_coarse = ops.reshape(predictions, (-1, h, w, self.ns_coarse, 4))
    #         rgb_coarse, _, weight_coarse = render_predictions(predictions_coarse, t_vals, rand=True)
    #         loss_coarse = self.loss_fn(images, rgb_coarse)

    #     # Compute middle values of t_vals for sampling
    #     t_vals_coarse_mid = (0.5 * (t_vals[..., 1:] + t_vals[..., :-1]))

    #     # Apply hierarchical sampling and get the t_vals for fine model
    #     t_vals_fine = sample_pdf(t_vals_coarse_mid, weight_coarse, self.ns_fine)
    #     t_vals_fine_all = ops.sort(ops.concatenate([t_vals, t_vals_fine], axis=-1), axis=-1)

    #     rays_flat_fine, dirs_flat_fine = sample_rays_flat(ray_origins, ray_directions, t_vals_fine_all, l_xyz, l_dir)

    #     with tf.GradientTape() as fine_tape:
    #         # Get the predictions from the model
    #         predictions_fine = self.fine_model([rays_flat_fine, dirs_flat_fine], training=True)
    #         predictions_fine = ops.reshape(predictions_fine, (-1, h, w, self.ns_fine + self.ns_coarse, 4))
    #         rgb_fine, _, _ = render_predictions(predictions_fine, t_vals_fine_all, rand=True)
    #         loss_fine = self.loss_fn(images, rgb_fine)

        
    #     # Apply gradient updates for the course model
    #     tv_coarse = self.coarse_model.trainable_variables
    #     grads_coarse = coarse_tape.gradient(loss_coarse, tv_coarse)
    #     self.optimizer_coarse.apply_gradients(zip(grads_coarse, tv_coarse))

    #     # Apply gradient updates for the fine model
    #     tv_fine = self.fine_model.trainable_variables
    #     grads_fine = fine_tape.gradient(loss_fine, tv_fine)
    #     self.optimizer_fine.apply_gradients(zip(grads_fine, tv_fine))


    #     # Get the PSNR of the reconstructed images and the source images
    #     psnr = ops.psnr(images, rgb_fine, max_val=1.0)

    #     # Compute the metrics
    #     self.loss_coarse_tracker.update_state(loss_coarse)
    #     self.loss_tracker.update_state(loss_fine)
    #     self.psnr_tracker.update_state(psnr)
    #     return {
    #         "loss_coarse": self.loss_coarse_tracker.result(),
    #         "loss": self.loss_tracker.result(),
    #         "psnr": self.psnr_tracker.result(),
    #     }
    
    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        l_xyz = int((ops.shape(self.coarse_model.inputs[0])[-1] - 3) / 6)
        l_dir = int((ops.shape(self.coarse_model.inputs[1])[-1] - 3) / 6)
        (ray_origins, ray_directions, rays_flat, dirs_flat, t_vals) = rays

        # Get image dimensions
        h, w = ops.shape(images)[1:3]

        rgbs, _, _ = self.forward_render(ray_origins, ray_directions, t_vals, h, w, l_xyz, l_dir, training=False)

        (rgb_coarse, rgb_fine) = rgbs

        # # Get the predictions from the model.
        # predictions = self.coarse_model([rays_flat, dirs_flat], training=False)
        # predictions_coarse = ops.reshape(predictions, (-1, h, w, self.ns_coarse, 4))
        # rgb_coarse, depth_coarse = render_predictions(predictions_coarse, t_vals, rand=True)
        loss_coarse = self.loss_fn(images, rgb_coarse)

        # # Compute middle values of t_vals for sampling
        # t_vals_coarse_mid = (0.5 * (t_vals[..., 1:] + t_vals[..., :-1]))

        # # Apply hierarchical sampling and get the t_vals for fine model
        # t_vals_fine = sample_pdf(t_vals_coarse_mid, depth_coarse[..., None], self.ns_fine)
        # t_vals_fine_all = ops.sort(ops.concatenate([t_vals, t_vals_fine], axis=-1), axis=-1)

        # rays_flat_fine, dirs_flat_fine = sample_rays_flat(ray_origins, ray_directions, t_vals_fine_all, l_xyz, l_dir)

        # predictions_fine = self.fine_model([rays_flat_fine, dirs_flat_fine], training=False)
        # predictions_fine = ops.reshape(predictions_fine, (-1, h, w, self.ns_fine + self.ns_coarse, 4))
        # rgb_fine, depth_fine = render_predictions(predictions_fine, t_vals_fine_all, rand=True)
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
    
    def forward_render(self, ray_origins, ray_directions, t_vals, height, width, l_xyz, l_dir, training=False):
        rays_flat, dirs_flat = sample_rays_flat(ray_origins, ray_directions, t_vals, l_xyz, l_dir)
        predictions = self.coarse_model([rays_flat, dirs_flat], training=training)
        predictions_coarse = ops.reshape(predictions, (-1, height, width, self.ns_coarse, 4))
        rgb_coarse, depth_coarse, weights_coarse = render_predictions(predictions_coarse, t_vals, rand=True)

        t_vals_coarse_mid = (0.5 * (t_vals[..., 1:] + t_vals[..., :-1]))

        t_vals_fine = sample_pdf(t_vals_coarse_mid, weights_coarse, self.ns_fine)
        t_vals_fine_all = ops.sort(ops.concatenate([t_vals, t_vals_fine], axis=-1), axis=-1)

        rays_flat_fine, dirs_flat_fine = sample_rays_flat(ray_origins, ray_directions, t_vals_fine_all, l_xyz, l_dir)

        predictions_fine = self.fine_model([rays_flat_fine, dirs_flat_fine], training=training)
        predictions_fine = ops.reshape(predictions_fine, (-1, height, width, self.ns_fine + self.ns_coarse, 4))
        rgb_fine, depth_fine, weights_fine = render_predictions(predictions_fine, t_vals_fine_all, rand=True)

        return (rgb_coarse, rgb_fine), (depth_coarse, depth_fine), (weights_coarse, weights_fine)

class NeRF(keras.Model):
    def __init__(self, nerf_model, batch_size, num_samples):
        super().__init__()
        self.nerf_model = nerf_model
        self.batch_size = batch_size
        self.num_samples = num_samples

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_tracker = keras.metrics.Mean(name="psnr")
    
    def train_step(self, inputs):
        # Get the image and the rays)
        (images, rays) = inputs
        (rays_flat, t_vals)  = rays

        # Get image dimensions
        h, w = ops.shape(images)[1:3]

        with tf.GradientTape() as tape:
            # Get the predictions from the model
            predictions = self.nerf_model(rays_flat, training=True)
            predictions = ops.reshape(predictions, (-1, h, w, self.num_samples, 4))
            rgb, _, _ = render_predictions(predictions, t_vals, rand=True)
            # rgb, _ = render_rgb_depth(
            #     self.nerf_model, 
            #     rays_flat, t_vals, 
            #     batch_size=self.batch_size,
            #     h=h,
            #     w=w,
            #     num_samples=self.num_samples,
            #     rand=True,
            #     train=True
            # )
            loss = self.loss_fn(images, rgb)
    

        # Get the trainable variables
        trainable_vars = self.nerf_model.trainable_variables

        # Get the gradeitns of the trainable variables with respect to the loss
        gradients = tape.gradient(loss, trainable_vars)

        # Apply the grads and optimize the model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Get the PSNR of the reconstructed images and the source images
        # psnr = tf.image.psnr(images, rgb, max_val=1.0)
        psnr = ops.psnr(images, rgb, max_val=1.0)

        # Compute the metrics
        self.loss_tracker.update_state(loss)
        self.psnr_tracker.update_state(psnr)
        return {
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
        }
    
    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        # Get image dimensions
        h, w = ops.shape(images)[1:3]

        # Get the predictions from the model.
        predictions = self.nerf_model(rays_flat, training=False)
        # predictions = self.nerf_model.predict(rays_flat)
        predictions = ops.reshape(predictions, (-1, h, w, self.num_samples, 4))
        rgb, _, _ = render_predictions(predictions, t_vals, rand=True)
        # rgb, _ = render_rgb_depth(
        #     model=self.nerf_model, 
        #     rays_flat=rays_flat, 
        #     t_vals=t_vals, 
        #     batch_size=self.batch_size,
        #     h=h,
        #     w=w,
        #     num_samples=self.num_samples,
        #     rand=True,
        #     train=False
        # )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = ops.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_tracker.update_state(psnr)
        return {
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_tracker]
    