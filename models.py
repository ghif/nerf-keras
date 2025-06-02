import keras
from keras import layers
from keras import ops
import tensorflow as tf

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
            rgb, _ = render_rgb_depth(
                self.nerf_model, 
                rays_flat, t_vals, 
                batch_size=self.batch_size,
                h=h,
                w=w,
                num_samples=self.num_samples,
                rand=True,
                train=True
            )
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
        rgb, _ = render_rgb_depth(
            model=self.nerf_model, 
            rays_flat=rays_flat, 
            t_vals=t_vals, 
            batch_size=self.batch_size,
            h=h,
            w=w,
            num_samples=self.num_samples,
            rand=True
        )
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
    