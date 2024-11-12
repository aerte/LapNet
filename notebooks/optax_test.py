import jax
import jax.numpy as jnp
import optax
import numpy as np

BATCH_SIZE = 5
NUM_TRAIN_STEPS = 1_000
RAW_TRAINING_DATA = np.random.randint(255, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))

TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)

initial_params = {
    'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
    'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
}


def net(x: jnp.ndarray, params: optax.Params) -> jnp.ndarray:
  x = jnp.dot(x, params['hidden'])
  x = jax.nn.relu(x)
  x = jnp.dot(x, params['output'])
  return x


def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  y_hat = net(batch, params)

  # optax also provides a number of common loss functions.
  loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

  return loss_value.mean()

def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
  opt_state = optimizer.init(params)

  @jax.jit
  def step(params, opt_state, batch, labels):
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

  for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
    params, opt_state, loss_value = step(params, opt_state, batch, labels)
    if i % 100 == 0:
      print(f'step {i}, loss: {loss_value}')

  return params

# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
optimizer = optax.adam(learning_rate=1e-2)
params = fit(initial_params, optimizer)


#####################################################################################################

# Define the custom transformation
def scale_by_adaptive_magnitude(decay_rate=0.9, eps=1e-8):
  """
  Scales updates by the inverse of the running average of the gradient magnitudes.

  Args:
  - decay_rate: Float, controls the smoothing of the running average.
  - eps: Float, a small epsilon to prevent division by zero.

  Returns:
  - An Optax transformation that scales updates adaptively.
  """

  def init_fn(params):
    # Initialize a state to hold the running averages of the gradient magnitudes
    avg_grads = jax.tree_map(lambda p: jnp.zeros_like(p), params)
    return avg_grads

  def update_fn(updates, state, params=None):
    # Compute the new average gradient magnitudes
    new_avg_grads = jax.tree_map(
      lambda g, avg_g: decay_rate * avg_g + (1 - decay_rate) * jnp.abs(g),
      updates,
      state
    )

    # Scale updates by the inverse of the running average
    scaled_updates = jax.tree_map(
      lambda g, avg_g: g / (avg_g + eps),
      updates,
      new_avg_grads
    )

    return scaled_updates, new_avg_grads

  return optax.GradientTransformation(init_fn, update_fn)


# Define the full custom optimizer
def custom_optimizer():
  return optax.chain(
    scale_by_adaptive_magnitude(),  # Custom transformation
    optax.scale_by_schedule(optax.constant_schedule(0.1)),  # Constant learning rate schedule
    optax.scale(-1.0)  # Scale to perform gradient descent
  )


# Initialize the optimizer with example parameters
params = {'w': jnp.array([1.0, 2.0, 3.0]), 'b': jnp.array([0.5])}  # Example parameters
optimizer = custom_optimizer()
opt_state = optimizer.init(params)

# Define a dummy gradient for testing
grads = {'w': jnp.array([0.1, -0.2, 0.15]), 'b': jnp.array([-0.05])}

# Apply the custom optimizer's update
updates, opt_state = optimizer.update(grads, opt_state)
new_params = optax.apply_updates(params, updates)

print("Updated parameters:", new_params)

params = fit(initial_params, optimizer)