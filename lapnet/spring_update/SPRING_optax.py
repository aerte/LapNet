import chex
from optax import Params, GradientTransformation, Updates, EmptyState
import jax.numpy as jnp
import jax
import jax.flatten_util

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol, runtime_checkable, Sequence, Union

from typing import NamedTuple
import lapnet.spring_update.optax_tree_utils as otu

NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.')

PyTree = Any
Shape = Sequence[int]

OptState = chex.ArrayTree  # States are arbitrary nests of `jnp.ndarrays`.
Params = chex.ArrayTree  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

Schedule = Callable[[chex.Numeric], chex.Numeric]
ScheduleState = Any
ScalarOrSchedule = Union[float, jax.Array, Schedule]

class TransformUpdateExtraArgsFn(Protocol):
  """An update function accepting additional keyword arguments."""

  def __call__(
      self,
      updates: Updates,
      state: OptState,
      params: Optional[Params] = None,
      **extra_args: Any,
  ) -> tuple[Updates, OptState]:
    """Update function with optional extra arguments.

    For example, an update function that requires an additional loss parameter
    (which might be useful for implementing learning rate schedules that depend
    on the current loss value) could be expressed as follows:

    >>> def update(updates, state, params=None, *, loss, **extra_args):
    ...   del extra_args
    ...   # use loss value

    Note that the loss value is keyword only, (it follows a ``*`` in the
    signature of the function). This implies users will get explicit errors if
    they try to use this gradient transformation without providing the required
    argument.

    Args:
      updates: The gradient updates passed to this transformation.
      state: The state associated with this transformation
      params: Optional params.
      **extra_args: Additional keyword arguments passed to this transform. All
        implementors of this interface should accept arbitrary keyword
        arguments, ignoring those that are not needed for the current
        transformation. Transformations that require specific extra args should
        specify these using keyword-only arguments.
    Returns:
      Transformed updates, and an updated value for the state.
    """

class GradientTransformationExtraArgs(GradientTransformation):
  """A specialization of GradientTransformation that supports extra args.

  Extends the existing GradientTransformation interface by adding support for
  passing extra arguments to the update function.

  Note that if no extra args are provided, then the API of this function is
  identical to the case of ``TransformUpdateFn``. This means that we can safely
  wrap any gradient transformation (that does not support extra args) as one
  that does. The new gradient transformation will accept (and ignore) any
  extra arguments that a user might pass to it. This is the behavior implemented
  by ``optax.with_extra_args_support()``.

  Attributes:
    update: Overrides the type signature of the update in the base type to
      accept extra arguments.
  """
  update: TransformUpdateExtraArgsFn


import os


def save_dict_to_desktop(data, extra_string = None, filename="output.txt"):
    # Get the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    # Create the full path for the file
    file_path = os.path.join(desktop_path, filename)

    # Write the dictionary to the file
    with open(file_path, "w") as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
        if extra_string is not None:
            file.write(extra_string)

    print(f"File saved to {file_path}")


################################ SPRING OPTAX CODE ###########################

from .spring_reqs import P, Array

class SpringState(NamedTuple):
    """State for the SPRING algorithm."""
    # Here we define any running parameters that we need
    damping: chex.Scalar = 0.001
    mu: chex.Scalar = 0.99
    momentum: chex.Scalar = 0.0
    prev_gradient: chex.PyTreeDef = None



def scale_by_spring(
        network: Callable,
        damping: float = 0.001,
        mu: float = 0.99,
        momentum: float = 0.0

) -> GradientTransformationExtraArgs:
    """Scales the update by SPRING."""


    # the init method needs to prepare the sizes of any running
    # variables like presumably the prev_grad
    def init_fn(params: Params) -> SpringState:
        prev_grad = otu.tree_zeros_like(params)
        del params
        return SpringState(damping=damping, mu=mu, momentum=momentum, prev_gradient=prev_grad)


    def update_fn(
            updates: Updates,
            state: SpringState,
            params: Params = None,
            keys = None,
            centered_energies = None, #SPRING PARAMETERS BELOW
            data = None,
    ):
        nchains = data.shape[0]

        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(state.prev_gradient)
        prev_grad_decayed = mu * prev_grad

        log_psi_grads = network(params, data) / jnp.sqrt(
            nchains
        )
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True) # Equation (9)

        T = Ohat @ Ohat.T
        ones = jnp.ones((nchains, 1))
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains) # Inner bracket of Equation (32)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed    # Given above Equation (31)

        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(
            T_reg, epsion_tilde, assume_a="pos"
        )                                                               # Equation (32)

        SR_G = dtheta_residual + prev_grad_decayed                      # Equation (33)
        SR_G = (1 - momentum) * SR_G + momentum * prev_grad

        updates = unravel_fn(SR_G)

        state = SpringState(damping=damping, mu=mu, momentum=momentum, prev_gradient=updates)

        return updates, state

    return GradientTransformationExtraArgs(init_fn, update_fn)















