"""Code directly from the VMC network repository."""

"""Stochastic reconfiguration (SR) routine."""
import jax
import jax.flatten_util
import jax.numpy as jnp

#from vmcnet.utils.typing import Array, ModelApply, P, Tuple

import chex

from spring_reqs import *



def get_spring_update_fn(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    momentum: chex.Scalar = 0.0,
):
    """
    Get the SPRING update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping (float): damping parameter
        mu (float): SPRING-specific regularization

    Returns:
        Callable: SPRING update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def spring_update_fn(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]

        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad

        log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(
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
        SR_G = (1 - momentum) * SR_G + momentum * prev_grad             # Equation (34)

        return unravel_fn(SR_G)

    return spring_update_fn


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads

UpdateParamFn = Callable[[P, D, S, chex.PRNGKey], Tuple[P, D, S, Dict, chex.PRNGKey]]


def _update_metrics_with_noclip(
    energy_noclip: float, variance_noclip: float, metrics: Dict
) -> Dict:
    if energy_noclip is not None:
        metrics.update({"energy_noclip": energy_noclip})
    if variance_noclip is not None:
        metrics.update({"variance_noclip": variance_noclip})
    return metrics



def _make_traced_fn_with_single_metrics(
    update_param_fn: UpdateParamFn[P, D, S],
    apply_pmap: bool,
    metrics_to_get_first: Optional[chex.Iterable[str]] = None,
) -> UpdateParamFn[P, D, S]:
    if not apply_pmap:
        return jax.jit(update_param_fn)

    pmapped_update_param_fn = pmap(update_param_fn)

    def pmapped_update_param_fn_with_single_metrics(params, data, optimizer_state, key):
        params, data, optimizer_state, metrics, key = pmapped_update_param_fn(
            params, data, optimizer_state, key
        )
        if metrics_to_get_first is None:
            metrics = get_first(metrics)
        else:
            for metric in metrics_to_get_first:
                distributed_metric = metrics.get(metric)
                if distributed_metric is not None:
                    metrics[metric] = get_first(distributed_metric)

        return params, data, optimizer_state, metrics, key

    return pmapped_update_param_fn_with_single_metrics


def create_grad_energy_update_param_fn(
    energy_data_val_and_grad: ValueGradEnergyFn[P],
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy.

    See :func:`~vmcnet.train.vmc.vmc_loop` for its usage.

    Args:
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        optimizer_apply (Callable): applies an update to the parameters. Has signature
            (grad_energy, params, optimizer_state) -> (new_params, new_optimizer_state).
        get_position_fn (GetPositionFromData): gets the walker positions from the MCMC
            data.
        update_data_fn (Callable): function which updates data for new params
        apply_pmap (bool, optional): whether to apply jax.pmap to the walker function.
            If False, applies jax.jit. Defaults to True.

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
        The function is pmapped if apply_pmap is True, and jitted if apply_pmap is
        False.
    """

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        key, subkey = jax.random.split(key)

        energy_data, grad_energy = energy_data_val_and_grad(params, subkey, position)
        energy, aux_energy_data = energy_data

        grad_energy = pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(
            grad_energy,
            params,
            optimizer_state,
            data,
            dict(centered_local_energies=aux_energy_data["centered_local_energies"]),
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": aux_energy_data["variance"]}
        metrics = _update_metrics_with_noclip(
            aux_energy_data["energy_noclip"],
            aux_energy_data["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = _make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn