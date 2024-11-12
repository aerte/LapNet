# Copyright 2020 DeepMind Technologies Limited.
# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Callable, Tuple

import chex
from lapnet import constants
from lapnet import hamiltonian
from lapnet import networks
import jax
import jax.numpy as jnp
import kfac_jax
from kfac_jax import utils as kfac_utils


# Evaluation of total energy.
# (params, key, (electrons, atoms)) -> energy, auxiliary_loss_data
TotalEnergy = Callable[
    [networks.ParamTree, chex.PRNGKey, Tuple[jnp.ndarray, jnp.ndarray]],
    Tuple[jnp.ndarray, 'AuxiliaryLossData']]


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
  variance: jnp.DeviceArray
  local_energy: jnp.DeviceArray
  outlier_mask: jnp.DeviceArray


def make_loss(network: networks.LogWaveFuncLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              rm_outlier=False,
              el_partition: int = 1,
              local_energy_outlier_width=0.0) -> TotalEnergy:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    rm_outlier: If True, outliers will be removed from the computation from both
      loss and its gradients, otherwise outliers would be clipped when
      computing gradients, in which case clipping won't happen in the computation
      of the loss value.
    local_energy_outlier_width: If greater than zero, the local energy outliers
      will be identified as the ones that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. Those outliers will be removed from the calculation
      of both the energy and its gradient, if `rm_outlier` is True.
  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0), out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)
  batch_local_energy_with_param = lambda params, data: (params, batch_local_energy(params, None, data))

  def pmean_with_mask(value, mask):
    '''
    Only take pmean with the not-masked-out value (namely mask > 0). Here `mask`
    is expected to only take value between 0 and 1.
    '''
    return (kfac_utils.psum_if_pmap(jnp.sum(value * mask), axis_name=constants.PMAP_AXIS_NAME) /
            (kfac_utils.psum_if_pmap(jnp.sum(mask), axis_name=constants.PMAP_AXIS_NAME)))

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.shape[0])
    if el_partition > 0 :
      btz = data.shape[0] // el_partition
      data = data.reshape((-1,btz)+data.shape[1:])
      _,e_l = jax.lax.scan(batch_local_energy_with_param, params, data)
      e_l = e_l.reshape(-1)
    else:
      e_l = batch_local_energy(params, keys, data)
    # is_finite is false for inf and nan. We should throw them away anyways.
    is_finite = jnp.isfinite(e_l)
    # Then we convert nan to 0 and inf to large numbers, otherwise we won't
    # be able to mask them out. It's ok to do this cast because they will be
    # masked away in the following computation.
    e_l = jnp.nan_to_num(e_l)

    # if not `rm_outlier`, which means we will do clipping instead, in which case
    # we don't clip when computing the energy but do clip in gradient computation.
    if rm_outlier and local_energy_outlier_width > 0.:
      # This loss is computed only for outlier computation
      loss = pmean_with_mask(e_l, is_finite)
      tv = pmean_with_mask(jnp.abs(e_l - loss), is_finite)
      mask = (
        (loss - local_energy_outlier_width * tv < e_l) &
        (loss + local_energy_outlier_width * tv > e_l) &
        is_finite)
    else:
      mask = is_finite

    loss = pmean_with_mask(e_l, mask)
    variance = pmean_with_mask((e_l - loss)**2, mask)

    return loss, AuxiliaryLossData(variance=variance,
                                   local_energy=e_l,
                                   outlier_mask=mask)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?

      # We have to apply mask here to remove the effect of possible inf and nan.
      median = jnp.median(jax.lax.all_gather(aux_data.local_energy,axis_name=constants.PMAP_AXIS_NAME))
      #median = jnp.median(aux_data.local_energy)
      tv = pmean_with_mask(jnp.abs(aux_data.local_energy - median), aux_data.outlier_mask)
      diff = jnp.clip(aux_data.local_energy,
                      median - clip_local_energy * tv,
                      median + clip_local_energy * tv)
      # renormalize diff
      diff = diff - pmean_with_mask(diff, aux_data.outlier_mask)
      device_batch_size = jnp.sum(aux_data.outlier_mask)
    else:
      diff = aux_data.local_energy - loss
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
    diff *= aux_data.outlier_mask

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    primals = primals[0], primals[2]
    tangents = tangents[0], tangents[2]
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, aux_data

    tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
    return primals_out, tangents_out

  return total_energy







def make_loss_local(network: networks.LogWaveFuncLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              rm_outlier=False,
              el_partition: int = 1,
              local_energy_outlier_width=0.0) -> TotalEnergy:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    rm_outlier: If True, outliers will be removed from the computation from both
      loss and its gradients, otherwise outliers would be clipped when
      computing gradients, in which case clipping won't happen in the computation
      of the loss value.
    local_energy_outlier_width: If greater than zero, the local energy outliers
      will be identified as the ones that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. Those outliers will be removed from the calculation
      of both the energy and its gradient, if `rm_outlier` is True.
  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0), out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)
  batch_local_energy_with_param = lambda params, data: (params, batch_local_energy(params, None, data))

  def pmean_with_mask(value, mask):
    '''
    Only take pmean with the not-masked-out value (namely mask > 0). Here `mask`
    is expected to only take value between 0 and 1.
    '''
    return (kfac_utils.psum_if_pmap(jnp.sum(value * mask), axis_name=constants.PMAP_AXIS_NAME) /
            (kfac_utils.psum_if_pmap(jnp.sum(mask), axis_name=constants.PMAP_AXIS_NAME)))

  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.shape[0])
    if el_partition > 0 :
      btz = data.shape[0] // el_partition
      data = data.reshape((-1,btz)+data.shape[1:])
      _,e_l = jax.lax.scan(batch_local_energy_with_param, params, data)
      e_l = e_l.reshape(-1)
    else:
      e_l = batch_local_energy(params, keys, data)
    # is_finite is false for inf and nan. We should throw them away anyways.
    is_finite = jnp.isfinite(e_l)
    # Then we convert nan to 0 and inf to large numbers, otherwise we won't
    # be able to mask them out. It's ok to do this cast because they will be
    # masked away in the following computation.
    e_l = jnp.nan_to_num(e_l)

    # if not `rm_outlier`, which means we will do clipping instead, in which case
    # we don't clip when computing the energy but do clip in gradient computation.
    if rm_outlier and local_energy_outlier_width > 0.:
      # This loss is computed only for outlier computation
      loss = pmean_with_mask(e_l, is_finite)
      tv = pmean_with_mask(jnp.abs(e_l - loss), is_finite)
      mask = (
        (loss - local_energy_outlier_width * tv < e_l) &
        (loss + local_energy_outlier_width * tv > e_l) &
        is_finite)
    else:
      mask = is_finite

    #loss = pmean_with_mask(e_l, mask)
    loss = e_l
    variance = pmean_with_mask((e_l - loss)**2, mask)



    return loss, AuxiliaryLossData(variance=variance,
                                   local_energy=e_l,
                                   outlier_mask=mask)

    # removed custom jvp

  return total_energy
