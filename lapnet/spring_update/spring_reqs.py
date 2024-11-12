"""Type definitions that can be reused across the VMCNet codebase.

Because type-checking with numpy/jax numpy can be tricky and does not always agree with
type-checkers, this package uses types for static type-checking when possible, but
otherwise they are intended for documentation and clarity.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from jax import Array
#from jax.typing import ArrayLike
import kfac_jax
import optax

import jax.numpy as jnp
import jax


# Currently using PyTree = Any just to improve readability in the code.
# A pytree is a "tree-like structure built out of container-like Python objects": see
# https://jax.readthedocs.io/en/latest/pytrees.html
PyTree = Any

# TypeVar for an arbitrary PyTree
T = TypeVar("T", bound=PyTree)

# TypeVar for a pytree containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxiliary MCMC data
D = TypeVar("D", bound=PyTree)

# TypeVar for MCMC metadata which is required to take a metropolis step.
M = TypeVar("M", bound=PyTree)

# TypeVar for a pytree containing model params
P = TypeVar("P", bound=PyTree)

# TypeVar for a pytree containing optimizer state
S = TypeVar("S", bound=PyTree)

# Actual optimizer states currently used
# TODO: Figure out how to make kfac_opt.State not be interpreted by mypy as Any
OptimizerState = Union[kfac_jax.optimizer.OptimizerState, optax.OptState]

LearningRateSchedule = Callable[[Array], Array]

ModelParams = Dict[str, Any]

# VMC state needed for a checkpoint. Values are:
#  1. The epoch
#  2. The MCMC walker data
#  3. The model parameters
#  4. The optimizer state
#  5. The RNG key
#CheckpointData = Tuple[int, D, P, S, PRNGKey]

ArrayList = List[Array]

# Single array in (sign, logabs) form
SLArray = Tuple[Array, Array]

SLArrayList = List[SLArray]

ParticleSplit = Union[int, Sequence[int]]

InputStreams = Tuple[Array, Optional[Array], Optional[Array], Optional[Array]]
ComputeInputStreams = Callable[[Array], InputStreams]

Backflow = Callable[[Array, Optional[Array]], Array]

Jastrow = Callable[[Array, Array, Array, Array, Array], Array]

ModelApply = Callable[[P, Array], Array]
#LocalEnergyApply = Callable[[P, Array, Optional[PRNGKey]], Array]

GetPositionFromData = Callable[[D], Array]
GetAmplitudeFromData = GetPositionFromData[D]
UpdateDataFn = Callable[[D, P], D]

ClippingFn = Callable[[Array, Array], Array]
#%%
## Required code for the Spring optimizer
import functools
from typing import Callable
from jax import core
import chex

EnergyAuxData = Dict[str, Any]
EnergyData = Tuple[Array, EnergyAuxData]
ValueGradEnergyFn = Callable[[P, chex.PRNGKey, Array], Tuple[EnergyData, P]]


def get_first(obj: T) -> T:
    """Get the first object in each leaf of a pytree.

    Can be used to grab the first instance of a replicated object on the first local
    device.
    """
    return jax.tree_map(lambda x: x[0], obj)


pmap = functools.partial(jax.pmap, axis_name="pmap_axis")

def wrap_if_pmap(p_func: Callable) -> Callable:
    """Make a function run if in a pmapped context."""

    def p_func_if_pmap(obj, axis_name):
        try:
            core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap

pmean_if_pmap = functools.partial(wrap_if_pmap(jax.lax.pmean), axis_name="pmap_axis")

"""Helper functions for pytrees."""
import chex
import jax.flatten_util


def tree_sum(tree1: T, tree2: T) -> T:
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_diff(tree1: T, tree2: T) -> T:
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a - b, tree1, tree2)


def tree_dist(tree1: T, tree2: T, mode="squares") -> Array:
    """Distance between two pytrees with the same structure."""
    dT = tree_diff(tree1, tree2)
    if mode == "squares":
        return tree_inner_product(dT, dT)
    raise ValueError(f"Unknown mode {mode}")


def tree_prod(tree1: T, tree2: T) -> T:
    """Leaf-wise product of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a * b, tree1, tree2)


def multiply_tree_by_scalar(tree: T, scalar: chex.Numeric) -> T:
    """Multiply all leaves of a pytree by a scalar."""
    return jax.tree_map(lambda x: scalar * x, tree)


def tree_inner_product(tree1: T, tree2: T) -> Array:
    """Inner product of two pytrees with the same structure."""
    leaf_inner_prods = jax.tree_map(lambda a, b: jnp.sum(a * b), tree1, tree2)
    return jnp.sum(jax.flatten_util.ravel_pytree(leaf_inner_prods)[0])


def tree_reduce_l1(xs: PyTree) -> chex.Numeric:
    """L1 norm of a pytree as a flattened vector."""
    concat_xs, _ = jax.flatten_util.ravel_pytree(xs)
    return jnp.sum(jnp.abs(concat_xs))