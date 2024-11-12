import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_structure

pytree = [{'biases': (128,), 'weights': (1, 128)},
 {'biases': (128,), 'weights': (128, 128)},
 {'biases': (1,), 'weights': (128, 1)}]

array, unravel_fun = ravel_pytree(pytree[0])
print(array.shape)
for leave in pytree[1:]:
    temp = ravel_pytree(leave)[0]
    # print(temp)
    array = jnp.vstack((array, ravel_pytree(leave)[0]))
#print(array.shape)
print(array)
print(unravel_fun(temp))

unravel_leaves = jax.vmap(ravel_pytree, in_axes=())(pytree)