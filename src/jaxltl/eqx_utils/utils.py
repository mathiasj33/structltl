import equinox as eqx
import jax
import jax.numpy as jnp


def add_batch_dim(module: eqx.Module, batch_size: int) -> eqx.Module:
    """Add a batch dimension to all array fields of an Equinox Module.

    Args:
        module: The Equinox Module to add a batch dimension to.
        batch_size: The size of the batch dimension to add.

    Returns:
        A new Equinox Module with a batch dimension added to all array fields.
    """
    params, static = eqx.partition(module, eqx.is_array)
    batched_params = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None, ...], (batch_size,) + x.shape), params
    )
    return eqx.combine(batched_params, static)
