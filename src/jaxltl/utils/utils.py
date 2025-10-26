"""Common utility functions for different modules."""

import jax
import jax.numpy as jnp


def map_assignment_to_index(
    propositions: jax.Array, assignments: jax.Array
) -> jax.Array:
    """Maps a boolean assignment to its corresponding index in the environment's assignments.

    Args:
        propositions: A boolean array of shape (num_propositions,) representing the assignment.
        assignments: A boolean array of shape (num_assignments, num_propositions) representing all possible assignments.

    Returns:
        An integer array of shape () representing the index of the matching assignment.
    """

    comparison = jnp.all(assignments == propositions, axis=1)
    return jnp.argmax(comparison)
