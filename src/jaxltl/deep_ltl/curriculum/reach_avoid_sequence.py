from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReachAvoidSequence(NamedTuple):
    """A reach-avoid sequence consisting of sets of assignments to reach and avoid."""

    # TODO: +1 for epsilon
    # Each row consists of assignment indices, sorted in descending order, with -1 padding
    reach: jax.Array  # shape: (max_length, num_assignments)
    avoid: jax.Array  # shape: (max_length, num_assignments)

    def advance(self) -> "ReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence, with
        the last step padded.
        """
        seq = jax.tree.map(lambda x: jnp.roll(x, -1, axis=0), self)
        seq = jax.tree.map(lambda x: x.at[-1, :].set(-1), seq)
        return seq

    @property
    def depth(self) -> int:
        """Compute the depth of the sequence (number of non-padded steps)."""

        padded_steps = self.reach[:, 0] == -1
        return int(jnp.sum(~padded_steps))
