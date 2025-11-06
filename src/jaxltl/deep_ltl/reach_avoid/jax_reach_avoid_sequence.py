from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EpsilonType,
    ReachAvoidSequence,
)
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class JaxReachAvoidSequence(NamedTuple):
    """Jax representation of a reach-avoid sequence consisting of sets of assignments to reach and avoid."""

    # TODO: +1 for epsilon
    # Each row consists of assignment indices with -1 padding
    reach: jax.Array  # shape: (max_length, num_assignments)
    avoid: jax.Array  # shape: (max_length, num_assignments)

    def advance(self) -> "JaxReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence, with
        the last step padded.
        """
        seq = jax.tree.map(lambda x: jnp.roll(x, -1, axis=0), self)
        seq = jax.tree.map(lambda x: x.at[-1, :].set(-1), seq)
        return seq

    @property
    def depth(self) -> jax.Array:
        """Compute the depth of the sequence (number of non-padded steps)."""

        padded_steps = self.reach[:, 0] == -1
        return jnp.sum(~padded_steps)

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[ReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxReachAvoidSequence":
        """Converts a mapping from LDBA states to lists of ReachAvoidSequences into a
        batched Jax reach-avoid sequence.

        Returns:
            JaxReachAvoidSequence: with shape
                reach: (num_states, max_num_seqs, max_length, num_assignments)
                avoid: (num_states, max_num_seqs, max_length, num_assignments)
        """

        max_seqs = max(len(seqs) for seqs in state_to_seqs.values())
        max_length = max(
            len(seq.reach_avoid) for seqs in state_to_seqs.values() for seq in seqs
        )
        num_states = len(state_to_seqs)
        # Use numpy arrays and then convert to jax arrays for efficiency
        reach = -np.ones(
            (num_states, max_seqs, max_length, len(env.assignments)),
            dtype=np.int32,
        )
        avoid = -np.ones_like(reach)
        for state, seqs in state_to_seqs.items():
            for seq_idx, seq in enumerate(seqs):
                for i, (r, a) in enumerate(seq.reach_avoid):
                    if isinstance(r, EpsilonType):
                        continue  # TODO
                    for j, assignment in enumerate(r):
                        reach[state, seq_idx, i, j] = env.assignments.index(assignment)
                    for j, assignment in enumerate(a):
                        avoid[state, seq_idx, i, j] = env.assignments.index(assignment)
        return cls(reach=jnp.array(reach), avoid=jnp.array(avoid))
