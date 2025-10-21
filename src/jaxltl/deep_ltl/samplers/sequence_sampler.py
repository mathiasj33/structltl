"""Module for sequence sampling strategies in DeepLTL."""

from abc import abstractmethod
from typing import NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp


class ReachAvoidSequence(NamedTuple):
    """A reach-avoid sequence consisting of propositions to reach and avoid."""

    reach: jax.Array
    avoid: jax.Array


class SequenceSampler(eqx.Module):
    """Base class for sequence sampling strategies."""

    num_propositions: int
    max_length: int

    def __init__(self, num_propositions: int, max_length: int):
        super().__init__()
        self.num_propositions = num_propositions
        self.max_length = max_length

    @abstractmethod
    def sample(self, key: jax.Array) -> ReachAvoidSequence:
        """Sample a reach-avoid sequence."""
        pass


class ReachSampler(SequenceSampler):
    def __init__(self, num_propositions: int, max_length: int):
        super().__init__(num_propositions, max_length)

    @override
    def sample(self, key: jax.Array) -> ReachAvoidSequence:
        reach = jax.random.randint(
            key, (self.max_length,), 1, self.num_propositions + 1
        )
        avoid = jnp.zeros_like(reach)
        return ReachAvoidSequence(reach=reach, avoid=avoid)
