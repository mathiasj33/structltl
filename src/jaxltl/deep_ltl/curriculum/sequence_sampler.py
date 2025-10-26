"""Module for sequence sampling strategies in DeepLTL."""

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax


class ReachAvoidSequence(NamedTuple):
    """A reach-avoid sequence consisting of sets of assignments to reach and avoid."""

    # TODO: +1 for epsilon
    reach: jax.Array  # shape: (max_length, num_assignments + 1)  # + 1 for padding
    avoid: jax.Array  # shape: (max_length, num_assignments + 1)


class SequenceSampler(eqx.Module):
    """Base class for sequence sampling strategies."""

    num_assignments: int
    max_length: int

    def __init__(self, num_assignments: int, max_length: int):
        self.num_assignments = num_assignments
        self.max_length = max_length

    @abstractmethod
    def sample(self, key: jax.Array) -> ReachAvoidSequence:
        """Sample a reach-avoid sequence."""
        pass
