"""Module for sequence sampling strategies in DeepLTL."""

from abc import abstractmethod

import equinox as eqx
import jax

from jaxltl.deep_ltl.curriculum.reach_avoid_sequence import ReachAvoidSequence


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
