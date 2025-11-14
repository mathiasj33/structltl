from collections.abc import Sequence

import equinox as eqx
import jax

from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.ltl.logic.assignment import Assignment


class GraphSequenceSampler(eqx.Module):
    """Base class for graph-based sequence samplers."""

    propositions: tuple[str, ...]
    assignments: tuple[Assignment, ...]

    def __init__(self, propositions: Sequence[str], assignments: Sequence[Assignment]):
        self.propositions = tuple(propositions)
        self.assignments = tuple(assignments)

    def sample(self, key: jax.Array) -> GraphReachAvoidSequence:
        raise NotImplementedError
