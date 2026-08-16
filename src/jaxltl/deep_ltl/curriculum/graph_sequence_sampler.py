from abc import abstractmethod
from collections.abc import Sequence

import jax

from jaxltl.deep_ltl.curriculum.sequence_sampler import SequenceSampler
from jaxltl.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)


class GraphSequenceSampler(SequenceSampler):
    """Base class for graph-based sequence samplers."""

    propositions: tuple[str, ...]
    assignments: tuple[Assignment, ...]
    max_length: int
    max_nodes: int
    max_edges: int
    max_clauses: int
    sample_clauses: bool

    def __init__(
        self,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_length: int,
        max_nodes: int,
        max_edges: int,
        max_clauses: int = 0,
        sample_clauses: bool = False,
    ):
        self.propositions = tuple(propositions)
        self.assignments = tuple(assignments)
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_clauses = max_clauses
        self.sample_clauses = sample_clauses

    def sample(self, key: jax.Array) -> JaxGraphReachAvoidSequence:
        # TODO: clean up all of this mess
        graph_seq = self.sample_graph(key)
        if self.sample_clauses:
            return JaxClauseReachAvoidSequence.from_seq(
                graph_seq,
                self.propositions,
                self.assignments,
                self.max_clauses,
            )
        return JaxGraphReachAvoidSequence.from_seq(
            graph_seq,
            self.propositions,
            self.assignments,
            self.max_nodes,
            self.max_edges,
        )

    @abstractmethod
    def sample_graph(self, key: jax.Array) -> BooleanReachAvoidSequence:
        raise NotImplementedError
