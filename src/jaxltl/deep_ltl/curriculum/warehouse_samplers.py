import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.graph_sequence_sampler import GraphSequenceSampler
from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import Node, VarNode


@functools.lru_cache(maxsize=1024)
def _compute_satisfying_assignments(
    graph: Node | None, all_assignments: tuple[Assignment, ...]
) -> frozenset[Assignment]:
    """Computes the set of assignments that satisfy a given boolean formula."""
    if graph is None:
        return frozenset()
    return frozenset(a for a in all_assignments if graph.eval(a))


class WarehousePropSampler(GraphSequenceSampler):
    """Samples simple reach-avoid sequences by sampling boolean formulae."""

    def __init__(
        self,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_length: int,
        max_nodes: int,
        max_edges: int,
    ):
        super().__init__(propositions, assignments, max_length, max_nodes, max_edges)

    def sample_graph(self, key: jax.Array) -> GraphReachAvoidSequence:
        reach_avoid_assignments = []
        reach_avoid_graphs = []

        # 1. Sample Reach Formula
        prop = jax.random.randint(key, (), 0, 2, dtype=jnp.int32).item()
        sample_props = ["crate", "vase"]
        reach = sample_props[prop]
        reach_graph = VarNode(reach)

        # 2. Sample Avoid Formula
        avoid_graph = None

        # 3. Compute satisfying assignments
        reach_assigns = _compute_satisfying_assignments(reach_graph, self.assignments)
        avoid_assigns = _compute_satisfying_assignments(avoid_graph, self.assignments)

        reach_avoid_graphs.append((reach_graph, avoid_graph))
        reach_avoid_assignments.append((reach_assigns, avoid_assigns))

        # Pad to max_length
        num_padding = self.max_length - len(reach_avoid_graphs)
        if num_padding > 0:
            padding_assignments = (frozenset(), frozenset())
            padding_graphs = (None, None)
            reach_avoid_assignments.extend([padding_assignments] * num_padding)
            reach_avoid_graphs.extend([padding_graphs] * num_padding)

        return GraphReachAvoidSequence(reach_avoid_assignments, reach_avoid_graphs)
