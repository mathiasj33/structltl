from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EpsilonType
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl.logic.boolean_parser import (
    AndNode,
    EmptyNode,
    MultiAndNode,
    MultiOrNode,
    Node,
    NotNode,
    OrNode,
    VarNode,
)

# Define integer constants for node types
NODE_TYPE_AND = 0
NODE_TYPE_OR = 1
NODE_TYPE_NOT = 2
NODE_TYPE_EMPTY = 3
NODE_TYPE_EPSILON = 4
NODE_TYPE_VAR = 5


class JaxGraphReachAvoidSequence(NamedTuple):
    """Jax representation of a reach-avoid sequence with assignments and graphs."""

    # Assignment-based representation
    reach_assignments: jax.Array  # shape: (..., max_length, num_assignments)
    avoid_assignments: jax.Array  # shape: (..., max_length, num_assignments)

    # Graph-based representation
    reach_graphs: jraph.GraphsTuple
    avoid_graphs: jraph.GraphsTuple

    repeat_last: jax.Array
    last_index: jax.Array

    def advance(self) -> "JaxGraphReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence."""
        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return self._replace(last_index=self.last_index + 1)

        def _advance_step():
            # Advance assignment arrays one step
            new_reach = jnp.roll(self.reach_assignments, -1, axis=-2)
            new_avoid = jnp.roll(self.avoid_assignments, -1, axis=-2)

            # Pad the last row with -1s
            new_reach = new_reach.at[..., -1, :].set(-1)
            new_avoid = new_avoid.at[..., -1, :].set(-1)

            # Advance graph arrays one step by rolling the sequence dimension
            def roll_graphs(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
                # Reshape and roll nodes
                original_shape = graphs.nodes.shape  # type: ignore[attribute-error]
                num_states, max_seqs, seq_len, max_nodes, num_features = original_shape
                batch_size = num_states * max_seqs

                reshaped_nodes = graphs.nodes.reshape(  # type: ignore[attribute-error]
                    batch_size, seq_len, max_nodes, num_features
                )
                rolled_nodes = jnp.roll(reshaped_nodes, -1, axis=1)
                rolled_nodes = rolled_nodes.at[:, -1, ...].set(-1)

                # Roll n_node and n_edge, which also have a sequence dimension
                rolled_n_node = jnp.roll(graphs.n_node, -1, axis=-1)
                rolled_n_node = rolled_n_node.at[..., -1].set(0)

                rolled_n_edge = jnp.roll(graphs.n_edge, -1, axis=-1)
                rolled_n_edge = rolled_n_edge.at[..., -1].set(0)

                # Note: senders/receivers/edges are flattened and don't need direct rolling.
                # jraph handles their interpretation based on the rolled n_node/n_edge.
                return graphs._replace(
                    nodes=rolled_nodes.reshape(original_shape),
                    n_node=rolled_n_node,
                    n_edge=rolled_n_edge,
                )

            new_reach_graphs = roll_graphs(self.reach_graphs)
            new_avoid_graphs = roll_graphs(self.avoid_graphs)

            return JaxGraphReachAvoidSequence(
                reach_assignments=new_reach,
                avoid_assignments=new_avoid,
                reach_graphs=new_reach_graphs,
                avoid_graphs=new_avoid_graphs,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros_like(self.last_index),
            )

        return jax.lax.cond(
            jnp.all(should_repeat),  # Ensure condition is a scalar for cond
            _repeat_step,
            _advance_step,
        )

    @property
    def depth(self) -> jax.Array:
        """Compute the depth of the sequence (number of non-padded steps)."""
        # Depth is determined by the assignment sequence
        padded_steps = self.reach_assignments[..., 0] == -1
        return jnp.sum(~padded_steps, axis=-1)

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[GraphReachAvoidSequence]],
        env: Environment | EnvWrapper,
        max_nodes: int,
        max_edges: int,
    ) -> "JaxGraphReachAvoidSequence":
        """
        Converts a mapping from LDBA states to lists of GraphReachAvoidSequences
        into a batched Jax representation.
        """
        num_states = len(state_to_seqs)
        max_seqs = max((len(seqs) for seqs in state_to_seqs.values()), default=0)
        max_len = max(
            (
                len(s.reach_avoid_assignments)
                for seqs in state_to_seqs.values()
                for s in seqs
            ),
            default=0,
        )

        # --- Assignment processing ---
        reach_assign = -np.ones(
            (num_states, max_seqs, max_len, len(env.assignments)), dtype=np.int32
        )
        avoid_assign = -np.ones_like(reach_assign)

        for state, seqs in state_to_seqs.items():
            for s_idx, seq in enumerate(seqs):
                for t_idx, (r, a) in enumerate(seq.reach_avoid_assignments):
                    if isinstance(r, EpsilonType):
                        reach_assign[state, s_idx, t_idx, 0] = len(env.assignments)
                    else:
                        for j, assign in enumerate(r):
                            reach_assign[state, s_idx, t_idx, j] = (
                                env.assignments.index(assign)
                            )
                    for j, assign in enumerate(a):
                        avoid_assign[state, s_idx, t_idx, j] = env.assignments.index(
                            assign
                        )

        # --- Graph processing ---
        reach_graph_list, avoid_graph_list = [], []
        for state in range(num_states):
            for s_idx in range(max_seqs):
                for t_idx in range(max_len):
                    try:
                        seq = state_to_seqs[state][s_idx]
                        r_graph, a_graph = seq.reach_avoid_graphs[t_idx]
                    except (KeyError, IndexError):
                        r_graph, a_graph = None, None  # Padding graphs

                    reach_graph_list.append(
                        _convert_to_jraph(
                            r_graph, env.propositions, max_nodes, max_edges
                        )
                    )
                    avoid_graph_list.append(
                        _convert_to_jraph(
                            a_graph, env.propositions, max_nodes, max_edges
                        )
                    )

        reach_graphs = jraph.batch(reach_graph_list)
        avoid_graphs = jraph.batch(avoid_graph_list)

        # Reshape graph features to match batching structure
        # New shape: (num_states, max_seqs, max_len, max_nodes, num_features)
        graph_node_shape = (num_states, max_seqs, max_len, max_nodes, 2)
        reach_graphs = reach_graphs._replace(
            nodes=reach_graphs.nodes.reshape(graph_node_shape),  # type: ignore[attribute-error]
        )
        avoid_graphs = avoid_graphs._replace(
            nodes=avoid_graphs.nodes.reshape(graph_node_shape),  # type: ignore[attribute-error]
        )

        return cls(
            reach_assignments=jnp.array(reach_assign),
            avoid_assignments=jnp.array(avoid_assign),
            reach_graphs=reach_graphs,
            avoid_graphs=avoid_graphs,
            repeat_last=jnp.ones((num_states, max_seqs), dtype=jnp.int32),
            last_index=jnp.zeros((num_states, max_seqs), dtype=jnp.int32),
        )


def _convert_to_jraph(
    graph_root: Node | EpsilonType | None,
    propositions: tuple[str, ...],
    max_nodes: int,
    max_edges: int,
) -> jraph.GraphsTuple:
    """Converts a single boolean formula graph to a padded jraph.GraphsTuple."""
    if graph_root is None:  # Padding graph
        return jraph.GraphsTuple(
            nodes=np.full((max_nodes, 2), -1, dtype=np.int32),
            edges=None,
            senders=np.zeros(max_edges, dtype=np.int32),  # type: ignore[arg-type]
            receivers=np.zeros(max_edges, dtype=np.int32),  # type: ignore[arg-type]
            n_node=np.array([0]),  # type: ignore[arg-type]
            n_edge=np.array([0]),  # type: ignore[arg-type]
            globals={"root_indices": np.array([-1], dtype=np.int32)},
        )

    node_map: dict[Node | EpsilonType, int] = {}
    node_features, senders, receivers = [], [], []

    def add_node(node: Node | EpsilonType) -> int:
        if node in node_map:
            return node_map[node]
        idx = len(node_map)
        node_map[node] = idx
        node_features.append(_get_node_features_as_int(node, propositions))
        return idx

    def build_graph(node: Node | EpsilonType):
        parent_idx = add_node(node)
        children: Sequence[Node] = []
        if isinstance(node, MultiAndNode | MultiOrNode):
            children = node.operands
        elif isinstance(node, AndNode | OrNode):
            children = [node.left, node.right]
        elif isinstance(node, NotNode):
            children = [node.operand]

        for child in children:
            child_idx = build_graph(child)
            senders.append(child_idx)
            receivers.append(parent_idx)

        return parent_idx

    root_idx = build_graph(graph_root)

    num_nodes = len(node_features)
    num_edges = len(senders)

    # Padding
    padded_nodes = np.full((max_nodes, 2), -1, dtype=np.int32)
    if num_nodes > 0:
        padded_nodes[:num_nodes] = np.array(node_features, dtype=np.int32)

    padded_senders = np.zeros(max_edges, dtype=np.int32)
    padded_senders[:num_edges] = senders

    padded_receivers = np.zeros(max_edges, dtype=np.int32)
    padded_receivers[:num_edges] = receivers

    return jraph.GraphsTuple(
        nodes=padded_nodes,
        edges=None,
        senders=padded_senders,  # type: ignore[arg-type]
        receivers=padded_receivers,  # type: ignore[arg-type]
        n_node=np.array([num_nodes]),  # type: ignore[arg-type]
        n_edge=np.array([num_edges]),  # type: ignore[arg-type]
        globals={"root_indices": np.array([root_idx], dtype=np.int32)},
    )


def _get_node_features_as_int(
    node: Node | EpsilonType, propositions: tuple[str, ...]
) -> list[int]:
    """Creates an integer feature vector for a graph node.
    Returns:
        A list of two integers: [node_type_id, proposition_id].
        proposition_id is -1 for non-variable nodes.
    """
    prop_id = -1
    if isinstance(node, VarNode):
        node_type_id = NODE_TYPE_VAR
        try:
            prop_id = propositions.index(node.name)
        except ValueError as err:
            raise ValueError(
                f"Proposition '{node.name}' not in environment propositions."
            ) from err
    elif isinstance(node, AndNode | MultiAndNode):
        node_type_id = NODE_TYPE_AND
    elif isinstance(node, OrNode | MultiOrNode):
        node_type_id = NODE_TYPE_OR
    elif isinstance(node, NotNode):
        node_type_id = NODE_TYPE_NOT
    elif isinstance(node, EmptyNode):
        node_type_id = NODE_TYPE_EMPTY
    elif isinstance(node, EpsilonType):
        node_type_id = NODE_TYPE_EPSILON
    else:
        raise TypeError(f"Unknown node type for featurization: {type(node)}")

    return [node_type_id, prop_id]
