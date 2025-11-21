from collections.abc import Sequence
from dataclasses import replace
from typing import TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EpsilonType
from jaxltl.ltl.logic.assignment import Assignment
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


class NodeData(TypedDict):
    prop_idx: jax.Array
    type_idx: jax.Array
    mask: jax.Array


class EdgeData(TypedDict):
    mask: jax.Array


def _roll_graphs(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Rolls a single sequence of graphs by one step."""
    # This function is only designed to work on a single, unbatched sequence.
    seq_len = graphs.n_node.shape[0]

    # Infer max_nodes and max_edges from the total size and sequence length.
    nodes = cast(NodeData, graphs.nodes)
    edges = cast(EdgeData, graphs.edges)
    senders = cast(jax.Array, graphs.senders)
    receivers = cast(jax.Array, graphs.receivers)

    total_nodes = nodes["type_idx"].shape[0]
    total_edges = senders.shape[0]

    max_nodes = total_nodes // seq_len
    max_edges = total_edges // seq_len

    # 1. Roll the per-graph feature arrays along the sequence axis.
    # We don't need to do this, since all n_node values are max_nodes and n_edge values
    # are max_edges.
    # rolled_n_node = jnp.roll(graphs.n_node, -1, axis=-1)
    # rolled_n_edge = jnp.roll(graphs.n_edge, -1, axis=-1)

    # 2. Roll the flattened node and edge arrays by one block.
    rolled_nodes = jax.tree.map(lambda x: jnp.roll(x, -max_nodes, axis=0), nodes)
    rolled_edges = jax.tree.map(lambda x: jnp.roll(x, -max_edges, axis=0), edges)
    rolled_senders = jnp.roll(senders, -max_edges, axis=0)
    rolled_receivers = jnp.roll(receivers, -max_edges, axis=0)

    # 3. Pad the last step of the sequence.
    # Pad nodes
    pad_nodes_start_idx = total_nodes - max_nodes
    rolled_nodes["mask"] = rolled_nodes["mask"].at[pad_nodes_start_idx:].set(False)
    padded_nodes = jax.tree.map(
        lambda v: v.at[pad_nodes_start_idx:].set(-1),
        {k: v for k, v in rolled_nodes.items() if k != "mask"},  # still JIT compatible
    )
    padded_nodes["mask"] = rolled_nodes["mask"]

    # Pad edges (create self-loops on the first node of each padded graph)
    pad_edges_start_idx = total_edges - max_edges
    padded_edges = jax.tree.map(
        lambda x: x.at[pad_edges_start_idx:].set(False), rolled_edges
    )
    last_graph_node_start_idx = total_nodes - max_nodes
    padded_senders = rolled_senders.at[pad_edges_start_idx:].set(
        last_graph_node_start_idx
    )
    padded_receivers = rolled_receivers.at[pad_edges_start_idx:].set(
        last_graph_node_start_idx
    )

    # 4. Adjust sender/receiver indices for the blocks that shifted forward.
    padded_senders -= max_nodes
    padded_receivers -= max_nodes

    padded_senders = padded_senders.at[-max_edges:].set(last_graph_node_start_idx)
    padded_receivers = padded_receivers.at[-max_edges:].set(last_graph_node_start_idx)

    # Set n_node and n_edge for the last step.
    # Again, we don't need to do this since they are constant.
    # rolled_n_node = rolled_n_node.at[-1].set(max_nodes)
    # rolled_n_edge = rolled_n_edge.at[-1].set(max_edges)

    return graphs._replace(
        nodes=padded_nodes,
        edges=padded_edges,
        senders=padded_senders,
        receivers=padded_receivers,
        n_node=graphs.n_node,
        n_edge=graphs.n_edge,
    )


class JaxGraphReachAvoidSequence(JaxReachAvoidSequence):
    """Jax representation of a reach-avoid sequence with assignments and graphs."""

    # Graph-based representation
    reach_graphs: jraph.GraphsTuple
    avoid_graphs: jraph.GraphsTuple

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    @override
    def advance(self) -> "JaxGraphReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence."""
        if self.reach_graphs.n_node.ndim > 1:
            raise ValueError(
                "The `advance` method is only supported for unbatched sequences "
                "(created with `from_seq`), not for batched sequences "
                "(created with `from_state_to_seqs`)."
            )

        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return replace(self, last_index=self.last_index + 1)

        def _advance_step():
            # Advance assignment arrays one step
            new_reach = jnp.roll(self.reach, -1, axis=-2)
            new_avoid = jnp.roll(self.avoid, -1, axis=-2)

            # Pad the last row with -1s
            new_reach = new_reach.at[-1, :].set(-1)
            new_avoid = new_avoid.at[-1, :].set(-1)

            # Advance graph arrays one step
            new_reach_graphs = _roll_graphs(self.reach_graphs)
            new_avoid_graphs = _roll_graphs(self.avoid_graphs)

            return JaxGraphReachAvoidSequence(
                reach=new_reach,
                avoid=new_avoid,
                reach_graphs=new_reach_graphs,
                avoid_graphs=new_avoid_graphs,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros_like(self.last_index),
            )

        return jax.lax.cond(
            jnp.all(should_repeat),
            _repeat_step,
            _advance_step,
        )

    @classmethod
    def from_seq(
        cls,
        seq: GraphReachAvoidSequence,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_nodes: int,
        max_edges: int,
    ) -> "JaxGraphReachAvoidSequence":
        """
        Converts a single GraphReachAvoidSequence into a batched Jax representation.
        """
        seq_len = len(seq.reach_avoid)

        # --- Pre-computation for efficiency ---
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)

        # --- Assignment processing ---
        reach_assign = -np.ones((seq_len, len(assignments)), dtype=np.int32)
        avoid_assign = -np.ones_like(reach_assign)

        for t_idx, (r, a) in enumerate(seq.reach_avoid):
            if isinstance(r, EpsilonType):
                reach_assign[t_idx, 0] = epsilon_idx
            else:
                for j, assign in enumerate(r):
                    reach_assign[t_idx, j] = assignment_map[assign]
            for j, assign in enumerate(a):
                avoid_assign[t_idx, j] = assignment_map[assign]

        # --- Graph processing (Fixed-Block Strategy) ---
        total_graphs = seq_len
        total_nodes = total_graphs * max_nodes
        total_edges = total_graphs * max_edges

        # Initialize flat arrays for all graphs in the batch
        all_reach_nodes = {
            "type_idx": -np.ones((total_nodes, 1), dtype=np.int32),
            "prop_idx": -np.ones((total_nodes, 1), dtype=np.int32),
            "mask": np.zeros((total_nodes, 1), dtype=np.bool_),
        }
        all_reach_edges = {"mask": np.zeros((total_edges, 1), dtype=np.bool_)}
        all_reach_senders = np.zeros(total_edges, dtype=np.int32)
        all_reach_receivers = np.zeros(total_edges, dtype=np.int32)
        all_reach_n_node = np.full(total_graphs, max_nodes, dtype=np.int32)
        all_reach_n_edge = np.full(total_graphs, max_edges, dtype=np.int32)

        all_avoid_nodes = jax.tree.map(np.copy, all_reach_nodes)
        all_avoid_edges = jax.tree.map(np.copy, all_reach_edges)
        all_avoid_senders = np.copy(all_reach_senders)
        all_avoid_receivers = np.copy(all_reach_receivers)
        all_avoid_n_node = np.copy(all_reach_n_node)
        all_avoid_n_edge = np.copy(all_reach_n_edge)

        for t_idx in range(seq_len):
            node_offset = t_idx * max_nodes
            edge_offset = t_idx * max_edges

            r_graph_root, a_graph_root = seq.reach_avoid_graphs[t_idx]

            # --- Process Reach Graph ---
            r_nodes, r_edges, r_send, r_recv, r_n_node, r_n_edge = _convert_to_arrays(
                r_graph_root, propositions, max_nodes, max_edges
            )
            for key, nodes in all_reach_nodes.items():
                nodes[node_offset : node_offset + r_n_node] = r_nodes[key]
            for key, edges in all_reach_edges.items():
                edges[edge_offset : edge_offset + r_n_edge] = r_edges[key]
            all_reach_senders[edge_offset : edge_offset + r_send.shape[0]] = (
                r_send + node_offset
            )
            all_reach_receivers[edge_offset : edge_offset + r_recv.shape[0]] = (
                r_recv + node_offset
            )
            all_reach_senders[
                edge_offset + r_send.shape[0] : edge_offset + max_edges
            ] = node_offset
            all_reach_receivers[
                edge_offset + r_recv.shape[0] : edge_offset + max_edges
            ] = node_offset

            # --- Process Avoid Graph ---
            a_nodes, a_edges, a_send, a_recv, a_n_node, a_n_edge = _convert_to_arrays(
                a_graph_root, propositions, max_nodes, max_edges
            )
            for key, nodes in all_avoid_nodes.items():
                nodes[node_offset : node_offset + a_n_node] = a_nodes[key]
            for key, edges in all_avoid_edges.items():
                edges[edge_offset : edge_offset + a_n_edge] = a_edges[key]
            all_avoid_senders[edge_offset : edge_offset + a_send.shape[0]] = (
                a_send + node_offset
            )
            all_avoid_receivers[edge_offset : edge_offset + a_recv.shape[0]] = (
                a_recv + node_offset
            )
            all_avoid_senders[
                edge_offset + a_send.shape[0] : edge_offset + max_edges
            ] = node_offset
            all_avoid_receivers[
                edge_offset + a_recv.shape[0] : edge_offset + max_edges
            ] = node_offset

        # Reshape n_node/n_edge to have batch dimensions
        reach_graphs = jraph.GraphsTuple(
            nodes=jax.tree.map(lambda x: x.reshape(-1), all_reach_nodes),
            edges=jax.tree.map(lambda x: x.reshape(-1), all_reach_edges),
            senders=all_reach_senders,  # type: ignore[operator]
            receivers=all_reach_receivers,  # type: ignore[operator]
            n_node=all_reach_n_node,  # type: ignore[operator]
            n_edge=all_reach_n_edge,  # type: ignore[operator]
            globals=None,
        )
        avoid_graphs = jraph.GraphsTuple(
            nodes=jax.tree.map(lambda x: x.reshape(-1), all_avoid_nodes),
            edges=jax.tree.map(lambda x: x.reshape(-1), all_avoid_edges),
            senders=all_avoid_senders,  # type: ignore[operator]
            receivers=all_avoid_receivers,  # type: ignore[operator]
            n_node=all_avoid_n_node,  # type: ignore[operator]
            n_edge=all_avoid_n_edge,  # type: ignore[operator]
            globals=None,
        )

        return cls(
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_graphs=reach_graphs,
            avoid_graphs=avoid_graphs,
            repeat_last=jnp.array(seq.repeat_last, dtype=jnp.int32),
            last_index=jnp.array(0, dtype=jnp.int32),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[GraphReachAvoidSequence]],
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
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
            (len(s.reach_avoid) for seqs in state_to_seqs.values() for s in seqs),
            default=0,
        )

        # --- Pre-computation for efficiency ---
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)

        # --- Assignment processing ---
        reach_assign = -np.ones(
            (num_states, max_seqs, max_len, len(assignments)), dtype=np.int32
        )
        avoid_assign = -np.ones_like(reach_assign)
        repeat_last_arr = np.ones((num_states, max_seqs), dtype=np.int32)

        for state, seqs in state_to_seqs.items():
            for s_idx, seq in enumerate(seqs):
                repeat_last_arr[state, s_idx] = seq.repeat_last
                for t_idx, (r, a) in enumerate(seq.reach_avoid):
                    if isinstance(r, EpsilonType):
                        reach_assign[state, s_idx, t_idx, 0] = epsilon_idx
                    else:
                        for j, assign in enumerate(r):
                            reach_assign[state, s_idx, t_idx, j] = assignment_map[
                                assign
                            ]
                    for j, assign in enumerate(a):
                        avoid_assign[state, s_idx, t_idx, j] = assignment_map[assign]

        # --- Graph processing (Fixed-Block Strategy) ---
        total_nodes_per_seq = max_len * max_nodes
        total_edges_per_seq = max_len * max_edges

        # Initialize arrays for all graphs in the batch
        all_reach_nodes = {
            "type_idx": -np.ones(
                (num_states, max_seqs, total_nodes_per_seq, 1), dtype=np.int32
            ),
            "prop_idx": -np.ones(
                (num_states, max_seqs, total_nodes_per_seq, 1), dtype=np.int32
            ),
            "mask": np.zeros(
                (num_states, max_seqs, total_nodes_per_seq, 1), dtype=np.bool_
            ),
        }
        all_reach_edges = {
            "mask": np.zeros(
                (num_states, max_seqs, total_edges_per_seq, 1), dtype=np.bool_
            )
        }
        all_reach_senders = np.zeros(
            (num_states, max_seqs, total_edges_per_seq), dtype=np.int32
        )
        all_reach_receivers = np.zeros(
            (num_states, max_seqs, total_edges_per_seq), dtype=np.int32
        )
        all_reach_n_node = np.full(
            (num_states, max_seqs, max_len), max_nodes, dtype=np.int32
        )
        all_reach_n_edge = np.full(
            (num_states, max_seqs, max_len), max_edges, dtype=np.int32
        )

        # Create identical structures for avoid graphs
        all_avoid_nodes = jax.tree.map(np.copy, all_reach_nodes)
        all_avoid_edges = jax.tree.map(np.copy, all_reach_edges)
        all_avoid_senders = np.copy(all_reach_senders)
        all_avoid_receivers = np.copy(all_reach_receivers)
        all_avoid_n_node = np.copy(all_reach_n_node)
        all_avoid_n_edge = np.copy(all_reach_n_edge)

        for state in range(num_states):
            for s_idx in range(max_seqs):
                for t_idx in range(max_len):
                    node_offset = t_idx * max_nodes
                    edge_offset = t_idx * max_edges
                    try:
                        seq = state_to_seqs[state][s_idx]
                        r_graph_root, a_graph_root = seq.reach_avoid_graphs[t_idx]
                    except (KeyError, IndexError):
                        r_graph_root, a_graph_root = None, None

                    # --- Process Reach Graph ---
                    r_nodes, r_edges, r_send, r_recv, r_n_node, r_n_edge = (
                        _convert_to_arrays(
                            r_graph_root, propositions, max_nodes, max_edges
                        )
                    )
                    for key, nodes_array in all_reach_nodes.items():
                        nodes_array[
                            state, s_idx, node_offset : node_offset + r_n_node
                        ] = r_nodes[key]
                    for key, edges_array in all_reach_edges.items():
                        edges_array[
                            state, s_idx, edge_offset : edge_offset + r_n_edge
                        ] = r_edges[key]
                    all_reach_senders[
                        state, s_idx, edge_offset : edge_offset + r_n_edge
                    ] = r_send + node_offset
                    all_reach_receivers[
                        state, s_idx, edge_offset : edge_offset + r_n_edge
                    ] = r_recv + node_offset
                    all_reach_senders[
                        state, s_idx, edge_offset + r_n_edge : edge_offset + max_edges
                    ] = node_offset
                    all_reach_receivers[
                        state, s_idx, edge_offset + r_n_edge : edge_offset + max_edges
                    ] = node_offset

                    # --- Process Avoid Graph ---
                    a_nodes, a_edges, a_send, a_recv, a_n_node, a_n_edge = (
                        _convert_to_arrays(
                            a_graph_root, propositions, max_nodes, max_edges
                        )
                    )
                    for key, nodes_array in all_avoid_nodes.items():
                        nodes_array[
                            state, s_idx, node_offset : node_offset + a_n_node
                        ] = a_nodes[key]
                    for key, edges_array in all_avoid_edges.items():
                        edges_array[
                            state, s_idx, edge_offset : edge_offset + a_n_edge
                        ] = a_edges[key]
                    all_avoid_senders[
                        state, s_idx, edge_offset : edge_offset + a_n_edge
                    ] = a_send + node_offset
                    all_avoid_receivers[
                        state, s_idx, edge_offset : edge_offset + a_n_edge
                    ] = a_recv + node_offset
                    all_avoid_senders[
                        state, s_idx, edge_offset + a_n_edge : edge_offset + max_edges
                    ] = node_offset
                    all_avoid_receivers[
                        state, s_idx, edge_offset + a_n_edge : edge_offset + max_edges
                    ] = node_offset

        # Reshape n_node/n_edge to have batch dimensions
        reach_graphs = jraph.GraphsTuple(
            nodes=jax.tree.map(lambda x: np.squeeze(x, axis=-1), all_reach_nodes),
            edges=jax.tree.map(lambda x: np.squeeze(x, axis=-1), all_reach_edges),
            senders=all_reach_senders,  # type: ignore[operator]
            receivers=all_reach_receivers,  # type: ignore[operator]
            n_node=all_reach_n_node,  # type: ignore[operator]
            n_edge=all_reach_n_edge,  # type: ignore[operator]
            globals=None,
        )
        avoid_graphs = jraph.GraphsTuple(
            nodes=jax.tree.map(lambda x: np.squeeze(x, axis=-1), all_avoid_nodes),
            edges=jax.tree.map(lambda x: np.squeeze(x, axis=-1), all_avoid_edges),
            senders=all_avoid_senders,  # type: ignore[operator]
            receivers=all_avoid_receivers,  # type: ignore[operator]
            n_node=all_avoid_n_node,  # type: ignore[operator]
            n_edge=all_avoid_n_edge,  # type: ignore[operator]
            globals=None,
        )

        return cls(
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_graphs=reach_graphs,
            avoid_graphs=avoid_graphs,
            repeat_last=jnp.array(repeat_last_arr),
            last_index=jnp.zeros((num_states, max_seqs), dtype=jnp.int32),
        )


def _convert_to_arrays(
    graph_root: Node | EpsilonType | None,
    propositions: Sequence[str],
    max_nodes: int,
    max_edges: int,
):
    """Converts a single boolean formula graph to padded numpy arrays."""
    if graph_root is None:  # Padding graph
        # Return empty arrays; padding is handled by the caller.
        nodes = {
            "type_idx": np.array([], dtype=np.int32).reshape(0, 1),
            "prop_idx": np.array([], dtype=np.int32).reshape(0, 1),
            "mask": np.array([], dtype=np.bool_).reshape(0, 1),
        }
        edges = {"mask": np.array([], dtype=np.bool_).reshape(0, 1)}
        senders = np.array([], dtype=np.int32)
        receivers = np.array([], dtype=np.int32)
        n_node = np.array(0)
        n_edge = np.array(0)
        return nodes, edges, senders, receivers, n_node, n_edge

    node_map: dict[Node | EpsilonType, int] = {}
    node_features: list[list[int]] = []
    senders, receivers = [], []

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

    _ = build_graph(graph_root)

    num_nodes = len(node_features)
    num_edges = len(senders)

    if num_nodes > max_nodes:
        raise ValueError(
            f"Graph has {num_nodes} nodes, exceeding max_nodes={max_nodes}"
        )
    if num_edges > max_edges:
        raise ValueError(
            f"Graph has {num_edges} edges, exceeding max_edges={max_edges}"
        )

    # Create node feature arrays (unpadded)
    type_arr = -np.ones((num_nodes, 1), dtype=np.int32)
    prop_arr = -np.ones((num_nodes, 1), dtype=np.int32)
    node_mask_arr = np.zeros((num_nodes, 1), dtype=np.bool_)

    if num_nodes > 0:
        features_arr = np.array(node_features, dtype=np.int32)
        type_arr[:, 0] = features_arr[:, 0]
        prop_arr[:, 0] = features_arr[:, 1]
        node_mask_arr[:, 0] = True

    nodes = {"type_idx": type_arr, "prop_idx": prop_arr, "mask": node_mask_arr}
    edges = {"mask": np.ones((num_edges, 1), dtype=np.bool_)}

    # Create edge arrays (unpadded)
    senders_arr = np.array(senders, dtype=np.int32)
    receivers_arr = np.array(receivers, dtype=np.int32)

    return (
        nodes,
        edges,
        senders_arr,
        receivers_arr,
        np.array(num_nodes),
        np.array(num_edges),
    )


def _get_node_features_as_int(
    node: Node | EpsilonType, propositions: Sequence[str]
) -> list[int]:
    """Creates an integer feature vector for a graph node.
    Returns:
        A list of two integers: [type_id, prop_id].
        type_id is -1 for non-special nodes.
        prop_id is -1 for non-proposition nodes.
    """
    prop_id = -1
    if isinstance(node, VarNode):
        type_id = -1
        try:
            prop_id = propositions.index(node.name)
        except ValueError as err:
            raise ValueError(
                f"Proposition '{node.name}' not in environment propositions."
            ) from err
    elif isinstance(node, AndNode | MultiAndNode):
        type_id = NODE_TYPE_AND
    elif isinstance(node, OrNode | MultiOrNode):
        type_id = NODE_TYPE_OR
    elif isinstance(node, NotNode):
        type_id = NODE_TYPE_NOT
    elif isinstance(node, EmptyNode):
        type_id = NODE_TYPE_EMPTY
    elif isinstance(node, EpsilonType):
        type_id = NODE_TYPE_EPSILON
    else:
        raise TypeError(f"Unknown node type for featurization: {type(node)}")

    return [type_id, prop_id]
