"""Utility functions for evaluation scripts."""

import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.reach_avoid import path_search
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl.automata import ltl2ldba
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from jaxltl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)


def preprocess_graph_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxGraphReachAvoidSequence]:
    """Preprocesses formulas into a batched JaxLDBA and batched JaxGraphReachAvoidSequence."""

    ldbas, seqs = [], []
    for formula in formulas:
        ldba, batched_seqs = _preprocess_graph_formula(formula, env)
        ldbas.append(ldba)
        seqs.append(batched_seqs)
    ldba = _batch_ldbas(ldbas)
    batched_seqs = _batch_graph_sequences(seqs)
    return ldba, batched_seqs


def _preprocess_graph_formula(
    formula: str, env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxGraphReachAvoidSequence]:
    """Preprocesses the formula into a JaxLDBA and batched JaxGraphReachAvoidSequence."""

    ldba = _build_ldba(formula, env)
    jldba = JaxLDBA.from_ldba(ldba, env)
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)

    # Convert assignment-based sequences to graph-based sequences
    state_to_graph_seqs = {}
    for state, seq_list in state_to_seqs.items():
        state_to_graph_seqs[state] = [
            BooleanReachAvoidSequence.from_reach_avoid_sequence(seq, env)
            for seq in seq_list
        ]

    batched_seqs = JaxGraphReachAvoidSequence.from_state_to_seqs(
        state_to_graph_seqs,
        env.propositions,
        env.assignments(),
        env.max_nodes,
        env.max_edges,
    )
    return jldba, batched_seqs


def _build_ldba(formula: str, env: Environment | EnvWrapper):
    ldba = ltl2ldba(formula, env.propositions)
    ldba.prune(env.assignments())
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _batch_ldbas(ldbas: list[JaxLDBA]) -> JaxLDBA:
    """Batch multiple JaxLDBAs into a single JaxLDBA with an added batch dimension."""

    num_states = jnp.array([ldba.num_states for ldba in ldbas], dtype=jnp.int32)
    max_num_states = jnp.max(num_states)
    batch_size = len(ldbas)
    num_assignments = ldbas[0].transitions.shape[1] - 1

    transitions = -jnp.ones(
        (batch_size, max_num_states, num_assignments + 1), dtype=jnp.int32
    )
    accepting = jnp.zeros((batch_size, max_num_states, num_assignments), dtype=bool)
    sink_states = jnp.zeros((batch_size, max_num_states), dtype=bool)
    initial_states = jnp.zeros((batch_size,), dtype=jnp.int32)

    for i, ldba in enumerate(ldbas):
        transitions = transitions.at[i, : ldba.num_states, :].set(ldba.transitions)
        accepting = accepting.at[i, : ldba.num_states, :].set(ldba.accepting)
        sink_states = sink_states.at[i, : ldba.num_states].set(ldba.sink_states)
        initial_states = initial_states.at[i].set(ldba.initial_state)

    return JaxLDBA(
        num_states=num_states,
        initial_state=initial_states,
        transitions=transitions,
        accepting=accepting,
        sink_states=sink_states,
        finite=jnp.array([ldba.finite for ldba in ldbas]),
    )


def _batch_graph_sequences(
    seqs: list[JaxGraphReachAvoidSequence],
) -> JaxGraphReachAvoidSequence:
    """Batch multiple JaxGraphReachAvoidSequences into a single JaxGraphReachAvoidSequence."""

    def pad_and_stack_pytree(pytrees: list):
        """Pads and stacks a list of pytrees."""
        # Find the max shape for each leaf array across all pytrees
        max_shapes = jax.tree.map(
            lambda *xs: jnp.max(jnp.array([x.shape for x in xs]), axis=0), *pytrees
        )

        def pad_leaf(leaf, max_shape):
            pad_config = tuple(
                (0, max_d - d) for d, max_d in zip(leaf.shape, max_shape, strict=True)
            )
            return jnp.pad(
                leaf, pad_width=pad_config, mode="constant", constant_values=-1
            )

        padded_pytrees = [
            jax.tree.map(pad_leaf, pytree, max_shapes) for pytree in pytrees
        ]

        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *padded_pytrees)

    # Batch the GraphTuple pytrees
    batched_reach_graphs = pad_and_stack_pytree([s.reach_graphs for s in seqs])
    batched_avoid_graphs = pad_and_stack_pytree([s.avoid_graphs for s in seqs])

    # For other fields, we pad and stack them individually.
    def pad_and_stack(get_field_fn):
        fields = [get_field_fn(s) for s in seqs]

        # Find max dimensions for this specific field
        max_shape = list(fields[0].shape)
        for field in fields[1:]:
            for i, dim in enumerate(field.shape):
                max_shape[i] = max(max_shape[i], dim)

        padded_fields = []
        for field in fields:
            pad_config = tuple(
                (0, max_d - d) for d, max_d in zip(field.shape, max_shape, strict=True)
            )
            padded_field = jnp.pad(
                field, pad_width=pad_config, mode="constant", constant_values=-1
            )
            padded_fields.append(padded_field)

        return jnp.stack(padded_fields, axis=0)

    batched_reach = pad_and_stack(lambda s: s.reach)
    batched_avoid = pad_and_stack(lambda s: s.avoid)
    batched_repeat_last = pad_and_stack(lambda s: s.repeat_last)
    batched_last_index = pad_and_stack(lambda s: s.last_index)

    # Reconstruct the batched sequence
    return JaxGraphReachAvoidSequence(
        reach=batched_reach,
        avoid=batched_avoid,
        reach_graphs=batched_reach_graphs,
        avoid_graphs=batched_avoid_graphs,
        repeat_last=batched_repeat_last,
        last_index=batched_last_index,
    )
