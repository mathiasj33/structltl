import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jaxltl.deep_ltl.curriculum.zone_env_graph_samplers import (
    GraphZoneReachAvoidSampler,
    GraphZoneReachStaySampler,
)
from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON
from jaxltl.environments.zone_env.zone_env import ZoneEnv
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import MultiOrNode, Node, NotNode, VarNode


@functools.lru_cache(maxsize=1024)
def _compute_satisfying_assignments(
    graph: Node | None, all_assignments: tuple[Assignment, ...]
) -> frozenset[Assignment]:
    """Computes the set of assignments that satisfy a given boolean formula."""
    if graph is None:
        return frozenset()
    return frozenset(a for a in all_assignments if graph.eval(a))


def assert_ragged_set_equal(actual, expected, pad_val=-1):
    """Asserts that two dense arrays representing ragged sets are equal."""
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    npt.assert_equal(
        actual_np.shape, expected_np.shape, "Arrays must have the same shape."
    )
    actual_mask = actual_np != pad_val
    expected_mask = expected_np != pad_val
    npt.assert_array_equal(
        actual_mask, expected_mask, "Padding structure (mask) must be identical."
    )
    for i in range(actual_np.shape[0]):
        actual_valid = actual_np[i][actual_mask[i]]
        expected_valid = expected_np[i][expected_mask[i]]
        npt.assert_array_equal(
            np.sort(actual_valid),
            np.sort(expected_valid),
            f"Row {i} non-padded elements do not match: {actual_np[i]} vs {expected_np[i]}",
        )


def test_from_state_to_seqs_and_advance():
    env = ZoneEnv()
    all_assignments = tuple(env.assignments)

    # 1. Manually create a GraphReachAvoidSequence
    # Step 0: Reach (green | red), Avoid yellow
    # Step 1: Reach purple, Avoid Not(purple)
    # Step 2: Reach EPSILON, Avoid green
    g0 = VarNode("green")
    r0 = VarNode("red")
    y0 = VarNode("yellow")
    p1 = VarNode("purple")
    g2 = VarNode("green")

    reach_graph_0 = MultiOrNode([g0, r0])
    avoid_graph_0 = y0
    reach_graph_1 = p1
    avoid_graph_1 = NotNode(p1)
    reach_graph_2 = EPSILON
    avoid_graph_2 = g2

    graph_seq = GraphReachAvoidSequence(
        reach_avoid_assignments=[
            (
                _compute_satisfying_assignments(reach_graph_0, all_assignments),
                _compute_satisfying_assignments(avoid_graph_0, all_assignments),
            ),
            (
                _compute_satisfying_assignments(reach_graph_1, all_assignments),
                _compute_satisfying_assignments(avoid_graph_1, all_assignments),
            ),
            (
                EPSILON,
                _compute_satisfying_assignments(avoid_graph_2, all_assignments),
            ),
        ],
        reach_avoid_graphs=[
            (reach_graph_0, avoid_graph_0),
            (reach_graph_1, avoid_graph_1),
            (reach_graph_2, avoid_graph_2),
        ],
    )

    state_to_seqs = {0: [graph_seq]}
    max_nodes, max_edges = 10, 10

    # 2. Convert to JAX version
    jax_seq = JaxGraphReachAvoidSequence.from_state_to_seqs(
        state_to_seqs, env, max_nodes, max_edges
    )

    # Get the unbatched view for easier inspection
    reach_assignments_unbatched = jax_seq.reach_assignments[0, 0]
    avoid_assignments_unbatched = jax_seq.avoid_assignments[0, 0]
    reach_nodes_unbatched = jax_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]

    print("\n--- Initial JAX Sequence ---")
    print("Reach Assignments:\n", reach_assignments_unbatched)
    print("Avoid Assignments:\n", avoid_assignments_unbatched)
    print("Reach Graph Nodes (first step):\n", reach_nodes_unbatched[0])  # type: ignore[operator]

    # 3. Assert initial conversion is correct
    # Expected assignment indices from ZoneEnv
    # red=0, green=1, purple=2, yellow=3
    # The default ZoneEnv has max_assignments=5, so the width is 5.
    # Epsilon is index len(assignments) = 5
    expected_reach_assigns = jnp.array(
        [
            [0, 1, -1, -1, -1],  # red, green
            [2, -1, -1, -1, -1],  # purple
            [5, -1, -1, -1, -1],  # Epsilon
        ],
        dtype=jnp.int32,
    )
    expected_avoid_assigns = jnp.array(
        [
            [3, -1, -1, -1, -1],  # yellow
            [0, 1, 3, 4, -1],  # not purple -> red, green, yellow
            [1, -1, -1, -1, -1],  # green
        ],
        dtype=jnp.int32,
    )
    assert_ragged_set_equal(reach_assignments_unbatched, expected_reach_assigns)
    assert_ragged_set_equal(avoid_assignments_unbatched, expected_avoid_assigns)

    # 4. Test advance function
    advanced_seq = jax_seq.advance()
    advanced_reach_unbatched = advanced_seq.reach_assignments[0, 0]
    advanced_avoid_unbatched = advanced_seq.avoid_assignments[0, 0]
    advanced_reach_nodes_unbatched = advanced_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]

    print("\n--- Advanced JAX Sequence ---")
    print("Reach Assignments:\n", advanced_reach_unbatched)
    print("Avoid Assignments:\n", advanced_avoid_unbatched)
    print(
        "Reach Graph Nodes (first step):\n",
        advanced_reach_nodes_unbatched[0],  # type: ignore[operator]
    )

    # 5. Assert advanced sequence is correct
    expected_advanced_reach = jnp.array(
        [
            [2, -1, -1, -1, -1],
            [5, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    expected_advanced_avoid = jnp.array(
        [
            [0, 1, 3, 4, -1],
            [1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    assert_ragged_set_equal(advanced_reach_unbatched, expected_advanced_reach)
    assert_ragged_set_equal(advanced_avoid_unbatched, expected_advanced_avoid)

    # Check that graph nodes have also been rolled and padded
    # The first step's nodes should now be the second step's from the original
    original_step1_nodes = reach_nodes_unbatched[1]  # type: ignore[operator]
    advanced_step0_nodes = advanced_reach_nodes_unbatched[0]  # type: ignore[operator]
    npt.assert_array_equal(original_step1_nodes, advanced_step0_nodes)

    # The last step's nodes should now be padded with -1
    last_step_nodes = advanced_reach_nodes_unbatched[-1]  # type: ignore[operator]
    assert np.all(last_step_nodes == -1)


def test_sampler_to_jax_and_advance():
    env = ZoneEnv()
    sampler = GraphZoneReachAvoidSampler(
        depth=(1, 3),
        reach=(1, 3),
        avoid=(0, 3),
        propositions=env.propositions,
        assignments=env.assignments,
    )
    key = jax.random.key(0)

    for j in range(50):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample(subkey)

        if j < 5:
            print(f"\n\n--- Reach-Avoid Sample {j + 1} ---")
            print(f"Assignments: {graph_seq}")
            print("Graphs:")
            for i, (rg, ag) in enumerate(graph_seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")

        # Convert to JAX version
        state_to_seqs = {0: [graph_seq]}
        max_nodes, max_edges = 10, 10
        jax_seq = JaxGraphReachAvoidSequence.from_state_to_seqs(
            state_to_seqs, env, max_nodes, max_edges
        )

        # --- Print Initial JAX Sequence ---
        reach_assigns = jax_seq.reach_assignments[0, 0]
        avoid_assigns = jax_seq.avoid_assignments[0, 0]
        reach_nodes = jax_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]
        avoid_nodes = jax_seq.avoid_graphs.nodes[0, 0]  # type: ignore[operator]

        if j < 5:
            print("\n--- Initial JAX Sequence ---")
            print("Reach Assignments:\n", reach_assigns)
            print("Avoid Assignments:\n", avoid_assigns)
            print("Reach Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{reach_nodes[i]}")  # type: ignore[operator]
            print(
                "Reach Graph Edges (n_edge per step):",
                jax_seq.reach_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )
            print("Avoid Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{avoid_nodes[i]}")  # type: ignore[operator]
            print(
                "Avoid Graph Edges (n_edge per step):",
                jax_seq.avoid_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        # --- Print Advanced JAX Sequence ---
        adv_reach_assigns = advanced_seq.reach_assignments[0, 0]
        adv_avoid_assigns = advanced_seq.avoid_assignments[0, 0]
        adv_reach_nodes = advanced_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]
        adv_avoid_nodes = advanced_seq.avoid_graphs.nodes[0, 0]  # type: ignore[operator]
        if j < 5:
            print("\n--- Advanced JAX Sequence ---")
            print("Reach Assignments:\n", adv_reach_assigns)
            print("Avoid Assignments:\n", adv_avoid_assigns)
            print("Reach Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{adv_reach_nodes[i]}")  # type: ignore[operator]
            print(
                "Reach Graph Edges (n_edge per step):",
                advanced_seq.reach_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

            print("Avoid Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{adv_avoid_nodes[i]}")  # type: ignore[operator]
            print(
                "Avoid Graph Edges (n_edge per step):",
                advanced_seq.avoid_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

        # --- Assertions ---
        # Check that assignments are rolled
        npt.assert_array_equal(adv_reach_assigns[:-1], reach_assigns[1:])
        npt.assert_array_equal(adv_avoid_assigns[:-1], avoid_assigns[1:])

        # Check that graph nodes and edge counts are rolled
        npt.assert_array_equal(adv_reach_nodes[:-1], reach_nodes[1:])  # type: ignore[operator]
        npt.assert_array_equal(
            advanced_seq.reach_graphs.n_edge[..., :-1],
            jax_seq.reach_graphs.n_edge[..., 1:],
        )

        # Check that the last step is padded
        assert np.all(adv_reach_assigns[-1] == -1)
        assert np.all(adv_reach_nodes[-1] == -1)  # type: ignore[operator]
        assert np.all(advanced_seq.reach_graphs.n_edge[..., -1] == 0)
        assert np.all(advanced_seq.avoid_graphs.n_edge[..., -1] == 0)


def test_reach_stay_sampler_to_jax_and_advance():
    env = ZoneEnv()
    sampler = GraphZoneReachStaySampler(
        num_stay=3,
        avoid=(0, 2),
        propositions=env.propositions,
        assignments=env.assignments,
    )
    key = jax.random.key(42)

    for j in range(50):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample(subkey)

        if j < 5:
            print(f"\n\n--- Reach-Stay Sample {j + 1} ---")
            print(f"Assignments: {graph_seq}")
            print("Graphs:")
            for i, (rg, ag) in enumerate(graph_seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")

        # Convert to JAX version
        state_to_seqs = {0: [graph_seq]}
        max_nodes, max_edges = 10, 10
        jax_seq = JaxGraphReachAvoidSequence.from_state_to_seqs(
            state_to_seqs, env, max_nodes, max_edges
        )

        # --- Print Initial JAX Sequence ---
        reach_assigns = jax_seq.reach_assignments[0, 0]
        avoid_assigns = jax_seq.avoid_assignments[0, 0]
        reach_nodes = jax_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]
        avoid_nodes = jax_seq.avoid_graphs.nodes[0, 0]  # type: ignore[operator]

        if j < 5:
            print("\n--- Initial JAX Sequence ---")
            print("Reach Assignments:\n", reach_assigns)
            print("Avoid Assignments:\n", avoid_assigns)
            print("Reach Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{reach_nodes[i]}")  # type: ignore[operator]
            print(
                "Reach Graph Edges (n_edge per step):",
                jax_seq.reach_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

            print("Avoid Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{avoid_nodes[i]}")  # type: ignore[operator]
            print(
                "Avoid Graph Edges (n_edge per step):",
                jax_seq.avoid_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        # --- Print Advanced JAX Sequence ---
        adv_reach_assigns = advanced_seq.reach_assignments[0, 0]
        adv_avoid_assigns = advanced_seq.avoid_assignments[0, 0]
        adv_reach_nodes = advanced_seq.reach_graphs.nodes[0, 0]  # type: ignore[operator]
        adv_avoid_nodes = advanced_seq.avoid_graphs.nodes[0, 0]  # type: ignore[operator]

        if j < 5:
            print("\n--- Advanced JAX Sequence ---")
            print("Reach Assignments:\n", adv_reach_assigns)
            print("Avoid Assignments:\n", adv_avoid_assigns)
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{adv_reach_nodes[i]}")  # type: ignore[operator]
            print(
                "Reach Graph Edges (n_edge per step):",
                advanced_seq.reach_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

            print("Avoid Graph Nodes (per step):")
            for i in range(len(graph_seq)):
                print(f"  Step {i}:\n{adv_avoid_nodes[i]}")  # type: ignore[operator]
            print(
                "Avoid Graph Edges (n_edge per step):",
                advanced_seq.avoid_graphs.n_edge.reshape(1, 1, -1)[0, 0],
            )

        # --- Assertions ---
        # Check that assignments are rolled
        npt.assert_array_equal(adv_reach_assigns[:-1], reach_assigns[1:])
        npt.assert_array_equal(adv_avoid_assigns[:-1], avoid_assigns[1:])

        # Check that graph nodes and edge counts are rolled
        npt.assert_array_equal(adv_reach_nodes[:-1], reach_nodes[1:])  # type: ignore[operator]
        npt.assert_array_equal(
            advanced_seq.reach_graphs.n_edge[..., :-1],
            jax_seq.reach_graphs.n_edge[..., 1:],
        )
        npt.assert_array_equal(
            advanced_seq.avoid_graphs.n_edge[..., :-1],
            jax_seq.avoid_graphs.n_edge[..., 1:],
        )

        # Check that the last step is padded
        assert np.all(adv_reach_assigns[-1] == -1)
        assert np.all(adv_reach_nodes[-1] == -1)  # type: ignore[operator]
        assert np.all(advanced_seq.reach_graphs.n_edge[..., -1] == 0)
        assert np.all(advanced_seq.avoid_graphs.n_edge[..., -1] == 0)
