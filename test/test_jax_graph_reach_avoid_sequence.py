import jax

from jaxltl.deep_ltl.curriculum.zone_env_graph_samplers import (
    GraphZoneReachAvoidSampler,
    GraphZoneReachStaySampler,
)
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.environments.zone_env.zone_env import ZoneEnv

# TODO: Add assertions


def test_sampler_to_jax_and_advance():
    env = ZoneEnv()
    sampler = GraphZoneReachAvoidSampler(
        depth=(1, 2),
        reach=(1, 2),
        avoid=(0, 2),
        propositions=env.propositions,
        assignments=env.assignments,
        max_length=3,
    )
    key = jax.random.key(0)

    for j in range(50):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample(subkey)

        # Convert to JAX version
        max_nodes, max_edges = 10, 10
        jax_seq = JaxGraphReachAvoidSequence.from_seq(
            graph_seq, env, max_nodes, max_edges
        )

        if j < 5:
            print(f"\n\n--- Reach-Avoid Sample {j + 1} ---")
            print(f"Assignments: {graph_seq}")
            print("\n--- Initial JAX Assignments (Reach) ---")
            print(jax_seq.reach)
            print("\n--- Initial JAX Graph (Reach) ---")
            print(jax_seq.reach_graphs)
            print("\n--- Initial JAX Assignments (Avoid) ---")
            print(jax_seq.avoid)
            print("\n--- Initial JAX Graph (Avoid) ---")
            print(jax_seq.avoid_graphs)

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        if j < 5:
            print("\n--- Advanced JAX Assignments (Reach) ---")
            print(advanced_seq.reach)
            print("\n--- Advanced JAX Graph (Reach) ---")
            print(advanced_seq.reach_graphs)
            print("\n--- Advanced JAX Assignments (Avoid) ---")
            print(advanced_seq.avoid)
            print("\n--- Advanced JAX Graph (Avoid) ---")
            print(advanced_seq.avoid_graphs)

        # Second advance
        advanced_seq = advanced_seq.advance()


def test_reach_stay_sampler_to_jax_and_advance():
    env = ZoneEnv()
    sampler = GraphZoneReachStaySampler(
        num_stay=60,
        avoid=(0, 2),
        propositions=env.propositions,
        assignments=env.assignments,
        max_length=3,
    )
    key = jax.random.key(42)

    for j in range(50):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample(subkey)

        # Convert to JAX version
        max_nodes, max_edges = 10, 10
        jax_seq = JaxGraphReachAvoidSequence.from_seq(
            graph_seq, env, max_nodes, max_edges
        )

        if j < 5:
            print(f"\n\n--- Reach-Avoid Sample {j + 1} ---")
            print(f"Assignments: {graph_seq}")
            print("\n--- Initial JAX Assignments (Reach) ---")
            print(jax_seq.reach)
            print("\n--- Initial JAX Graph (Reach) ---")
            print(jax_seq.reach_graphs)
            print("\n--- Initial JAX Assignments (Avoid) ---")
            print(jax_seq.avoid)
            print("\n--- Initial JAX Graph (Avoid) ---")
            print(jax_seq.avoid_graphs)

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        if j < 5:
            print("\n--- Advanced JAX Assignments (Reach) ---")
            print(advanced_seq.reach)
            print("\n--- Advanced JAX Graph (Reach) ---")
            print(advanced_seq.reach_graphs)
            print("\n--- Advanced JAX Assignments (Avoid) ---")
            print(advanced_seq.avoid)
            print("\n--- Advanced JAX Graph (Avoid) ---")
            print(advanced_seq.avoid_graphs)

        # Second advance
        advanced_seq = advanced_seq.advance()


def test_batch_to_jax_and_advance():
    env = ZoneEnv()
    sampler = GraphZoneReachAvoidSampler(
        depth=(2, 2),
        reach=(1, 2),
        avoid=(0, 2),
        propositions=env.propositions,
        assignments=env.assignments,
        max_length=3,
    )
    key = jax.random.key(0)

    graph_seqs = []
    for _ in range(6):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample(subkey)
        graph_seqs.append(graph_seq)

    # Convert to JAX version
    state_to_seqs = {
        0: [graph_seqs[0], graph_seqs[1], graph_seqs[2]],
        1: [graph_seqs[3], graph_seqs[4], graph_seqs[5]],
    }
    max_nodes, max_edges = 10, 10
    jax_seq = JaxGraphReachAvoidSequence.from_state_to_seqs(
        state_to_seqs, env, max_nodes, max_edges
    )

    print("\n\n--- Batched Reach-Avoid Sample ---")
    print(f"Assignments: {graph_seq}")
    print("\n--- Initial JAX Assignments (Reach) ---")
    print(jax_seq.reach)
    print("\n--- Initial JAX Graph (Reach) ---")
    print(jax_seq.reach_graphs)
    print("\n--- Initial JAX Assignments (Avoid) ---")
    print(jax_seq.avoid)
    print("\n--- Initial JAX Graph (Avoid) ---")
    print(jax_seq.avoid_graphs)
