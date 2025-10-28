import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.zone_env_samplers import ZoneReachAvoidSampler


def test_reach_avoid_sampler():
    sampler = ZoneReachAvoidSampler(
        depth=(3, 8),
        reach=(1, 3),
        avoid=(0, 2),
        num_assignments=4,
        max_length=9,
    )
    key = jax.random.key(0)
    for j in range(100):
        key, subkey = jax.random.split(key)
        seq = sampler.sample(subkey)

        if j < 5:
            print(
                f"\nSampled Reach-Avoid Sequence ({j}):\nReach:\n{jax.device_get(seq.reach)}\nAvoid:\n{jax.device_get(seq.avoid)}"
            )

        # Check shapes
        assert seq.reach.shape == (sampler.max_length, sampler.num_assignments)
        assert seq.avoid.shape == (sampler.max_length, sampler.num_assignments)

        # Check depth constraints
        num_depth = jnp.sum(seq.reach[:, 0] != -1)
        assert sampler.depth[0] <= num_depth <= sampler.depth[1]

        for i in range(num_depth):
            # Check reach set size constraints
            reach_set = seq.reach[i]
            num_reach = jnp.sum(reach_set != -1)
            assert sampler.reach[0] <= num_reach <= sampler.reach[1]

            # Check avoid set size constraints
            avoid_set = seq.avoid[i]
            num_avoid = jnp.sum(avoid_set != -1)
            assert sampler.avoid[0] <= num_avoid <= sampler.avoid[1]

            # Check reach-avoid disjointness
            assert len(set(reach_set.tolist()) & set(avoid_set.tolist()) - {-1}) == 0

            # Check reach-last reach disjointness
            if i > 0:
                last_reach_set = seq.reach[i - 1]
                assert (
                    len(set(reach_set.tolist()) & set(last_reach_set.tolist()) - {-1})
                    == 0
                )

        for i in range(num_depth, seq.reach.shape[0]):
            # Check padding
            assert jnp.all(seq.reach[i] == -1)
            assert jnp.all(seq.avoid[i] == -1)
