import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.zone_env_samplers import ZoneReachAvoidSampler


def test_reach_avoid_sampler():
    sampler = ZoneReachAvoidSampler(
        depth=(3, 8),
        reach=(1, 3),
        avoid=(0, 2),
        num_assignments=4,
        max_length=8,
    )
    key = jax.random.key(0)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        seq = sampler.sample(subkey)

        # Check shapes
        assert seq.reach.shape == (sampler.max_length, sampler.num_assignments + 1)
        assert seq.avoid.shape == (sampler.max_length, sampler.num_assignments + 1)

        # Check depth constraints
        padding_col = seq.reach[:, -1]
        depth = int(jnp.sum(~padding_col))
        assert sampler.depth[0] <= depth <= sampler.depth[1]

        for i in range(depth):
            # Check reach set size constraints
            reach_set = seq.reach[i, :-1]
            num_reach = int(jnp.sum(reach_set))
            assert sampler.reach[0] <= num_reach <= sampler.reach[1]

            # Check avoid set size constraints
            avoid_set = seq.avoid[i, :-1]
            num_avoid = int(jnp.sum(avoid_set))
            assert sampler.avoid[0] <= num_avoid <= sampler.avoid[1]

            # Check reach-avoid disjointness
            assert jnp.all((reach_set & seq.avoid[i, :-1]) == 0)

            # Check reach-last reach disjointness
            if i > 0:
                last_reach_set = seq.reach[i - 1, :-1]
                assert jnp.all((reach_set & last_reach_set) == 0)
