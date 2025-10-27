import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.curriculum import (
    ReachAvoidSequence,
    SequenceSampler,
)
from jaxltl.deep_ltl.curriculum.sampling_utils import sample_assignments


class ZoneReachAvoidSampler(SequenceSampler):
    """Samples simple reach-avoid sequences."""

    depth: tuple[int, int]
    reach: tuple[int, int]
    avoid: tuple[int, int]

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        *,
        num_assignments: int,
        max_length: int,
    ):
        super().__init__(num_assignments, max_length)
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.depth = depth
        self.reach = reach
        self.avoid = avoid

    @eqx.filter_jit
    def sample(self, key: jax.Array) -> ReachAvoidSequence:
        key, depth_key = jax.random.split(key)
        depth = jax.random.randint(depth_key, (), self.depth[0], self.depth[1] + 1)

        # 1. Pre-allocate output arrays (filled with padding)
        reach_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)
        avoid_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)

        def body_fn(i, carry):
            """
            This function is executed for each step of the jax.lax.fori_loop.
            `i` is the loop index (from 0 to depth-1).
            `carry` holds the state: (key, last_reach_mask, reach_seq, avoid_seq)
            """
            key, last_reach_props_mask, reach_seq_carry, avoid_seq_carry = carry

            key, reach_key, avoid_key = jax.random.split(key, 3)

            # --- 1. Sample Reach Set ---
            reach_mask = sample_assignments(
                ~last_reach_props_mask, self.reach, reach_key
            )
            new_last_reach_props_mask = reach_mask

            # --- 2. Sample Avoid Set ---
            available_avoid_mask = ~reach_mask & ~last_reach_props_mask
            num_available_avoid = jnp.sum(available_avoid_mask)
            na_min_clamped = jnp.minimum(self.avoid[0], num_available_avoid)
            na_max_clamped = jnp.minimum(self.avoid[1], num_available_avoid)

            avoid_mask = sample_assignments(
                available_avoid_mask, (na_min_clamped, na_max_clamped), avoid_key
            )

            # --- 3. Update the output arrays at index `i` ---
            reach = jnp.nonzero(reach_mask, size=self.num_assignments, fill_value=-1)[0]
            reach = jnp.sort(reach, descending=True)
            avoid = jnp.nonzero(avoid_mask, size=self.num_assignments, fill_value=-1)[0]
            avoid = jnp.sort(avoid, descending=True)
            reach_seq_new = reach_seq_carry.at[i].set(reach)
            avoid_seq_new = avoid_seq_carry.at[i].set(avoid)

            new_carry = (key, new_last_reach_props_mask, reach_seq_new, avoid_seq_new)
            return new_carry

        # 2. Define the initial state for the loop
        initial_carry = (
            key,
            jnp.zeros(self.num_assignments, dtype=bool),  # initial last_reach_mask
            reach_seq,  # initial (empty) reach_seq
            avoid_seq,  # initial (empty) avoid_seq
        )

        # 3. Run the loop from 0 up to (but not including) `depth`
        final_carry = jax.lax.fori_loop(0, depth, body_fn, initial_carry)

        # 4. Extract the final arrays from the carry
        _, _, final_reach_seq, final_avoid_seq = final_carry

        return ReachAvoidSequence(reach=final_reach_seq, avoid=final_avoid_seq)
