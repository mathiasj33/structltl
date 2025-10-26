import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.sampling_utils import sample_assignments


def count_true(arr):
    return int(jnp.sum(arr))


def test_no_available_returns_all_false():
    key = jax.random.key(0)
    available_mask = jnp.array([False, False, False], dtype=bool)
    reach_mask = sample_assignments(available_mask, (1, 3), key)
    assert jnp.all(reach_mask == jnp.array([False, False, False]))


def test_fixed_nr_all_available_counts_correct():
    key = jax.random.key(1)
    available_mask = jnp.ones(5, dtype=bool)
    reach_mask = sample_assignments(available_mask, (2, 2), key)
    assert count_true(reach_mask) == 2


def test_nr_greater_than_available_selects_all_available():
    key = jax.random.key(2)
    available_mask = jnp.array([True, False, True, False], dtype=bool)  # 2 available
    reach_mask = sample_assignments(
        available_mask, (3, 3), key
    )  # nr=3 but only 2 available
    assert jnp.array_equal(reach_mask, available_mask)


def test_deterministic_for_same_key():
    key = jax.random.key(42)
    available_mask = jnp.array([True, True, True, True], dtype=bool)
    m1 = sample_assignments(available_mask, (1, 3), key)
    m2 = sample_assignments(available_mask, (1, 3), key)
    assert jnp.array_equal(m1, m2)


def test_selection_counts_within_bounds_and_available():
    rng = jax.random.key(100)
    available_mask = jnp.array([True, True, True, False, False], dtype=bool)
    num_avail = int(jnp.sum(available_mask))
    min_r, max_r = 1, 4
    for _ in range(30):
        rng, subkey = jax.random.split(rng)
        mask = sample_assignments(available_mask, (min_r, max_r), subkey)
        k = count_true(mask)
        assert 0 <= k <= num_avail
        assert k <= max_r
        assert count_true(mask & ~available_mask) == 0


def test_zero_sample_range():
    key = jax.random.key(7)
    available_mask = jnp.array([True, True, True], dtype=bool)
    mask = sample_assignments(available_mask, (0, 0), key)  # nr=0
    assert count_true(mask) == 0
