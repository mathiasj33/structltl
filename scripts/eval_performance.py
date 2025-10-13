import math
import time

import jax

import jaxltl


def main():
    num_steps = int(1e6)
    num_envs = 2048
    num_batch_steps = math.ceil(num_steps / num_envs)
    seeds = jax.numpy.arange(5)

    env, params = jaxltl.make("ZoneEnv")

    def run(seed: jax.Array):
        keys = jax.random.split(jax.random.PRNGKey(seed), num_envs + 1)
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        states, _ = vmap_reset(keys[1:], params)

        def step(key, states):
            keys = jax.random.split(key, 2 * num_envs)
            sample_keys = keys[:num_envs]
            actions = jax.vmap(env.action_space().sample)(sample_keys)
            step_keys = keys[num_envs:]
            transition = vmap_step(step_keys, states, actions, params)
            return transition.state, transition.observation

        def scan_step(carry, _):
            key, states = carry
            next_states, obss = step(key, states)
            return (key, next_states), obss

        final_state, final_obs = jax.lax.scan(
            scan_step,
            (keys[0], states),
            None,
            length=num_batch_steps,
        )
        return final_state, final_obs

    vmapped = jax.vmap(run, in_axes=(0,))
    start = time.time()
    compiled = jax.jit(vmapped).trace(seeds).lower().compile()
    end = time.time()
    print(f"Compilation took {end - start:.2f} seconds")
    start = time.time()
    _, observations = jax.block_until_ready(compiled(seeds))
    end = time.time()
    print(f"Ran {num_steps} steps in {end - start:.2f} seconds")
    print(observations.features.shape)


if __name__ == "__main__":
    main()
