import equinox as eqx
import jax

import jaxltl
from jaxltl.deep_ltl.model import DeepLTLModel
from jaxltl.deep_ltl.samplers.sequence_sampler import ReachSampler
from jaxltl.deep_ltl.wrappers.sequence_wrapper import SequenceWrapper
from jaxltl.environments import environment
from jaxltl.environments.renderer.renderer import BaseRenderer
from jaxltl.environments.wrappers.auto_reset_wrapper import AutoResetWrapper


def main():
    env, params = jaxltl.make("ZoneEnv")
    sampler = ReachSampler(num_propositions=4, max_length=5)
    env = SequenceWrapper(env, sampler)
    env = AutoResetWrapper(env, reset_to_initial_state=False)

    model = DeepLTLModel(
        env.observation_space(params).shape[0],
        env.action_space(params).shape[0],
        jax.nn.tanh,
        embedding_dim=32,
        num_propositions=4,
        key=jax.random.key(0),
    )
    model = eqx.tree_deserialise_leaves("runs/ZoneEnv/tmp2/model.eqx", model)

    @jax.jit
    def random_policy(obs: environment.EnvObservation, key: jax.Array) -> jax.Array:
        return env.action_space(params).sample(key)

    @jax.jit
    def model_policy(obs: environment.EnvObservation, key: jax.Array) -> jax.Array:
        batch_obs = jax.tree.map(lambda x: x[None, ...], obs)
        dist, _ = model(batch_obs)
        action = dist.sample(seed=key).squeeze(0)
        # action = dist.mean().squeeze(0)
        return action

    renderer: BaseRenderer = env.get_renderer(params)
    renderer.run_render_loop(env, params, policy=model_policy)


if __name__ == "__main__":
    main()
