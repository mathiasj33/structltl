import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

import jaxltl
from jaxltl import eqx_utils
from jaxltl.deep_ltl.curriculum.curriculum import Curriculum
from jaxltl.deep_ltl.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxltl.environments import environment
from jaxltl.environments.renderer.renderer import BaseRenderer
from jaxltl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.rl.actor_critic import ActorCritic


@hydra.main(version_base="1.1", config_path="../conf", config_name="test")
def main(cfg: DictConfig):

    default_options = None
    if "default_options" in cfg.env:
        # Instantiate the default_options object from config.
        # The fields will be standard python types (e.g., lists).
        default_options_with_lists = hydra.utils.instantiate(cfg.env.default_options)

        # Convert all leaf elements (the lists) in the pytree to jax arrays.
        default_options = jax.tree.map(
            lambda x: jnp.array(x, dtype=jnp.float32), default_options_with_lists
        )

    env, params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        env = PrecomputedResetWrapper(
            env,
            params,
            jaxltl.DATA_DIR / cfg.env.name / cfg.env.precomputed_resets_path,
        )
    curriculum: Curriculum = hydra.utils.call(cfg.curriculum)
    env = CurriculumWrapper(env, curriculum)
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )

    if cfg.policy == "model":

        model = build_model(
            cfg.model,
            obs_dim=env.observation_space(params).shape[0],
            action_dim=env.action_space(params).shape[0],
            num_assignments=5,
            key=jax.random.key(0),
        )
        model = eqx_utils.load(f"runs/{cfg.env.name}/tmp/model.eqx", model)

        @jax.jit
        def model_policy(obs: environment.EnvObservation, key: jax.Array) -> jax.Array:
            batch_obs = jax.tree.map(lambda x: x[None, ...], obs)
            dist = model.get_action(batch_obs)
            # action = dist.sample(seed=key).squeeze(0)
            action = dist.mean().squeeze(0)
            return action

        policy = model_policy

    elif cfg.policy == "random":

        @jax.jit
        def random_policy(obs: environment.EnvObservation, key: jax.Array) -> jax.Array:
            return env.action_space(params).sample(key)

        policy = random_policy

    elif cfg.policy == "teleop":

        policy = None

    else:
        raise ValueError(f"Unknown policy type: {cfg.policy}")

    renderer: BaseRenderer = env.get_renderer(params)
    renderer.run_render_loop(
        env,
        params,
        options=None,
        policy=policy,
        time_scale=2,
        print_debug=cfg.print_debug,
    )


def build_model(
    model_cfg: DictConfig,
    obs_dim: int,
    action_dim: int,
    num_assignments: int,
    key: jax.Array,
) -> ActorCritic:
    model: ActorCritic = hydra.utils.instantiate(
        model_cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_assignments=num_assignments,
        key=key,
    )
    return model


if __name__ == "__main__":
    main()
