"""Main training entry point.

Uses Hydra for configuration management. See conf/train.yaml, or run
`python train.py --help` for details.
"""

import logging
import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import pandas as pd
from omegaconf import DictConfig

import jaxltl
from jaxltl import DATA_DIR, eqx_utils
from jaxltl.deep_ltl.curriculum.curriculum import Curriculum
from jaxltl.deep_ltl.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxltl.environments.wrappers import AutoResetWrapper, LogWrapper, VectorizeWrapper
from jaxltl.environments.wrappers.auto_reset_wrapper import ResetStrategy
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.eqx_utils.training import ensemble_to_list
from jaxltl.rl.actor_critic import ActorCritic
from jaxltl.rl.algorithm import RLAlgorithm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if not cfg.use_gpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Using CPU for training")

    default_options = None
    if "default_options" in cfg.env:
        # Instantiate the default_options object from config.
        # The fields will be standard python types (e.g., lists).
        default_options_with_lists = hydra.utils.instantiate(cfg.env.default_options)

        # Convert all leaf elements (the lists) in the pytree to jax arrays.
        default_options = jax.tree.map(
            lambda x: jnp.array(x, dtype=jnp.float32), default_options_with_lists
        )

    env, env_params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = f"{DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    curriculum: Curriculum = hydra.utils.call(cfg.curriculum)
    env = CurriculumWrapper(
        env, curriculum, episode_window=cfg.curriculum_wrapper.episode_window
    )
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )
    env = LogWrapper(env)
    env = VectorizeWrapper(env)

    seeds = jnp.arange(cfg.num_seeds)
    keys = jax.vmap(jax.random.key)(seeds)
    split = jax.vmap(jax.random.split)(keys)
    keys, model_keys = split[:, 0], split[:, 1]

    make_models = eqx.filter_vmap(build_model, in_axes=(None, None, None, None, 0))
    models = make_models(
        cfg.model,
        env.observation_space(env_params).shape[0],
        env.action_space(env_params).shape[0],
        env.assignments.shape[0],
        model_keys,
    )

    start_time = time.time()

    def callback(metric, seed, step):
        seconds = time.time() - start_time
        metric["total_step"] = metric["total_step"] * cfg.rl_alg.num_envs
        last_return = metric["episode_return"][metric["done"]][-1]
        last_step = metric["total_step"][metric["done"]][-1]
        logger.info(
            f"Seed {seed}. Step {step}. SPS: {step / seconds:.2f}. Return: {last_return}. Last Step: {last_step}"
        )

    rl_alg: RLAlgorithm = hydra.utils.instantiate(cfg.rl_alg)
    train = eqx.filter_vmap(
        rl_alg.train, in_axes=(eqx.if_array(0), None, None, 0, None, None, 0)
    )
    train = eqx.filter_jit(train)

    logger.info("Starting training")
    models, metrics = jax.block_until_ready(
        train(models, env, env_params, keys, callback, 1_000_000, seeds)
    )
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    dfs = []
    for seed in range(cfg.num_seeds):
        seed_metrics = jax.tree.map(lambda x: x[seed], metrics)
        return_values = seed_metrics["episode_return"][seed_metrics["done"]].tolist()
        lengths = seed_metrics["episode_length"][seed_metrics["done"]].tolist()
        stages = seed_metrics["curriculum_stage"][seed_metrics["done"]].tolist()
        timesteps = (
            seed_metrics["total_step"][seed_metrics["done"]] * cfg.rl_alg.num_envs
        ).tolist()
        df = pd.DataFrame(
            {
                "timestep": timesteps,
                "return": return_values,
                "length": lengths,
                "curriculum_stage": stages,
            }
        )
        df["seed"] = seed
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("logs.csv", index=False)

    model = ensemble_to_list(models, cfg.num_seeds)[0]
    eqx_utils.save("model.eqx", model)
    logger.info("Model saved to model.eqx")


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
