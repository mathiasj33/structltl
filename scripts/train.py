import logging
import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import pandas as pd
from omegaconf import DictConfig

import jaxltl
from jaxltl.deep_ltl.model import DeepLTLModel
from jaxltl.environments.wrappers import AutoResetWrapper, LogWrapper, VectorizeWrapper
from jaxltl.eqx_utils.training import ensemble_to_list
from jaxltl.rl.algorithm import RLAlgorithm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if not cfg.use_gpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Using CPU for training")

    env, env_params = jaxltl.make(cfg.env)
    env = AutoResetWrapper(env, reset_to_initial_state=False)  # TODO
    env = LogWrapper(env)
    env = VectorizeWrapper(env)

    seeds = jnp.arange(cfg.num_seeds)
    keys = jax.vmap(jax.random.key)(seeds)
    split = jax.vmap(jax.random.split)(keys)
    keys, model_keys = split[:, 0], split[:, 1]

    make_models = eqx.filter_vmap(DeepLTLModel, in_axes=(None, None, None, 0))
    models = make_models(
        env.observation_space(env_params).shape[0],
        env.action_space(env_params).shape[0],
        jax.nn.tanh,
        model_keys,
    )

    def callback(metric, seed):
        logger.info(f"Seed {seed}. Callback!")

    rl_alg: RLAlgorithm = hydra.utils.instantiate(cfg.rl_alg)
    train = eqx.filter_vmap(
        rl_alg.train, in_axes=(eqx.if_array(0), None, None, 0, None, None, 0)
    )
    train = eqx.filter_jit(train)

    start_time = time.time()
    logger.info("Starting training")
    models, metrics = jax.block_until_ready(
        train(models, env, env_params, keys, None, None, seeds)
    )
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    dfs = []
    for seed in range(cfg.num_seeds):
        seed_metrics = jax.tree.map(lambda x: x[seed], metrics)
        return_values = seed_metrics["episode_return"][seed_metrics["done"]].tolist()
        timesteps = (
            seed_metrics["total_step"][seed_metrics["done"]] * cfg.rl_alg.num_envs
        ).tolist()
        df = pd.DataFrame({"timestep": timesteps, "return": return_values})
        df = df.groupby("timestep").mean().reset_index()
        df["seed"] = seed
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("logs.csv", index=False)

    model = ensemble_to_list(models, cfg.num_seeds)[0]
    eqx.tree_serialise_leaves("model.eqx", model)
    logger.info("Model saved to model.eqx")


if __name__ == "__main__":
    main()
