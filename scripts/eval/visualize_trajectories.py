"""Visualise trajectories of trained models on specified LTL formulas."""

import logging
import time
from functools import partial

import equinox as eqx
import hydra
import jax
from jax import numpy as jnp
from omegaconf import DictConfig

from jaxltl.deep_ltl.eval.utils import (
    build_env,
    load_batched_models,
    make_eval_fn,
    preprocess_formulas,
    preprocess_graph_formulas,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="visualize_traj")
def main(cfg: DictConfig):
    # build environment
    env, env_params = build_env(cfg, None)

    # construct ldba and batched sequences for formula
    # NOTE: consider replacing entirely with preprocess_graph_formulas in future
    if "ltl_gnn" in cfg.model._target_:
        ldba, batched_seqs = preprocess_graph_formulas([cfg.eval.formula], env)
    else:
        ldba, batched_seqs = preprocess_formulas([cfg.eval.formula], env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, _ = load_batched_models(cfg, env, env_params, key=model_key)

    # select single model from ensemble (while keeping batch dimension)
    params, static = eqx.partition(models, eqx.is_array)
    params = jax.tree.map(lambda x: x[cfg.eval.model_index], params)
    params = jax.tree.map(lambda x: x[None, ...], params)  # add batch dim
    model = eqx.combine(params, static)

    # set up evaluator
    eval_fn = make_eval_fn(cfg, num_models=1, return_trajs=True)

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    metrics, returns, lengths, trajs = eval_fn(
        model,
        env,
        env_params,
        ldba,
        batched_seqs,
        eval_key,
    )  # shape: (1, 1, num_episodes)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # plot trajectories
    trajs = jax.tree.map(partial(jnp.squeeze, axis=[0, 1]), trajs)
    lengths = jax.tree.map(partial(jnp.squeeze, axis=[0, 1]), lengths)

    if cfg.replay:
        renderer = env.get_renderer(env_params)
        renderer.replay_trajectories(
            trajs,
            lengths,
            frames_per_step=cfg.render.frames_per_step,
            pause_between_episodes=cfg.render.pause_between_episodes,
        )
        renderer.close()

    env.plot_trajectories(
        trajs,
        lengths,
        env_params,
        num_cols=cfg.plotting.cols,
        num_rows=cfg.plotting.rows,
    )


if __name__ == "__main__":
    main()
