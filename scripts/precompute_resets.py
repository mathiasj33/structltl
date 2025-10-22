"""Script to precompute environment reset states and save them to disk. Can be used
with the PrecomputedResetWrapper to speed up training."""

import logging
import math
import time

import hydra
import jax
from omegaconf import DictConfig

import jaxltl
from jaxltl import eqx_utils

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="precompute")
def main(cfg: DictConfig):
    num_batch_resets = math.ceil(cfg.num_resets / cfg.num_envs)
    env, params = jaxltl.make(cfg.env)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    key = jax.random.key(0)

    @jax.jit
    def body(key, _):
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, cfg.num_envs)
        states, _ = vmap_reset(subkeys, params)
        return key, states

    start_time = time.time()
    _, states = jax.lax.scan(body, key, None, length=num_batch_resets)
    jax.block_until_ready(states)
    seconds = time.time() - start_time
    logger.info(f"Performed {cfg.num_resets} resets in {seconds:.2f} seconds")

    # Reshape states to (num_resets, ...)
    states = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), states)
    nbytes = sum(x.nbytes for x in jax.tree.leaves(states))
    logger.info(f"Total data size: {nbytes / 2**20:.2f} MB")
    logger.info(f"Shape: {states.position.shape}")

    folder = jaxltl.DATA_DIR / cfg.env
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "train" if cfg["train"] else "test"
    file = folder / f"sampled_resets_{suffix}.eqx"
    logger.info(f"Saving to {file}")
    eqx_utils.save(
        file, states, metadata={"batch_dim": num_batch_resets * cfg.num_envs}
    )


if __name__ == "__main__":
    main()
