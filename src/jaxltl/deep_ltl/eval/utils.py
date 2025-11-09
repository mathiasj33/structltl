"""Utility functions for evaluation scripts."""

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

import jaxltl
from jaxltl import DATA_DIR, eqx_utils
from jaxltl.deep_ltl.eval.eval import Evaluator
from jaxltl.deep_ltl.reach_avoid import path_search
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.hydra_utils.utils import resolve_default_options
from jaxltl.ltl.automata import ltl2ldba
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from jaxltl.rl.actor_critic import ActorCritic


def build_env(
    cfg: DictConfig, env_params: EnvParams | None
) -> tuple[EnvWrapper, EnvParams]:
    """Builds the environment."""

    env, env_params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = f"{DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    default_options = resolve_default_options(cfg.env)
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )
    env = VectorizeWrapper(env)
    return env, env_params


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Preprocesses formulas into a batched JaxLDBA and batched JaxReachAvoidSequence."""

    ldbas, seqs = [], []
    for formula in formulas:
        ldba, batched_seqs = _preprocess_formula(formula, env)
        ldbas.append(ldba)
        seqs.append(batched_seqs)
    ldba = _batch_ldbas(ldbas)
    batched_seqs = _batch_sequences(seqs)
    return ldba, batched_seqs


def _preprocess_formula(
    formula: str, env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Preprocesses the formula into a JaxLDBA and batched JaxReachAvoidSequence."""

    ldba = _build_ldba(formula, env)
    jldba = JaxLDBA.from_ldba(ldba, env)
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)
    batched_seqs = JaxReachAvoidSequence.from_state_to_seqs(state_to_seqs, env)
    return jldba, batched_seqs


def _build_ldba(formula: str, env: Environment | EnvWrapper):
    ldba = ltl2ldba(formula, env.propositions)
    ldba.prune(env.assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _batch_ldbas(ldbas: list[JaxLDBA]) -> JaxLDBA:
    """Batch multiple JaxLDBAs into a single JaxLDBA with an added batch dimension."""

    num_states = jnp.array([ldba.num_states for ldba in ldbas], dtype=jnp.int32)
    max_num_states = jnp.max(num_states)
    batch_size = len(ldbas)
    num_assignments = ldbas[0].transitions.shape[1] - 1

    transitions = -jnp.ones(
        (batch_size, max_num_states, num_assignments + 1), dtype=jnp.int32
    )
    accepting = jnp.zeros((batch_size, max_num_states, num_assignments), dtype=bool)
    initial_states = jnp.zeros((batch_size,), dtype=jnp.int32)

    for i, ldba in enumerate(ldbas):
        transitions = transitions.at[i, : ldba.num_states, :].set(ldba.transitions)
        accepting = accepting.at[i, : ldba.num_states, :].set(ldba.accepting)
        initial_states = initial_states.at[i].set(ldba.initial_state)

    return JaxLDBA(
        num_states=num_states,
        initial_state=initial_states,
        transitions=transitions,
        accepting=accepting,
        finite=jnp.array([ldba.finite for ldba in ldbas]),
    )


def _batch_sequences(
    seqs: list[JaxReachAvoidSequence],
) -> JaxReachAvoidSequence:
    """Batch multiple JaxReachAvoidSequences into a single JaxReachAvoidSequence with an added batch dimension.

    Args:
        seqs: List of JaxReachAvoidSequence to batch. Shape: (num_states, num_seqs, max_length, num_assignments)

    Returns:
        JaxReachAvoidSequence: Batched sequence. Shape: (batch_size, max_num_states, max_num_seqs, max_length, num_assignments)
    """
    max_num_states = max(seq.reach.shape[0] for seq in seqs)
    max_num_seqs = max(seq.reach.shape[1] for seq in seqs)
    max_length = max(seq.reach.shape[2] for seq in seqs)

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_seqs - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (seqs)
        )

        if x.ndim > 2:
            pad_axis2 = max(0, max_length - x.shape[2])
            pad_config += ((0, pad_axis2),)  # Axis 2 (length)

        pad_config += ((0, 0),) * (x.ndim - len(pad_config))

        return jnp.pad(x, pad_width=pad_config, mode="constant", constant_values=-1)

    seqs = jax.tree.map(pad_leaf, seqs)
    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *seqs,
    )


def load_batched_models(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> ActorCritic:
    """Load a batched model (over seeds) from disk."""

    model_path = f"runs/{cfg.env.name}/{cfg.run}/models.eqx"
    metadata = eqx_utils.load_metadata(model_path)
    num_models = metadata["num_models"]
    model = hydra.utils.instantiate(
        cfg.model,
        obs_dim=env.observation_space(env_params).shape[0],
        action_dim=env.action_space(env_params).shape[0],
        num_assignments=len(env.assignments),
        key=key,
    )
    models = eqx_utils.add_batch_dim(model, num_models)
    models = eqx_utils.load(model_path, models)
    return models


def load_model_checkpoints(
    cfg: DictConfig,
    env: Environment | EnvWrapper,
    env_params: EnvParams,
    *,
    key: jax.Array,
) -> tuple[ActorCritic, list[int]]:
    """Load model checkpoints from disk.

    Returns:
        Batched ActorCritic model with shape (num_checkpoints, num_seeds, ...),
        list of checkpoint steps.
    """

    make_model = lambda key: hydra.utils.instantiate(
        cfg.model,
        obs_dim=env.observation_space(env_params).shape[0],
        action_dim=env.action_space(env_params).shape[0],
        num_assignments=len(env.assignments),
        key=key,
    )
    model = make_model(key)
    params, static = eqx.partition(model, eqx.is_array)

    # load checkpoints
    step_to_models = defaultdict(dict)
    checkpoint_folder = Path(f"runs/{cfg.env.name}/{cfg.run}/checkpoints")
    for file in checkpoint_folder.iterdir():
        seed = re.search(r"seed(\d+)", file.name).group(1)  # type: ignore
        step = re.search(r"step(\d+)", file.name).group(1)  # type: ignore
        checkpoint_params = eqx_utils.load(file, params)
        step_to_models[int(step)][int(seed)] = checkpoint_params

    seeds_per_step = [set(seeds.keys()) for seeds in step_to_models.values()]
    if not all(seeds == seeds_per_step[0] for seeds in seeds_per_step):
        raise ValueError("Not all checkpoints have the same seeds.")

    # load initial models
    for seed in range(len(seeds_per_step[0])):
        key, subkey = jax.random.split(key)
        init_params, _ = eqx.partition(make_model(subkey), eqx.is_array)
        step_to_models[0][seed] = init_params

    sorted_steps = sorted(step_to_models)
    models_list = []
    for step in sorted_steps:
        seeds_dict = step_to_models[step]
        models_per_seed = []
        for seed in sorted(seeds_dict):
            models_per_seed.append(seeds_dict[seed])
        models_list.append(
            jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_per_seed)
        )
    models = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *models_list)
    return eqx.combine(models, static), sorted_steps


def make_eval_fn(cfg: DictConfig) -> Callable:
    """Creates an eval function that is vmapped over formulas and seeds. The function
    returns arrays of shape (num_formulas, num_models, num_episodes)."""

    evaluator = Evaluator(
        num_episodes=cfg.eval.num_episodes, discount=cfg.eval.discount
    )
    eval_fn = eqx.filter_vmap(  # map over seeds
        evaluator.eval,
        in_axes=(eqx.if_array(0), None, None, None, None, None, None),
    )
    # map over formulas
    eval_fn = jax.vmap(eval_fn, in_axes=(None, None, None, None, 0, 0, None))
    return eval_fn
