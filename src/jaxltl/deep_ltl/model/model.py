from typing import NamedTuple, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxltl.deep_ltl.model.actor.continuous_actor import ContinuousActor
from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor_critic import ActorCritic


class DeepLTLModel(ActorCritic):
    env_net: MLP
    embedding: nn.Embedding
    actor: ContinuousActor
    critic: MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        env_key, actor_key, critic_key, embedding_key = jax.random.split(key, 4)
        self.env_net = hydra.utils.instantiate(
            config.env_net, in_size=obs_dim, key=env_key
        )
        embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions,
            embedding_size=embedding_dim,
            key=embedding_key,
        )
        joint_dim = config.env_net.out_size + embedding_dim
        self.actor = hydra.utils.instantiate(
            config.actor, in_size=joint_dim, action_dim=action_dim, key=actor_key
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=joint_dim,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )

    @override
    def _get_action(self, features: jax.Array) -> distrax.Distribution:
        return self.actor(features)

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        value = jax.vmap(self.critic)(features)
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = self.flatten_features(obs.features)
        x = jax.vmap(self.env_net)(x)
        emb = jax.vmap(self.embedding)(obs.seq.reach[:, 0])  # current goal
        return jnp.concatenate([x, emb], axis=-1)

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
