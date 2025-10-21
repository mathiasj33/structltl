from collections.abc import Callable
from typing import NamedTuple, override

import distrax
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree

from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor_critic import ActorCritic


class DeepLTLModel(ActorCritic):
    actor: MLP
    critic: MLP
    embedding: nn.Embedding

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation: Callable[[jax.Array], jax.Array],
        embedding_dim: int,
        num_propositions: int,
        key: jax.Array,
    ):
        key, embed_key = jax.random.split(key)
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions,
            embedding_size=embedding_dim,
            key=embed_key,
        )
        actor_key, critic_key = jax.random.split(key, 2)
        self.actor = MLP(
            in_size=obs_dim + embedding_dim,
            out_size=action_dim,
            hidden_sizes=[64, 64],
            activation=activation,
            final_layer_activation=False,
            weight_init_scales=[jnp.sqrt(2).item(), jnp.sqrt(2).item(), 0.01],
            key=actor_key,
        )
        self.critic = MLP(
            in_size=obs_dim + embedding_dim,
            out_size=1,
            hidden_sizes=[64, 64],
            activation=activation,
            final_layer_activation=False,
            weight_init_scales=[jnp.sqrt(2).item(), jnp.sqrt(2).item(), 1.0],
            key=critic_key,
        )

    @override
    def _get_action(self, features: jax.Array) -> distrax.Distribution:
        actor_mean = jax.vmap(self.actor)(features)
        # pi = distrax.Categorical(logits=actor_mean)
        pi = distrax.MultivariateNormalDiag(loc=actor_mean)
        return pi

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        value = jax.vmap(self.critic)(features)
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = self.flatten_features(obs.features)
        emb = jax.vmap(self.embedding)(obs.seq.reach[:, 0])  # current goal
        x = jnp.concatenate([x, emb], axis=-1)
        return x

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
