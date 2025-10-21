from collections.abc import Callable
from typing import NamedTuple, override

import distrax
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import EnvObservation
from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor_critic import ActorCritic


class DeepLTLModel(ActorCritic):
    actor: MLP
    critic: MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        activation: Callable[[jax.Array], jax.Array],
        key: jax.Array,
    ):
        actor_key, critic_key = jax.random.split(key, 2)
        self.actor = MLP(
            in_size=obs_dim,
            out_size=action_dim,
            hidden_sizes=[64, 64],
            activation=activation,
            final_layer_activation=False,
            weight_init_scales=[jnp.sqrt(2).item(), jnp.sqrt(2).item(), 0.01],
            key=actor_key,
        )
        self.critic = MLP(
            in_size=obs_dim,
            out_size=1,
            hidden_sizes=[64, 64],
            activation=activation,
            final_layer_activation=False,
            weight_init_scales=[jnp.sqrt(2).item(), jnp.sqrt(2).item(), 1.0],
            key=critic_key,
        )

    @override
    def __call__(self, obs: EnvObservation) -> tuple[distrax.Distribution, jax.Array]:
        x = self.flatten_features(obs.features)
        actor_mean = jax.vmap(self.actor)(x)
        # pi = distrax.Categorical(logits=actor_mean)
        pi = distrax.MultivariateNormalDiag(loc=actor_mean)
        value = jax.vmap(self.critic)(x)
        return pi, value.squeeze(-1)

    @override
    def get_value(self, obs: EnvObservation) -> jax.Array:
        x = self.flatten_features(obs.features)
        value = jax.vmap(self.critic)(x)
        return value.squeeze(-1)

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
