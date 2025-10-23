from collections.abc import Callable

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxltl.networks.mlp import MLP


class ContinuousActor(eqx.Module):
    encoder: MLP
    action_mean: MLP
    action_std: MLP | None
    log_std: jax.Array | None

    def __init__(
        self,
        in_size: int,
        action_dim: int,
        hidden_sizes: list[int],
        state_dependent_std: bool = True,
        hidden_activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        output_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),  # noqa
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        enc_key, mean_key, std_key = jax.random.split(key, 3)
        self.encoder = MLP(
            in_size,
            hidden_sizes[-1],
            hidden_sizes[:-1],
            hidden_activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=enc_key,
        )
        self.action_mean = MLP(
            hidden_sizes[-1],
            action_dim,
            [],
            output_activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=mean_key,
        )
        if state_dependent_std:
            self.action_std = MLP(
                hidden_sizes[-1],
                action_dim,
                [],
                output_activation,
                weight_init,
                bias_init,
                final_layer_activation=True,
                key=std_key,
            )
            self.log_std = None
        else:
            self.log_std = jnp.zeros((action_dim,))
            self.action_std = None

    def __call__(self, x: jax.Array) -> distrax.Distribution:
        """Input shape: (batch_size, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        encoded = jax.vmap(self.encoder)(x)
        mean = jax.vmap(self.action_mean)(encoded)
        if self.action_std is not None:
            std = jax.nn.softplus(jax.vmap(self.action_std)(encoded))
        else:
            std = jnp.exp(self.log_std)[None, :].reshape(mean.shape)  # type: ignore
        std += 1e-6  # numerical stability
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
