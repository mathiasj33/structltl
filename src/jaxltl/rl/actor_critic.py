from abc import abstractmethod

import distrax
import equinox as eqx
import jax

from jaxltl.environments.environment import EnvObservation


class ActorCritic(eqx.Module):
    @abstractmethod
    def __call__(self, obs: EnvObservation) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass through the actor and critic networks.

        Args:
            obs: Batched observations.

        Returns:
            A tuple of (action distribution, state value).
        """
        pass

    @abstractmethod
    def get_value(self, obs: EnvObservation) -> jax.Array:
        """Get state value from the critic network.

        Args:
            obs: Batched observations.

        Returns:
            State value.
        """
        pass
