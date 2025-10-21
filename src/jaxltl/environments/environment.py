"""Abstract base class for all jaxltl environments.

Adapted from gymnax (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py)."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.spaces import Space

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer


@dataclass(frozen=True)
class EnvParams:
    """Base class for environment parameters.

    Note: changing environment parameters will require recompilation of jitted functions.
    """

    max_steps_in_episode: int


class EnvObservation[TObsFeatures: NamedTuple](NamedTuple):
    """Environment observation."""

    features: TObsFeatures
    propositions: jax.Array  # shape: (num_propositions,) boolean


class EnvTransition[TEnvState: eqx.Module, TObsFeatures: NamedTuple](NamedTuple):
    """Environment transition."""

    state: TEnvState
    observation: EnvObservation[TObsFeatures]
    reward: jax.Array  # shape: ()
    terminated: jax.Array  # shape: () boolean
    truncated: jax.Array  # shape: () boolean
    terminal_observation: EnvObservation[TObsFeatures]  # used if done
    info: dict[Any, Any]

    @property
    def done(self) -> jax.Array:
        """Whether the episode is done (terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)


class Environment[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
](eqx.Module):
    """Abstract base class for environments."""

    default_params: TEnvParams
    # Maps indices in obs.propositions to names
    propositions: tuple[str, ...]

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def reset(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        """Performs resetting of environment."""
        state, obs = self.reset_env(key, params)
        obs = EnvObservation(
            features=obs,
            propositions=self.compute_propositions(state, params),
        )
        return state, obs

    @abstractmethod
    def reset_env(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[TEnvState, TObsFeatures]:
        """Environment-specific reset."""
        pass

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TEnvState, TObsFeatures]:
        """Performs step transitions in the environment."""
        next_state, obs, reward, terminated, info = self.step_env(
            key, state, action, params
        )
        obs = EnvObservation(
            features=obs,
            propositions=self.compute_propositions(next_state, params),
        )
        transition = EnvTransition(
            state=next_state,
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=jnp.array(False, dtype=jnp.bool),
            terminal_observation=obs,
            info=info,
        )
        return jax.lax.stop_gradient(transition)

    @abstractmethod
    def step_env(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[TEnvState, TObsFeatures, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition.
        Returns: next_state, observation, reward, terminated, info"""
        pass

    @abstractmethod
    def compute_propositions(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Computes atomic propositions from environment state.

        Returns: boolean array of shape (num_propositions,)"""
        pass

    def observation_space(self, params: TEnvParams | None = None) -> Space:
        """Observation space of the environment."""
        if params is None:
            params = self.default_params
        return self._observation_space(params)

    @abstractmethod
    def _observation_space(self, params: TEnvParams) -> Space:
        pass

    def action_space(self, params: TEnvParams | None = None) -> Space:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return self._action_space(params)

    @abstractmethod
    def _action_space(self, params: TEnvParams) -> Space:
        pass

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def unwrapped(self, state: Any) -> TEnvState:
        """Returns the unwrapped environment state."""
        return state

    @abstractmethod
    def get_renderer(
        self, params: TEnvParams, **kwargs
    ) -> "BaseRenderer[TEnvState, TObsFeatures]":
        """Returns a renderer for the environment."""
        pass
