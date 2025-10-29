"""Abstract base class for all jaxltl environments.

Adapted from gymnax (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py).
"""

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


class EnvObservation[TObsFeatures: NamedTuple](eqx.Module):
    """Environment observation. Can be extended by wrappers to add additional fields."""

    features: TObsFeatures


class EnvTransition[TEnvState: eqx.Module, TObsFeatures: NamedTuple](NamedTuple):
    """Environment transition."""

    state: TEnvState
    observation: EnvObservation[TObsFeatures]
    reward: jax.Array  # shape: ()
    terminated: jax.Array  # shape: () boolean
    truncated: jax.Array  # shape: () boolean
    terminal_observation: EnvObservation[TObsFeatures]  # used if done
    # shape: (num_propositions,) int32: index in propositions / -1 for padding
    propositions: jax.Array
    info: dict[Any, Any]

    @property
    def done(self) -> jax.Array:
        """Whether the episode is done (terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)


class Environment[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](eqx.Module):
    """Abstract base class for environments."""

    default_params: TEnvParams
    # Maps indices in obs.propositions to names
    propositions: tuple[str, ...]

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        """Performs resetting of environment.

        Dependence on state is needed for some wrappers (e.g. CurriculumWrapper).
        """
        state = self._reset(key, state, params, options)
        return state, self.compute_obs(state, params)

    @abstractmethod
    def _reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> TEnvState:
        """Environment-specific reset."""
        pass

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        """Performs a cheap reset of the environment given the current state.
        Since JIT requires resetting on every step, this method can be used to implement
        a faster reset to improve performance. See AutoResetWrapper for further details.
        """

        state = self._cheap_reset(key, state, params, options)
        return state, self.compute_obs(state, params)

    @abstractmethod
    def _cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> TEnvState:
        """Environment-specific cheap reset."""
        pass

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TEnvState, TObsFeatures]:
        """Performs step transitions in the environment."""
        next_state, reward, terminated, info = self._step(key, state, action, params)
        obs = self.compute_obs(next_state, params)
        propositions = self.compute_propositions(next_state, params)
        transition = EnvTransition(
            state=next_state,
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=jnp.array(False, dtype=jnp.bool),
            terminal_observation=obs,
            propositions=propositions,
            info=info,
        )
        return jax.lax.stop_gradient(transition)

    @abstractmethod
    def _step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition.
        Returns: next_state, reward, terminated, info"""
        pass

    def compute_obs(
        self, state: TEnvState, params: TEnvParams
    ) -> EnvObservation[TObsFeatures]:
        """Compute the observation for a given state."""
        return EnvObservation(features=self._compute_obs(state, params))

    @abstractmethod
    def _compute_obs(self, state: TEnvState, params: TEnvParams) -> TObsFeatures:
        """Compute the environment-specific observation for a given state."""
        pass

    @abstractmethod
    def compute_propositions(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Computes atomic propositions from environment state.

        Returns: int32 array of shape (num_propositions,) where each entry is the index
        in self.propositions (or -1 for padding)."""
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

    def map_assignment_to_index(self, assignment: jax.Array) -> jax.Array:
        """Maps a proposition assignment to an index in the assignments array.

        Args:
            assignment: jax.Array of shape (num_propositions,) int32

        Returns:
            jax.Array of shape () int32: index in assignments array
        """
        matches = jnp.all(self.assignments == assignment, axis=1)  # (num_assignments,)
        return jnp.argmax(matches)  # () int32

    @property
    @abstractmethod
    def assignments(self) -> jax.Array:
        """Returns the possible assignments in the environment.

        Returns: array of shape (num_assignments, num_propositions) int32
        """
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
