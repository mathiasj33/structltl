"""Abstract base class for all jaxltl environments.

Adapted from gymnax (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py)."""

from abc import abstractmethod
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.spaces import Space


class EnvStateWrapper[TEnvState: eqx.Module, TObsFeatures: NamedTuple](eqx.Module):
    """Wraps the environment state with additional metadata."""

    timestep: jax.Array  # int
    state: TEnvState
    initial_state: TEnvState
    initial_obs: TObsFeatures


class EnvParams(eqx.Module):
    """Base class for environment parameters."""

    max_steps_in_episode: jax.Array  # int


class EnvObservation(eqx.Module):
    """Environment observation."""

    features: jax.Array  # shape: (num_features,)
    propositions: jax.Array  # shape: (num_propositions,) boolean


class EnvTransition[TEnvState: eqx.Module, TObsFeatures: NamedTuple](eqx.Module):
    """Environment transition."""

    state: EnvStateWrapper[TEnvState, TObsFeatures]
    observation: EnvObservation
    reward: jax.Array  # shape: ()
    terminated: jax.Array  # shape: () boolean
    truncated: jax.Array  # shape: () boolean
    terminal_observation: EnvObservation  # used if done
    info: dict[Any, Any]


class Environment[
    TEnvState: eqx.Module,
    TEnvParams: eqx.Module,
    TObsFeatures: NamedTuple,
](eqx.Module):
    """Abstract base class for environments. Handles truncation and auto-resets."""

    default_params: TEnvParams
    # Maps indices in obs.propositions to names
    propositions: tuple[str, ...]
    # Environments always auto-reset on termination or truncation.

    # When `reset_to_initial_state` is True, the environment resets to the same initial
    # state every time. This is computationally cheaper, but limits the distribution of
    # initial states that the agent is exposed to. This is the strategy used in Brax.

    # When False, the environment samples a new initial state every time. This is more
    # computationally expensive, since both the next state and an initial state need to
    # be computed at every step (due to requirements of JIT compilation). However, it can
    # be beneficial for training. This is the strategy used in Gymnax.
    reset_to_initial_state: bool

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def reset(
        self, key: jax.Array, params: TEnvParams | None = None
    ) -> tuple[EnvStateWrapper[TEnvState, TObsFeatures], EnvObservation]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        state, obs = self.reset_env(key, params)
        wrapper = EnvStateWrapper(
            timestep=jnp.array(0, dtype=jnp.int32),
            state=state,
            initial_state=state,
            initial_obs=obs,
        )
        obs = EnvObservation(
            features=self.flatten_obs(obs),
            propositions=self.compute_propositions(state, params),
        )
        return wrapper, obs

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
        state: EnvStateWrapper[TEnvState, TObsFeatures],
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> EnvTransition:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        next_state, obs, reward, terminated, info = self.step_env(
            key_step, state.state, action, params
        )
        next_state = EnvStateWrapper(
            timestep=state.timestep + 1,
            state=next_state,
            initial_state=state.initial_state,
            initial_obs=state.initial_obs,
        )
        obs = EnvObservation(
            features=self.flatten_obs(obs),
            propositions=self.compute_propositions(next_state.state, params),
        )

        if self.reset_to_initial_state:
            state_re, obs_re = state.initial_state, state.initial_obs
        else:
            state_re, obs_re = self.reset_env(key_reset, params)
        state_re = EnvStateWrapper(
            timestep=jnp.array(0, dtype=jnp.int32),
            state=state_re,
            initial_state=state_re,
            initial_obs=obs_re,
        )
        obs_re = EnvObservation(
            features=self.flatten_obs(obs_re),
            propositions=self.compute_propositions(state_re.state, params),
        )

        # Truncation
        truncated: jax.Array = next_state.timestep >= params.max_steps_in_episode  # type: ignore

        # Auto-reset environment based on termination
        done = jnp.logical_or(terminated, truncated)
        transition = EnvTransition(
            state=jax.lax.cond(done, lambda: state_re, lambda: next_state),
            observation=jax.lax.cond(done, lambda: obs_re, lambda: obs),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            terminal_observation=obs,
            info=info,
        )

        return transition

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

    @staticmethod
    def flatten_obs(obs: TObsFeatures) -> jax.Array:
        """Flattens observation NamedTuple into a single array."""
        return jnp.concatenate([jnp.ravel(v) for v in obs])

    @abstractmethod
    def unflatten_obs(self, obs: jax.Array) -> TObsFeatures:
        """Unflattens a single array into TObsFeatures."""
        pass
