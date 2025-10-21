from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class WrappedState[TEnvState: eqx.Module, TObsFeatures: NamedTuple](eqx.Module):
    timestep: jax.Array  # int
    state: TEnvState
    initial_state: TEnvState
    initial_obs: EnvObservation[TObsFeatures]


class AutoResetWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures]):
    """Automatically reset the environment on termination or truncation."""

    # When `reset_to_initial_state` is True, the environment resets to the same initial
    # state every time. This is computationally cheaper, but limits the distribution of
    # initial states that the agent is exposed to. This is the strategy used in Brax.

    # When False, the environment samples a new initial state every time. This is more
    # computationally expensive, since both the next state and an initial state need to
    # be computed at every step (due to requirements of JIT compilation). However, it can
    # be beneficial for training. This is the strategy used in Gymnax.
    reset_to_initial_state: bool

    def __init__(
        self,
        env: EnvWrapper[TEnvState, TEnvParams, TObsFeatures]
        | Environment[TEnvState, TEnvParams, TObsFeatures],
        reset_to_initial_state: bool,
    ):
        super().__init__(env)
        self.reset_to_initial_state = reset_to_initial_state

    @eqx.filter_jit
    def reset(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[WrappedState[TEnvState, TObsFeatures], EnvObservation[TObsFeatures]]:
        state, obs = super().reset(key, params)
        wrapper = WrappedState(
            timestep=jnp.array(0, dtype=jnp.int32),
            state=state,
            initial_state=state,
            initial_obs=obs,
        )
        return wrapper, obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: WrappedState[TEnvState, TObsFeatures],
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[WrappedState[TEnvState, TObsFeatures], TObsFeatures]:
        key_step, key_reset = jax.random.split(key, 2)
        transition = super().step(key_step, state.state, action, params)
        next_state = WrappedState(
            timestep=state.timestep + 1,
            state=transition.state,
            initial_state=state.initial_state,
            initial_obs=state.initial_obs,
        )
        if self.reset_to_initial_state:
            state_re, obs_re = state.initial_state, state.initial_obs
            state_re = WrappedState(
                timestep=jnp.array(0, dtype=jnp.int32),
                state=state_re,
                initial_state=state.initial_state,
                initial_obs=state.initial_obs,
            )
        else:
            state_re, obs_re = self.reset(key_reset, params)

        # Truncation
        truncated: jax.Array = next_state.timestep >= params.max_steps_in_episode  # type: ignore

        # Auto-reset environment based on termination
        done = jnp.logical_or(transition.terminated, truncated)
        transition = EnvTransition(
            state=jax.lax.cond(done, lambda: state_re, lambda: next_state),
            observation=jax.lax.cond(
                done, lambda: obs_re, lambda: transition.observation
            ),
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=truncated,
            terminal_observation=transition.terminal_observation,
            info=transition.info,
        )
        return transition

    def unwrapped(self, state: WrappedState[TEnvState, TObsFeatures]) -> TEnvState:
        return self._env.unwrapped(state.state)
