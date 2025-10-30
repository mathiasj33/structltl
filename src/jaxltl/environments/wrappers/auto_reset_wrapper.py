from enum import StrEnum, auto
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


class ResetStrategy(StrEnum):
    INITIAL = auto()
    CHEAP = auto()
    FULL = auto()


class AutoResetWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions]):
    """Automatically reset the environment on termination or truncation.

    Due to JIT compilation requirements, we have to compute a new reset state at every
    step of the environment. Since this can be computationally expensive in some environments
    (e.g. sampling layouts etc.), we provide three different reset strategies:

        - Initial: Always reset to the initial state obtained from the first reset call.
        - Cheap: Use the environment's cheap_reset method to compute a new state.
        - Full: Use the full reset method to compute a new state.

    Brax by default uses the 'Initial' strategy, whereas Gymnax environments use 'Full'.

    Note also that the PrecomputedResetWrapper can be used to always reset the environment to
    a randomly sampled state from a fixed set of pre-computed states. This can be used
    together with the 'Full' reset strategy without incurring the computational cost
    of computing a new reset state from scratch every time.
    """

    reset_strategy: ResetStrategy
    use_term_trunc: bool
    auto_reset_options: TResetOptions | None

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions]
            | Environment[TEnvState, TEnvParams, TObsFeatures, TResetOptions]
        ),
        reset_strategy: ResetStrategy,
        use_term_trunc: bool = True,
        auto_reset_options: TResetOptions | None = None,
    ):
        """
        Params:
            env: The environment to wrap.
            reset_strategy: The reset strategy to use.
            use_term_trunc: Whether to separate termination and truncation, or treat
                truncation as a form of termination. This is what original Gym environments
                do.
            auto_reset_options: The reset options to use for automatic resets.
        """
        super().__init__(env)
        self.reset_strategy = reset_strategy
        self.use_term_trunc = use_term_trunc
        self.auto_reset_options = auto_reset_options

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrappedState[TEnvState, TObsFeatures], EnvObservation[TObsFeatures]]:
        if options is None:
            options = self.auto_reset_options
        state, obs = super().reset(key, state, params, options)
        return self._wrap_reset_state(state, obs), obs

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrappedState[TEnvState, TObsFeatures], EnvObservation[TObsFeatures]]:
        if options is None:
            options = self.auto_reset_options
        state, obs = super().cheap_reset(key, state, params, options)
        return self._wrap_reset_state(state, obs), obs

    def _wrap_reset_state(
        self, state: TEnvState, obs: EnvObservation[TObsFeatures]
    ) -> WrappedState[TEnvState, TObsFeatures]:
        return WrappedState(
            timestep=jnp.array(0, dtype=jnp.int32),
            state=state,
            initial_state=state,
            initial_obs=obs,
        )

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
        match self.reset_strategy:
            case ResetStrategy.INITIAL:
                state_re, obs_re = state.initial_state, state.initial_obs
                state_re = WrappedState(
                    timestep=jnp.array(0, dtype=jnp.int32),
                    state=state_re,
                    initial_state=state.initial_state,
                    initial_obs=state.initial_obs,
                )
            case ResetStrategy.CHEAP:
                state_re, obs_re = self.cheap_reset(
                    key_reset,
                    transition.state,
                    params,
                    self.auto_reset_options,
                )
            case ResetStrategy.FULL:
                state_re, obs_re = self.reset(
                    key_reset, transition.state, params, self.auto_reset_options
                )

        # Truncation
        truncated: jax.Array = next_state.timestep >= params.max_steps_in_episode  # type: ignore
        terminated = (
            jnp.logical_or(transition.terminated, truncated)
            if self.use_term_trunc
            else transition.terminated
        )

        # Auto-reset environment based on termination
        done = jnp.logical_or(transition.terminated, truncated)
        transition = EnvTransition(
            state=jax.lax.cond(done, lambda: state_re, lambda: next_state),
            observation=jax.lax.cond(
                done, lambda: obs_re, lambda: transition.observation
            ),
            reward=transition.reward,
            terminated=terminated,
            truncated=truncated,
            terminal_observation=transition.terminal_observation,
            propositions=transition.propositions,
            info=transition.info,
        )
        return transition

    def unwrapped(self, state: WrappedState[TEnvState, TObsFeatures]) -> TEnvState:
        return self._env.unwrapped(state.state)
