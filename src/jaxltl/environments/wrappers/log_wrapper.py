from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class LogEnvState[TEnvState: eqx.Module](eqx.Module):
    state: TEnvState
    step: jax.Array  # int
    total_step: jax.Array  # int
    ret: jax.Array  # float


class LogWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures]):
    """Log the episode returns and lengths to the info dict."""

    def __init__(
        self,
        env: EnvWrapper[TEnvState, TEnvParams, TObsFeatures]
        | Environment[TEnvState, TEnvParams, TObsFeatures],
    ):
        super().__init__(env)

    @eqx.filter_jit
    def reset(
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[LogEnvState[TEnvState], EnvObservation[TObsFeatures]]:
        state, obs = super().reset(key, params)
        return self._wrap_reset_state(state), obs

    @eqx.filter_jit
    def cheap_reset(
        self, key: jax.Array, state: TEnvState, params: TEnvParams
    ) -> tuple[LogEnvState[TEnvState], EnvObservation[TObsFeatures]]:
        state, obs = super().cheap_reset(key, state, params)
        return self._wrap_reset_state(state), obs

    def _wrap_reset_state(self, state: TEnvState) -> LogEnvState[TEnvState]:
        return LogEnvState(
            state=state,
            step=jnp.array(0, dtype=jnp.int32),
            total_step=jnp.array(0, dtype=jnp.int32),
            ret=jnp.array(0.0, dtype=jnp.float32),
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[LogEnvState, TObsFeatures]:
        transition = super().step(key, state.state, action, params)
        ret = transition.reward + state.ret
        length = state.step + 1
        total_step = state.total_step + 1
        log_state = LogEnvState(
            step=(state.step + 1) * (1 - transition.done),
            state=transition.state,
            total_step=state.total_step + 1,
            ret=ret * (1.0 - transition.done),
        )
        info = {
            "episode_return": ret,
            "episode_length": length,
            "total_step": total_step,
            "done": transition.done,
        } | transition.info
        return EnvTransition(
            state=log_state,
            observation=transition.observation,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            terminal_observation=transition.terminal_observation,
            propositions=transition.propositions,
            info=info,
        )

    def unwrapped(self, state: LogEnvState) -> TEnvState:
        return self._env.unwrapped(state.state)
