from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class LogEnvState(WrapperState):
    step: jax.Array  # int
    total_step: jax.Array  # int
    ret: jax.Array  # float


class LogWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures]):
    """Log the episode returns and lengths to the info dict."""

    def __init__(
        self,
        env: EnvWrapper[TEnvParams, TObsFeatures]
        | Environment[Any, TEnvParams, TObsFeatures],
    ):
        super().__init__(env)

    @eqx.filter_jit
    def reset(
        self, key: jax.Array, state: LogEnvState | None, params: TEnvParams
    ) -> tuple[LogEnvState, EnvObservation[TObsFeatures]]:
        env_state, obs = super().reset(key, state, params)
        return self._wrap_reset_state(env_state), obs

    @eqx.filter_jit
    def cheap_reset(
        self, key: jax.Array, state: LogEnvState, params: TEnvParams
    ) -> tuple[LogEnvState, EnvObservation[TObsFeatures]]:
        env_state, obs = super().cheap_reset(key, state, params)
        return self._wrap_reset_state(env_state), obs

    def _wrap_reset_state(self, state: Any) -> LogEnvState:
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
        transition = super().step(key, state, action, params)
        ret = transition.reward + state.ret
        length = state.step + 1
        total_step = state.total_step + 1
        stage = transition.state.curriculum_stage
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
            "curriculum_stage": stage,
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
