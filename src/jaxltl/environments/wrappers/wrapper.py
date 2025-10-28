"""Environment wrappers.

Adapted from gymnax
(https://github.com/RobertTLange/gymnax/blob/main/gymnax/wrappers/purerl.py).
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition


class EnvWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](eqx.Module):
    """Base class for environment wrappers."""

    _env: "EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions] | Environment[TEnvState, TEnvParams, TObsFeatures, TResetOptions]"

    def __init__(
        self,
        env: "EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions] | Environment[TEnvState, TEnvParams, TObsFeatures, TResetOptions]",
    ):
        self._env = env

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        return self._env.reset(key, state, params, options)

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        return self._env.cheap_reset(key, state, params, options)

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TEnvState, TObsFeatures]:
        return self._env.step(key, state, action, params)

    # provide proxy access to regular attributes of wrapped environment
    def __getattr__(self, name):
        return getattr(self._env, name)

    def unwrapped(self, state: Any) -> TEnvState:
        """Returns the unwrapped environment state."""
        return self._env.unwrapped(state)
