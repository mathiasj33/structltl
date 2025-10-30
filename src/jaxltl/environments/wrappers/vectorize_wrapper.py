from functools import partial
from typing import Any, NamedTuple

import jax

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class VectorizeWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures]):
    """Vectorize the environment using vmap."""

    def __init__(
        self,
        env: EnvWrapper[TEnvParams, TObsFeatures]
        | Environment[Any, TEnvParams, TObsFeatures],
    ):
        super().__init__(env)

    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def reset(
        self, key: jax.Array, state: WrapperState | None, params: TEnvParams
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return super().reset(key, state, params)

    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def cheap_reset(
        self, key: jax.Array, state: WrapperState, params: TEnvParams
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return super().cheap_reset(key, state, params)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def step(
        self,
        key: jax.Array,
        state: WrapperState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        return super().step(key, state, action, params)
