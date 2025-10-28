from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class VectorizeWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions]):
    """Vectorize the environment using vmap."""

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvState, TEnvParams, TObsFeatures, TResetOptions]
            | Environment[TEnvState, TEnvParams, TObsFeatures, TResetOptions]
        ),
    ):
        super().__init__(env)

    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        return super().reset(key, state, params, options)

    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        return super().cheap_reset(key, state, params, options)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TEnvState, TObsFeatures]:
        return super().step(key, state, action, params)
