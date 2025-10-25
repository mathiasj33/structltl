from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.samplers.sequence_sampler import (
    ReachAvoidSequence,
    SequenceSampler,
)
from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper


class SequenceState[TEnvState: eqx.Module](eqx.Module):
    """State for SequenceWrapper."""

    state: TEnvState
    seq: ReachAvoidSequence
    current: jax.Array


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation for SequenceWrapper."""

    seq: ReachAvoidSequence

    @classmethod
    def from_obs(cls, obs: EnvObservation[TObsFeatures], seq: ReachAvoidSequence):
        return cls(features=obs.features, seq=seq)


class SequenceWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures]):
    """A wrapper that adds reach-avoid sequences to the environment."""

    sequence_sampler: SequenceSampler

    def __init__(
        self,
        env: EnvWrapper[TEnvState, TEnvParams, TObsFeatures]
        | Environment[TEnvState, TEnvParams, TObsFeatures],
        sequence_sampler: SequenceSampler,
    ):
        super().__init__(env)
        self.sequence_sampler = sequence_sampler

    @eqx.filter_jit
    def reset(
        self, key: jax.Array, state: SequenceState[TEnvState] | None, params: TEnvParams
    ) -> tuple[SequenceState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().reset(reset_key, state.state if state else None, params)
        if state:
            current = (state.current + 1) % (self.sequence_sampler.num_propositions + 1)
            current = jnp.where(current == 0, 1, current)
            state = SequenceState(
                state=re_state,
                seq=ReachAvoidSequence(
                    reach=jnp.array([current, 2, 2, 2, 2], dtype=jnp.int32),
                    avoid=jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32),
                ),
                current=current,
            )
        else:
            state = self._wrap_reset_state(re_state, sample_key)
        return state, SequenceObservation.from_obs(obs, state.seq)

    @eqx.filter_jit
    def cheap_reset(
        self, key: jax.Array, state: TEnvState, params: TEnvParams
    ) -> tuple[SequenceState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        state, obs = super().cheap_reset(reset_key, state, params)
        new_state = self._wrap_reset_state(state, sample_key)
        return new_state, SequenceObservation.from_obs(obs, new_state.seq)

    def _wrap_reset_state(
        self, state: TEnvState, key: jax.Array
    ) -> SequenceState[TEnvState]:
        seq = self.sequence_sampler.sample(key)
        return SequenceState(
            state=state,
            seq=seq,
            current=seq.reach[0],
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: SequenceState[TEnvState],
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[SequenceState[TEnvState], TObsFeatures]:
        transition = super().step(key, state.state, action, params)
        new_state = SequenceState(
            state=transition.state,
            seq=state.seq,
            current=state.current,
        )
        current = state.seq.reach[0]
        reached = transition.propositions[current - 1]
        reward = jax.lax.cond(reached, lambda: 1.0, lambda: 0.0)
        return EnvTransition(
            state=new_state,
            observation=SequenceObservation.from_obs(
                transition.observation, new_state.seq
            ),
            reward=reward,
            terminated=reached,
            truncated=transition.truncated,
            terminal_observation=SequenceObservation.from_obs(
                transition.terminal_observation, new_state.seq
            ),
            propositions=transition.propositions,
            info=transition.info,
        )

    def unwrapped(self, state: Any) -> TEnvState:
        return self._env.unwrapped(state.state)
