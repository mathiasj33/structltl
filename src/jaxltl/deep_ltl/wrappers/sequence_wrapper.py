from typing import Any, NamedTuple

import equinox as eqx
import jax

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


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation for SequenceWrapper."""

    seq: ReachAvoidSequence

    @classmethod
    def from_obs(cls, obs: EnvObservation[TObsFeatures], seq: ReachAvoidSequence):
        return cls(features=obs.features, propositions=obs.propositions, seq=seq)


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
        self, key: jax.Array, params: TEnvParams
    ) -> tuple[SequenceState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        state, obs = super().reset(reset_key, params)
        state = SequenceState(
            state=state,
            seq=self.sequence_sampler.sample(sample_key),
        )
        return state, SequenceObservation.from_obs(obs, state.seq)

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
        )
        props = transition.observation.propositions
        current = state.seq.reach[0]
        reached = props[current - 1]  # propositions are 1-indexed
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
            info=transition.info,
        )

    def unwrapped(self, state: Any) -> TEnvState:
        return self._env.unwrapped(state.state)
