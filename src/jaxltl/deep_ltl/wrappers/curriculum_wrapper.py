from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.curriculum import (
    Curriculum,
    ReachAvoidSequence,
)
from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper


class CurriculumState[TEnvState: eqx.Module](eqx.Module):
    """State for CurriculumWrapper."""

    state: TEnvState
    seq: ReachAvoidSequence  # current reach-avoid sequence
    curriculum_stage: jax.Array  # current stage in the curriculum


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation returned by CurriculumWrapper."""

    seq: ReachAvoidSequence

    @classmethod
    def from_obs(cls, obs: EnvObservation[TObsFeatures], seq: ReachAvoidSequence):
        return cls(features=obs.features, seq=seq)


class CurriculumWrapper[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvState, TEnvParams, TObsFeatures]):
    """A wrapper that adds reach-avoid sequences sampled from a curriculum to the environment."""

    curriculum: Curriculum

    def __init__(
        self,
        env: EnvWrapper[TEnvState, TEnvParams, TObsFeatures]
        | Environment[TEnvState, TEnvParams, TObsFeatures],
        curriculum: Curriculum,
    ):
        super().__init__(env)
        self.curriculum = curriculum

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: CurriculumState[TEnvState] | None,
        params: TEnvParams,
    ) -> tuple[CurriculumState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().reset(reset_key, state.state if state else None, params)
        stage = state.curriculum_stage if state else jnp.zeros((), dtype=jnp.int32)
        state = self._wrap_reset_state(stage, re_state, sample_key)
        return state, SequenceObservation.from_obs(obs, state.seq)

    @eqx.filter_jit
    def cheap_reset(
        self, key: jax.Array, state: CurriculumState[TEnvState], params: TEnvParams
    ) -> tuple[CurriculumState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().cheap_reset(reset_key, state.state, params)
        state = self._wrap_reset_state(state.curriculum_stage, re_state, sample_key)
        return state, SequenceObservation.from_obs(obs, state.seq)

    def _wrap_reset_state(
        self, stage: jax.Array, state: TEnvState, key: jax.Array
    ) -> CurriculumState[TEnvState]:
        seq = self.curriculum.sample(stage, key)
        return CurriculumState(
            state=state,
            seq=seq,
            curriculum_stage=stage,
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: CurriculumState[TEnvState],
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[CurriculumState[TEnvState], TObsFeatures]:
        transition = super().step(key, state.state, action, params)
        reach = state.seq.reach[0]  # (num_assignments,)
        avoid = state.seq.avoid[0]
        assignment = self._env.map_assignment_to_index(transition.propositions)
        avoided = jnp.logical_not(jnp.any(avoid == assignment))
        reached = jnp.logical_and(jnp.any(reach == assignment), avoided)
        seq: ReachAvoidSequence = jax.lax.cond(
            reached, lambda: state.seq.advance(), lambda: state.seq
        )
        reached_end = jnp.all(seq.reach[0] == -1)  # Check if reached is just padding
        reward = jax.lax.cond(
            reached_end,
            lambda: 1.0,
            lambda: jax.lax.cond(avoided, lambda: 0.0, lambda: -1.0),
        )
        terminated = jnp.logical_or(reached_end, ~avoided)
        new_state = CurriculumState(
            state=transition.state,
            seq=seq,
            curriculum_stage=state.curriculum_stage,
        )
        return EnvTransition(
            state=new_state,
            observation=SequenceObservation.from_obs(
                transition.observation, new_state.seq
            ),
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=SequenceObservation.from_obs(
                transition.terminal_observation, new_state.seq
            ),
            propositions=transition.propositions,
            info=transition.info,
        )

    def unwrapped(self, state: Any) -> TEnvState:
        return self._env.unwrapped(state.state)
