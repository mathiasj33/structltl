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
    last_returns: jax.Array  # shape (N,), returns from last N episodes
    returns_index: jax.Array  # int, index to write next return into last_returns
    # int, number of completed episodes in the current stage
    current_stage_episodes: jax.Array


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
    episode_window: int  # number of episodes to consider for average return

    def __init__(
        self,
        env: EnvWrapper[TEnvState, TEnvParams, TObsFeatures]
        | Environment[TEnvState, TEnvParams, TObsFeatures],
        curriculum: Curriculum,
        episode_window: int,
    ):
        super().__init__(env)
        self.curriculum = curriculum
        self.episode_window = episode_window

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: CurriculumState[TEnvState] | None,
        params: TEnvParams,
    ) -> tuple[CurriculumState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().reset(reset_key, state.state if state else None, params)
        state = self._wrap_reset_state(state, re_state, sample_key)
        return state, SequenceObservation.from_obs(obs, state.seq)

    @eqx.filter_jit
    def cheap_reset(
        self, key: jax.Array, state: CurriculumState[TEnvState], params: TEnvParams
    ) -> tuple[CurriculumState[TEnvState], SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().cheap_reset(reset_key, state.state, params)
        state = self._wrap_reset_state(state, re_state, sample_key)
        return state, SequenceObservation.from_obs(obs, state.seq)

    def _wrap_reset_state(
        self,
        state: CurriculumState[TEnvState] | None,
        re_state: TEnvState,
        key: jax.Array,
    ) -> CurriculumState[TEnvState]:
        if state is None:
            stage = jnp.zeros((), dtype=jnp.int32)
            return CurriculumState(
                state=re_state,
                seq=self.curriculum.sample(stage, key),
                curriculum_stage=stage,
                last_returns=jnp.zeros((self.episode_window,), dtype=jnp.float32),
                returns_index=jnp.zeros((), dtype=jnp.int32),
                current_stage_episodes=jnp.zeros((), dtype=jnp.int32),
            )
        threshold = self.curriculum.threshold(state.curriculum_stage)
        avg_return = jax.lax.cond(
            state.current_stage_episodes < self.episode_window,
            lambda: -jnp.inf,
            lambda: jnp.mean(state.last_returns),
        )
        change_stage = avg_return >= threshold
        stage = jax.lax.cond(
            change_stage,
            lambda: state.curriculum_stage + 1,
            lambda: state.curriculum_stage,
        )
        last_returns = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((self.episode_window,), dtype=jnp.float32),
            lambda: state.last_returns,
        )
        current_stage_episodes = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((), dtype=jnp.int32),
            lambda: state.current_stage_episodes,
        )
        returns_index = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((), dtype=jnp.int32),
            lambda: state.returns_index,
        )
        seq = self.curriculum.sample(stage, key)
        return CurriculumState(
            state=re_state,
            seq=seq,
            curriculum_stage=stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=current_stage_episodes,
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
        new_returns = state.last_returns.at[state.returns_index].set(
            jax.nn.relu(reward)  # binary success indicator
        )
        last_returns = jnp.where(terminated, new_returns, state.last_returns)
        returns_index = jnp.where(
            terminated,
            (state.returns_index + 1) % self.episode_window,
            state.returns_index,
        )
        num_episodes = state.current_stage_episodes + terminated.astype(jnp.int32)
        new_state = CurriculumState(
            state=transition.state,
            seq=seq,
            curriculum_stage=state.curriculum_stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=num_episodes,
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
