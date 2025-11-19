from abc import abstractmethod
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from jaxltl.deep_ltl.curriculum.graph_sequence_sampler import GraphSequenceSampler
from jaxltl.deep_ltl.curriculum.sequence_sampler import SequenceSampler
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence


def _stage_uses_graph_sampler(stage: "CurriculumStage") -> bool:
    """Checks if a curriculum stage uses a graph-based sampler."""
    if isinstance(stage, RandomCurriculumStage):
        return isinstance(stage.sampler, GraphSequenceSampler)
    if isinstance(stage, MultiRandomStage):
        return any(_stage_uses_graph_sampler(s) for s in stage.stages)
    return False


class CurriculumStage(eqx.Module):
    threshold: float

    def __init__(self, threshold: float | None):
        if threshold is None:
            threshold = jnp.inf
        self.threshold = threshold

    @abstractmethod
    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        pass


class RandomCurriculumStage(CurriculumStage):
    """A curriculum stage in which tasks are sampled randomly."""

    sampler: SequenceSampler

    def __init__(self, sampler: SequenceSampler, threshold: float | None):
        super().__init__(threshold)
        self.sampler = sampler

    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        return self.sampler.sample(key)


class MultiRandomStage(CurriculumStage):
    """A combination of multiple RandomCurriculumStages with associated sampling probabilities."""

    stages: list[RandomCurriculumStage]
    probs: jax.Array  # shape: (num_stages,)

    def __init__(
        self,
        stages: list[RandomCurriculumStage],
        probs: list[float],
        threshold: float | None,
    ):
        super().__init__(threshold)
        self.stages = stages
        self.probs = jnp.array(probs, dtype=jnp.float32) / jnp.sum(
            jnp.array(probs, dtype=jnp.float32)
        )

    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        key, stage_keys = jax.random.split(key)
        stage_keys = jax.random.split(stage_keys, len(self.stages))
        samples = [
            stage.sample(k) for stage, k in zip(self.stages, stage_keys, strict=True)
        ]
        samples = jax.tree.map(lambda *args: jnp.stack(args), *samples)
        index = jax.random.categorical(key, jnp.log(self.probs))
        return jax.tree.map(lambda x: x[index], samples)


class Curriculum(eqx.Module):
    """A curriculum consisting of multiple curriculum stages."""

    stages: tuple[CurriculumStage, ...]

    def __init__(self, stages: list[CurriculumStage]):
        self.stages = tuple(stages)

    @eqx.filter_jit
    def sample(self, stage: jax.Array, key: jax.Array) -> JaxReachAvoidSequence:
        branches = [lambda k, s=stage: s.sample(k) for stage in self.stages]
        return jax.lax.switch(stage, branches, key)

    @eqx.filter_jit
    def threshold(self, stage: jax.Array) -> jax.Array:
        thresholds = jnp.array([s.threshold for s in self.stages], dtype=jnp.float32)
        return thresholds[stage]


class PrecomputedCurriculum(Curriculum):
    """A curriculum that precomputes samples for each stage. This leads to much faster
    training, since this avoid sampling a new sequence at each step (due to JIT).
    However, it uses moderately more memory."""

    samples: list[JaxReachAvoidSequence]  # batched samples for each stage
    num_samples: int

    def __init__(self, stages: list[CurriculumStage], key: jax.Array, num_samples: int):
        super().__init__(stages)
        self.num_samples = num_samples
        self.samples = []
        stage_keys = jax.random.split(key, len(stages))
        for i, stage in enumerate(stages):
            keys = jax.random.split(stage_keys[i], num_samples)
            if _stage_uses_graph_sampler(stage):
                # Use a Python loop for graph-based samplers that create non-JAX objects.
                samples_list = [
                    stage.sample(k)
                    for k in tqdm(
                        keys, desc=f"Precomputing graph samples for stage {i}"
                    )
                ]
                # Manually batch the JAX-compatible outputs.
                samples = jax.tree.map(lambda *x: jnp.stack(x), *samples_list)
            else:
                # Use vmap for JIT-compatible assignment-based samplers.
                samples = jax.vmap(stage.sample)(keys)
            self.samples.append(samples)

    @override
    @eqx.filter_jit
    def sample(self, stage: jax.Array, key: jax.Array) -> JaxReachAvoidSequence:
        index = jax.random.randint(key, (), 0, self.num_samples)
        return jax.lax.switch(
            stage,
            [lambda idx, s=s: jax.tree.map(lambda x: x[idx], s) for s in self.samples],
            index,
        )
