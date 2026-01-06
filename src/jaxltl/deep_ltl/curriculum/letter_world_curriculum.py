from pathlib import Path

from jaxltl.deep_ltl.curriculum.simple_samplers import (
    SimpleReachAvoidSampler,
)
from jaxltl.deep_ltl.utils.batching import ReachAvoidSequenceBatcher
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    RandomCurriculumStage,
)


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
            # 1. Simple reach-avoid tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=(1, 2),
                    assignments=env.assignments,
                ),
                threshold=0.9,
            ),
            # 2. Depth 2 tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=(0, 2),
                    assignments=env.assignments,
                ),
                threshold=0.9,
            ),
            # 3. Depth 3 tasks
            RandomCurriculumStage(
                sampler=SimpleReachAvoidSampler(
                    depth=3,
                    reach=(1, 2),
                    avoid=(0, 3),
                    assignments=env.assignments,
                ),
                threshold=None,
            ),
        ],
        num_samples=int(1e5),
        batcher=ReachAvoidSequenceBatcher(),
        env=env,
        load_path=load_path,
    )
