import jax

from jaxltl.deep_ltl.curriculum.curriculum import (
    PrecomputedCurriculum,
    RandomCurriculumStage,
)
from jaxltl.deep_ltl.curriculum.zone_env_samplers import ZoneReachAvoidSampler

# TODO: fix the below
_num_assignments = (
    4  # NOTE: we assume that the empty assignment is the last one (with index 5)
)
_max_length = 3

make = lambda: PrecomputedCurriculum(
    [
        # 1. Simple reach tasks
        RandomCurriculumStage(
            sampler=ZoneReachAvoidSampler(
                depth=1,
                reach=1,
                avoid=0,
                num_assignments=_num_assignments,
                max_length=_max_length,
            ),
            threshold=0.85,
        ),
        # 2. Reach tasks of depth 2
        RandomCurriculumStage(
            sampler=ZoneReachAvoidSampler(
                depth=2,
                reach=1,
                avoid=0,
                num_assignments=_num_assignments,
                max_length=_max_length,
            ),
            threshold=0.95,
        ),
        # 3. Simple reach-avoid tasks
        RandomCurriculumStage(
            sampler=ZoneReachAvoidSampler(
                depth=1,
                reach=1,
                avoid=1,
                num_assignments=_num_assignments,
                max_length=_max_length,
            ),
            threshold=0.95,
        ),
        # 4. Reach-avoid tasks of depth 2
        RandomCurriculumStage(
            sampler=ZoneReachAvoidSampler(
                depth=2,
                reach=1,
                avoid=1,
                num_assignments=_num_assignments,
                max_length=_max_length,
            ),
            threshold=0.9,
        ),
        # 5. Mixed tasks
        RandomCurriculumStage(
            sampler=ZoneReachAvoidSampler(
                depth=(1, 2),
                reach=(1, 2),
                avoid=(0, 2),
                num_assignments=_num_assignments,
                max_length=_max_length,
            ),
            threshold=None,
        ),
    ],
    key=jax.random.key(0),
    num_samples=int(1e6),
)
