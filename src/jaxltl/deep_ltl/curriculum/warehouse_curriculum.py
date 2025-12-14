from pathlib import Path

import jax

from jaxltl.deep_ltl.curriculum.curriculum import (
    PrecomputedCurriculum,
    RandomCurriculumStage,
)
from jaxltl.deep_ltl.curriculum.warehouse_samplers import WarehousePropSampler
from jaxltl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxltl.ltl.logic.assignment import Assignment

propositions = WarehouseEnv.propositions
regions = [  # TODO: make static and clean up
    frozenset({"region_a"}),
    frozenset({"region_b"}),
    frozenset({"door"}),
    frozenset(),
]
items = [
    frozenset({"vase"}),
    frozenset({"crate"}),
    frozenset({"vase", "crate"}),
    frozenset(),
]
assignments = []
for r in regions:
    for i in items:
        assignments.append(Assignment(r | i))

_max_length = 1
_max_nodes = WarehouseEnv.max_nodes
_max_edges = WarehouseEnv.max_edges


def make(load_path: str | Path | None = None):
    return PrecomputedCurriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=WarehousePropSampler(
                    propositions=propositions,
                    assignments=assignments,
                    max_length=_max_length,
                    max_nodes=_max_nodes,
                    max_edges=_max_edges,
                ),
                threshold=None,
            ),
        ],
        key=jax.random.key(0),
        num_samples=int(1e3),
        load_path=load_path,
    )
