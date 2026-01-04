from typing import override

from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import ReachAvoidSequence
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import SampleBatcher


class ReachAvoidSequenceBatcher(
    SampleBatcher[ReachAvoidSequence, JaxReachAvoidSequence]
):
    """Batches reach-avoid sequences into a JaxReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[ReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> JaxReachAvoidSequence:
        return JaxReachAvoidSequence.from_reach_avoid_seqs(samples, env)
