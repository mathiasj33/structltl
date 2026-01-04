import hydra
from omegaconf import DictConfig

from jaxltl import DATA_DIR
from jaxltl.deep_ltl.wrappers.ldba_wrapper import LDBAWrapper
from jaxltl.deep_ltl.wrappers.sequence_wrapper import SequenceWrapper
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = SequenceWrapper(env)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
    else:
        env = LDBAWrapper(env)
    return env
