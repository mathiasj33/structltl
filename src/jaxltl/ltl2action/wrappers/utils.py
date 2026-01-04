import hydra
from omegaconf import DictConfig

from jaxltl import DATA_DIR
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxltl.ltl2action.wrappers.formula_closure_wrapper import FormulaClosureWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper:
    env = FormulaClosureWrapper(env)
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
    return env
