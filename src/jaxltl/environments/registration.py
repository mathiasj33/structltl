from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.zone_env.zone_env import ZoneEnv

_name_to_env = {"ZoneEnv": ZoneEnv}


def make(name: str) -> tuple[Environment, EnvParams]:
    """Create an environment by name.

    Returns:
        A tuple of the environment instance and its default parameters."""
    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class()
    return env, env.default_params  # type: ignore
