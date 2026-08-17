from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxltl.environments.zone_env_nm.zone_env_nm import ZoneEnvNM

_name_to_env = {
    "ZoneEnv-NM": ZoneEnvNM,
    "WarehouseEnv": WarehouseEnv,
}


def make(name: str, **kwargs) -> tuple[Environment, EnvParams]:
    """Create an environment by name.

    Returns:
        A tuple of the environment instance and its default parameters."""
    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class(**kwargs)
    return env, env.default_params  # type: ignore
