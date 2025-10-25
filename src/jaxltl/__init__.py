"""Public API for the jaxltl package."""

from pathlib import Path

from jaxltl.environments.registration import make
from jaxltl.hydra_utils.utils import register_custom_resolvers

register_custom_resolvers()

DATA_DIR = Path(__file__).parent.parent.parent / "data"

__all__ = ["make", "DATA_DIR"]
