"""Public API for the jaxltl package."""

from pathlib import Path

from jaxltl.environments.registration import make

DATA_DIR = Path(__file__).parent.parent.parent / "data"

__all__ = ["make", "DATA_DIR"]
