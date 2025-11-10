"""Serialization utilities for PyTrees with optional metadata."""

import json
from pathlib import Path

import equinox as eqx
from jaxtyping import PyTree


def save(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with optional metadata to a file.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    with open(path, "wb") as f:
        if not metadata:
            metadata = {}
        f.write(json.dumps(metadata, indent=None).encode("utf-8"))
        f.write(b"\n")
        eqx.tree_serialise_leaves(f, model)


def load_metadata(path: Path | str) -> dict:
    """Load metadata from a file.

    Args:
        path (Path): The path to the file from which to load the metadata.

    Returns:
        dict: The loaded metadata.
    """
    with open(path, "rb") as f:
        metadata = json.loads(f.readline().decode("utf-8"))
    return metadata


def load(path: Path | str, template: PyTree) -> PyTree:
    """Load a PyTree from a file.

    Args:
        path (Path): The path to the file from which to load the PyTree.
        template (PyTree): A template PyTree with the same structure as the one being loaded.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        f.readline()  # Discard metadata line
        model = eqx.tree_deserialise_leaves(f, template)
    return model
