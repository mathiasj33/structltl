"""Benchmark StructLTL forward passes for Warehouse reach formula depths.

For each ``reach-<depth>.yaml`` file, one formula is selected with a seeded
random generator.  Its first sequence from the initial LDBA state is paired
with a fixed Warehouse observation and evaluated repeatedly after JIT warm-up.

Example:
    pixi run python scripts/time_structltl_forward.py
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from collections.abc import Iterable
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from time_structltl_preprocessing import (
    DEFAULT_FORMULA_GLOB,
    ROOT,
    display_path,
    formula_length,
    load_formulas,
)

from jaxltl.deep_ltl.wrappers.curriculum_wrapper import SequenceObservation
from jaxltl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxltl.struct_ltl.eval.preprocessing import preprocess_formulas
from jaxltl.struct_ltl.model.struct_ltl import StructLTLModel

DEFAULT_OUTPUT = "scripts/structltl_forward_timings.csv"
DEFAULT_REPETITIONS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Timed forward passes per depth after warm-up (default: {DEFAULT_REPETITIONS}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Formula-selection seed.")
    parser.add_argument("--formula-glob", default=DEFAULT_FORMULA_GLOB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


@eqx.filter_jit
def forward(model: StructLTLModel, observation: SequenceObservation):
    """Run the complete StructLTL actor--critic forward pass for one observation."""
    distribution, value = model(observation)
    # Return arrays rather than the Distrax distribution object so Equinox can
    # safely JIT and synchronize the complete actor--critic computation.
    return distribution.mode(), value


def build_model(env: WarehouseEnv) -> StructLTLModel:
    """Construct the Warehouse StructLTL architecture used by the experiment config."""
    with hydra.initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name="experiment/struct_ltl/warehouse")
    model_factory = hydra.utils.instantiate(
        cfg.model,
        obs_shape=env.observation_space(env.default_params).shape,
        num_assignments=len(env.assignments()),
        num_propositions=len(env.propositions),
        env_params=env.default_params,
        key=jax.random.key(0),
        _partial_=True,
    )
    return model_factory(act_space=env.action_space(env.default_params))


def initial_sequence_observation(
    formula: str, env: WarehouseEnv, key: jax.Array
) -> SequenceObservation:
    """Create a batch-one observation with the first sequence of the initial state."""
    ldba, state_to_seqs = preprocess_formulas([formula], env)
    initial_state = int(ldba.initial_state[0])
    sequence = jax.tree.map(
        lambda value: value[0, initial_state, 0], state_to_seqs
    )
    _, observation = env.reset(key, None, env.default_params)
    batched_observation = jax.tree.map(lambda value: value[None, ...], observation)
    return SequenceObservation.from_obs(
        batched_observation,
        jax.tree.map(lambda value: value[None, ...], sequence),
        epsilon_enabled=jnp.zeros((1,), dtype=bool),
    )


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least one.")

    formula_paths = sorted(
        (ROOT / args.formula_glob).parent.glob(Path(args.formula_glob).name),
        key=formula_length,
    )
    if not formula_paths:
        raise FileNotFoundError(f"No formula files matched {args.formula_glob!r}.")

    env = WarehouseEnv()
    model = build_model(env)
    formula_rng = random.Random(args.seed)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for depth_index, formula_path in enumerate(formula_paths):
        formulas = load_formulas(formula_path)
        formula_index = formula_rng.randrange(len(formulas))
        observation = initial_sequence_observation(
            formulas[formula_index], env, jax.random.key(depth_index + 1)
        )

        # Compile and execute one pass before collecting any timings.
        jax.block_until_ready(forward(model, observation))
        timings = []
        for repetition in range(args.repetitions):
            start = time.perf_counter()
            jax.block_until_ready(forward(model, observation))
            seconds = time.perf_counter() - start
            timings.append(seconds)
            raw_rows.append(
                {
                    "formula_length": formula_length(formula_path),
                    "formula_file": str(formula_path.relative_to(ROOT)),
                    "formula_index": formula_index,
                    "repetition": repetition,
                    "forward_seconds": seconds,
                }
            )

        mean, std = mean_and_sample_std(timings)
        summary_rows.append(
            {
                "formula_length": formula_length(formula_path),
                "formula_file": str(formula_path.relative_to(ROOT)),
                "formula_index": formula_index,
                "num_measurements": args.repetitions,
                "forward_mean_seconds": mean,
                "forward_std_seconds": std,
            }
        )
        print(
            f"reach-{formula_length(formula_path)}, formula {formula_index + 1}/{len(formulas)}: "
            f"{mean * 1_000:.3f} ms mean over {args.repetitions} runs"
        )

    summary_output = (ROOT / args.output).resolve()
    raw_output = summary_output.with_name(f"{summary_output.stem}_raw.csv")
    write_csv(raw_output, raw_rows)
    write_csv(summary_output, summary_rows)
    print(f"Wrote summary to {display_path(summary_output)}")
    print(f"Wrote raw measurements to {display_path(raw_output)}")


if __name__ == "__main__":
    main()
