"""Benchmark StructLTL preprocessing for the Warehouse reach formula suites.

Each formula is loaded from its YAML file on every repetition.  The joblib cache
around :func:`ltl2ldba` is cleared before each construction, so LDBA timings
always include an invocation of Rabinizer rather than a cached automaton.

Example:
    pixi run python scripts/time_structltl_preprocessing.py --repetitions 10
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from jaxltl.deep_ltl.eval.preprocessing import _build_ldba
from jaxltl.deep_ltl.reach_avoid import path_search
from jaxltl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxltl.ltl.automata import ltl2ldba
from jaxltl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxltl.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)

REACH_LENGTH_PATTERN = re.compile(r"reach-(\d+)$")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMULA_GLOB = "conf/formulas/warehouse/reach-*.yaml"
DEFAULT_OUTPUT = "scripts/structltl_preprocessing_timings.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Fresh measurements per formula (default: 10).",
    )
    parser.add_argument(
        "--formula-glob",
        default=DEFAULT_FORMULA_GLOB,
        help=f"Formula files, relative to the repository root (default: {DEFAULT_FORMULA_GLOB}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Summary CSV, relative to the repository root (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def load_formulas(path: Path) -> list[str]:
    """Load a YAML formula list without retaining a parsed configuration cache."""
    formulas = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(formulas, list) or not all(isinstance(item, str) for item in formulas):
        raise ValueError(f"Expected {path} to contain a YAML list of formula strings.")
    return formulas


def formula_length(path: Path) -> int:
    match = REACH_LENGTH_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Formula file must be named reach-<length>.yaml, got {path.name}.")
    return int(match.group(1))


def generate_boolean_sequences(ldba: Any, env: WarehouseEnv) -> None:
    """Perform the non-LDBA part of StructLTL preprocessing for one formula."""
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)
    state_to_boolean_seqs = {
        state: [
            expanded_seq
            for seq in sequences
            for expanded_seq in BooleanReachAvoidSequence.from_reach_avoid_sequence(
                seq, env
            ).expand_clauses()
        ]
        for state, sequences in state_to_seqs.items()
    }
    # Construct the representation consumed by the default StructLTL model.
    JaxClauseReachAvoidSequence.from_state_to_seqs(state_to_boolean_seqs, env)


def timed_measurement(formula_path: Path, formula_index: int, repetition: int, env: WarehouseEnv) -> dict[str, object]:
    """Time one fully uncached preprocessing pass for a formula."""
    # ltl2ldba is decorated with joblib.Memory.cache.  Clearing before every pass
    # ensures the following call runs Rabinizer and cannot return a prior result.
    ltl2ldba.clear(warn=False)
    gc.collect()
    formula = load_formulas(formula_path)[formula_index]

    start = time.perf_counter()
    ldba = _build_ldba(formula, env)
    ldba_seconds = time.perf_counter() - start

    start = time.perf_counter()
    generate_boolean_sequences(ldba, env)
    boolean_seconds = time.perf_counter() - start

    return {
        "formula_length": formula_length(formula_path),
        "formula_file": str(formula_path.relative_to(ROOT)),
        "formula_index": formula_index,
        "repetition": repetition,
        "ldba_seconds": ldba_seconds,
        "boolean_generation_seconds": boolean_seconds,
    }


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for length in sorted({int(row["formula_length"]) for row in rows}):
        group = [row for row in rows if row["formula_length"] == length]
        ldba_mean, ldba_std = mean_and_sample_std(float(row["ldba_seconds"]) for row in group)
        boolean_mean, boolean_std = mean_and_sample_std(
            float(row["boolean_generation_seconds"]) for row in group
        )
        summaries.append(
            {
                "formula_length": length,
                "num_measurements": len(group),
                "ldba_mean_seconds": ldba_mean,
                "ldba_std_seconds": ldba_std,
                "boolean_generation_mean_seconds": boolean_mean,
                "boolean_generation_std_seconds": boolean_std,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> Path:
    """Show repository-relative paths where possible, otherwise keep absolute paths."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


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
    rows = []
    try:
        for formula_path in formula_paths:
            # Reload for each formula index rather than retaining the YAML content.
            num_formulas = len(load_formulas(formula_path))
            for formula_index in range(num_formulas):
                for repetition in range(args.repetitions):
                    row = timed_measurement(formula_path, formula_index, repetition, env)
                    rows.append(row)
                    print(
                        f"reach-{row['formula_length']}, formula {formula_index + 1}/{num_formulas}, "
                        f"run {repetition + 1}/{args.repetitions}: "
                        f"LDBA {float(row['ldba_seconds']):.3f}s, "
                        f"Boolean {float(row['boolean_generation_seconds']):.3f}s"
                    )
    finally:
        # Do not leave a benchmark-produced LDBA cache in the working tree.
        ltl2ldba.clear(warn=False)

    raw_output = (ROOT / args.output).resolve().with_name(
        f"{Path(args.output).stem}_raw.csv"
    )
    summary_output = (ROOT / args.output).resolve()
    write_csv(raw_output, rows)
    write_csv(summary_output, summarise(rows))
    print(f"Wrote summary to {display_path(summary_output)}")
    print(f"Wrote raw measurements to {display_path(raw_output)}")


if __name__ == "__main__":
    main()
