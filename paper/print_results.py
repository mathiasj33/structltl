import numpy as np
import pandas as pd
import yaml


def bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data = data[~np.isnan(data)]  # Remove NaN values
    rng = np.random.default_rng(42)
    bootstrap_means = np.array(
        [
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return float(np.mean(data)), float(lower), float(upper)


def main():
    env = "WarehouseEnv"
    formula_file = "conf/formulas/warehouse/rebuttal.yaml"
    runs = ["genz_ltl/rebuttal"]
    dfs = [load_df(run, env) for run in runs]
    df = pd.concat(dfs, ignore_index=True)

    # Compute bootstrap CIs for each formula and method
    results = []
    for (formula, method), group in df.groupby(["formula", "method"]):
        for col in ["return", "length"]:
            if np.isnan(group[col].values).any():
                print(
                    f"Warning: NaN values found for formula '{formula}', method '{method}', metric '{col}'"
                )
            mean, lower, upper = bootstrap_ci(group[col].values)
            results.append(
                {
                    "formula": formula,
                    "method": method,
                    "metric": col,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    agg = pd.DataFrame(results)
    print(agg)
    # agg.to_csv(
    #     f"paper/{env}_results_{'infinite' if not finite else 'finite'}.csv", index=False
    # )
    # path = Path(f"paper/{env}_results_rebuttal_cis.csv")
    # path.parent.mkdir(exist_ok=True)
    # agg.to_csv(path, index=False)
    # print(f"Wrote results to {path}")

    # Print in paper format
    print("\n" + "=" * 80)
    print("PAPER FORMAT")
    print("=" * 80)
    print_paper_format(agg, formula_file)
    print("\n" + "=" * 80)
    print_paper_format(agg, formula_file, col_name="length")


def print_paper_format(agg: pd.DataFrame, formula_file: str, col_name: str = "return"):
    """Print results in paper format: one row per formula with mean [CI lower, CI upper] for each method."""
    # Load formulas from file
    with open(formula_file) as f:
        formulas = yaml.safe_load(f)

    # Method order as specified
    # method_order = ["ltl2action", "deep_ltl", "struct_ltl"]
    method_order = ["genz_ltl"]

    # Print header
    header = "Formula" + " | " + " | ".join(method_order)
    print(header)
    print("-" * len(header))

    # Print results for each formula
    for formula in formulas:
        row_parts = [formula]
        for method in method_order:
            subset = agg[
                (agg["formula"] == formula)
                & (agg["method"] == method)
                & (agg["metric"] == col_name)
            ]
            if len(subset) == 0:
                row_parts.append("N/A")
            else:
                mean_val = subset["mean"].values[0]
                ci_lower = subset["ci_lower"].values[0]
                ci_upper = subset["ci_upper"].values[0]
                row_parts.append(
                    f"{mean_val:.2f} [-{mean_val - ci_lower:.2f}, +{ci_upper - mean_val:.2f}]"
                )
        print(" | ".join(row_parts))


def load_df(name: str, env: str) -> pd.DataFrame:
    df = pd.read_csv(f"runs/{env}/{name}/results_rebuttal.csv")
    df["method"] = name.split("/")[0]
    return df


if __name__ == "__main__":
    main()
