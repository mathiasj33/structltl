from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def compute_cis(env: str, finite: bool):
    runs = ["genz_ltl/main", "struct_ltl/main"]
    if not finite:
        runs.pop(0)  # Remove LTL2Action for infinite horizon
    dfs = [load_df(run, env, finite) for run in runs]
    df = pd.concat(dfs, ignore_index=True)

    # Compute bootstrap CIs for each formula and method
    results = []
    for (formula, method), group in df.groupby(["formula", "method"]):
        for col in ["return", "violations"]:
            if np.isnan(group[col].values).any() and col != "length":
                print(
                    f"Warning: NaN values found for formula '{formula}', method '{method}', metric '{col}'"
                )
            mean, lower, upper = bootstrap_ci(np.array(group[col].values))
            results.append(
                {
                    "env": env,
                    "finite": finite,
                    "formula": formula,
                    "method": method,
                    "metric": col,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(results)


def main():
    args = [
        ("ZoneEnvComplex", True),
        ("WarehouseEnv", True),
    ]
    dfs = []
    for env, finite in tqdm(args, desc="Computing CIs"):
        df = compute_cis(env, finite)
        dfs.append(df)
    agg = pd.concat(dfs, ignore_index=True)

    path = Path("paper/violations.csv")
    path.parent.mkdir(exist_ok=True)
    agg.to_csv(path, index=False)
    print(f"Wrote results to {path}")


def load_df(name: str, env: str, finite: bool) -> pd.DataFrame:
    df = pd.read_csv(f"runs/{env}/{name}/results_violations.csv")
    df["method"] = name.split("/")[0]
    return df


if __name__ == "__main__":
    main()
