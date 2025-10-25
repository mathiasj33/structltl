from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, "same") / np.convolve(z, y, "same")


def load_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bin_size = 4096 * 16
    df["bin"] = df["timestep"] // bin_size
    avg_returns = df.groupby(["seed", "bin"])["return"].mean().reset_index()
    avg_returns["timestep"] = avg_returns["bin"] * bin_size

    for seed in avg_returns["seed"].unique():
        mask = avg_returns["seed"] == seed
        avg_returns.loc[mask, "smooth_return"] = smooth(
            avg_returns.loc[mask, "return"], radius=10
        )

    avg_returns["name"] = Path(path).parent.name
    return avg_returns


dfs = [
    load_df(p)
    for p in [
        # "runs/ZoneEnv/tmp/logs.csv",
        # "runs/ZoneEnv/tmp2/logs.csv",
        # "runs/ZoneEnv/newstate/logs.csv",
        # "runs/ZoneEnv/nocorrect/logs.csv",
        "runs/ZoneEnv/sequential_long/logs.csv",
    ]
]
df = pd.concat(dfs, ignore_index=True)


# df["return"] = smooth(df["return"], 10)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(data=df, x="timestep", y="smooth_return", hue="name")
plt.show()
