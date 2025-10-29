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

    avg_data = df.groupby(["seed", "bin"])[["return", "length"]].mean().reset_index()

    avg_data["timestep"] = avg_data["bin"] * bin_size

    for seed in avg_data["seed"].unique():
        mask = avg_data["seed"] == seed
        avg_data.loc[mask, "smooth_return"] = smooth(
            avg_data.loc[mask, "return"], radius=10
        )
        avg_data.loc[mask, "smooth_length"] = smooth(
            avg_data.loc[mask, "length"], radius=10
        )

    avg_data["name"] = Path(path).parent.name
    return avg_data


dfs = [
    load_df(p)
    for p in [
        "runs/ZoneEnv/gru/logs.csv",
        "runs/ZoneEnv/noterm/logs.csv",
    ]
]
df = pd.concat(dfs, ignore_index=True)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.lineplot(data=df, x="timestep", y="return", hue="name", ax=axes[0], legend=False)
axes[0].set_title("Average Return")
axes[0].set_ylabel("Smoothed Return")

sns.lineplot(data=df, x="timestep", y="length", hue="name", ax=axes[1])
axes[1].set_title("Average Episode Length")
axes[1].set_ylabel("Smoothed Length")

# Move the legend outside the plot area
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
