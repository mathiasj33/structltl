from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from jaxltl.utils.plot_utils import smooth

sns.set_theme(style="darkgrid")


def load_df(path: str | Path, smooth_radius: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values(by=["seed", "timestep"])
    df["smooth_sr"] = df.groupby("seed")["metric"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    df["smooth_length"] = df.groupby("seed")["length"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    df["name"] = path
    return df


runs = [
    "struct_ltl/main",
    "deep_ltl/main",  # main10 for zoneenv
    "ltl2action/main",
    # "struct_ltl/noattn",
    # "struct_ltl/newattn",
    # "struct_ltl/attn_mod",
    # "deep_ltl/big",
    # "deep_ltl/main",
    # "struct_ltl/attn",
    # "struct_ltl/noattn",
    # "struct_ltl/attn_pos",
]

dfs = [load_df(f"runs/WarehouseEnv/{run}/eval_results_checkpoints.csv") for run in runs]
df = pd.concat(dfs, ignore_index=True)


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.lineplot(
    data=df,
    x="timestep",
    y="smooth_sr",
    hue="name",
    ax=axes[0],
    errorbar=("ci", 90),
)
axes[0].set_title("Average Success Rate")

sns.lineplot(data=df, x="timestep", y="smooth_length", hue="name", ax=axes[1])
axes[1].set_title("Average Length")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
