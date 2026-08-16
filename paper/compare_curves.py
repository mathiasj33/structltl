from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from jaxltl.utils.plot_utils import smooth

# Publication-ready settings
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_df(
    path: str | Path,
    smooth_radius: int = 10,
    name: str | None = None,
    env_name: str = "",
) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values(by=["seed", "timestep"])
    df["smooth_sr"] = df.groupby("seed")["metric"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    df["smooth_length"] = df.groupby("seed")["length"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    if name:
        df["Method"] = name
    else:
        df["Method"] = path.replace("runs/ZoneEnv/", "")
    df["Environment"] = env_name
    return df


zone_runs = [
    # "struct_ltl/main",
    # "struct_ltl/vf",
    # "deep_ltl/main10",  # Use main10 for ZoneEnv
    # "ltl2action/main",
    # "genz_ltl/main",
    # "genz_ltl/onehot",
    # "genz_ltl/nowall",
    # "genz_ltl/onehot",
]

zone_runs = ["struct_ltl/main", "struct_ltl/gru", "struct_ltl/myopic"]

# Load data for both environments
zone_dfs = [
    load_df(
        f"runs/ZoneEnvComplex/{run}/eval_results_checkpoints.csv",
        smooth_radius=5,
        env_name="ZoneEnv",
    )
    for run in zone_runs
]

zone_df = pd.concat(zone_dfs, ignore_index=True)

# Create figure with 4 columns in a single row
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# ZoneEnv - Success Rate
sns.lineplot(
    data=zone_df,
    x="timestep",
    y="smooth_sr",
    hue="Method",
    ax=axes[0],
    errorbar=("ci", 95),
    legend=True,
)
axes[0].set_title("ZoneEnv")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Success Rate")

# ZoneEnv - Steps (Length)
sns.lineplot(
    data=zone_df,
    x="timestep",
    y="smooth_length",
    hue="Method",
    ax=axes[1],
    errorbar=("ci", 95),
    legend=False,
)
axes[1].set_title("ZoneEnv")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Steps")

# Move legend to bottom
handles, labels = axes[0].get_legend_handles_labels()
legend = axes[0].get_legend()
if legend is not None:
    legend.remove()
# Reorder legend: LTL2Action, DeepLTL, StructLTL
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.3))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for legend
plt.savefig("compare.pdf", bbox_inches="tight", dpi=300)
# plt.show()
