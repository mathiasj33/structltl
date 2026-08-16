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
    path: str | Path, smooth_radius: int = 10, env_name: str = ""
) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values(by=["seed", "timestep"])
    df["smooth_sr"] = df.groupby("seed")["metric"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    df["smooth_length"] = df.groupby("seed")["length"].transform(
        lambda x: smooth(x, radius=smooth_radius)
    )
    # Extract run name from path
    method_name = Path(path).parent.parent.name
    method_map = {
        "struct_ltl": "StructLTL",
        "deep_ltl": "DeepLTL",
        "ltl2action": "LTL2Action",
    }
    df["Method"] = method_map.get(method_name, method_name)
    df["Environment"] = env_name
    return df


runs = [
    "struct_ltl/main",
    "deep_ltl/main",
    "ltl2action/main",
]

zone_runs = [
    "struct_ltl/main",
    "deep_ltl/main10",  # Use main10 for ZoneEnv
    "ltl2action/main",
]

# Load data for both environments
zone_dfs = [
    load_df(
        f"runs/ZoneEnv/{run}/eval_results_checkpoints.csv",
        smooth_radius=5,
        env_name="ZoneEnv",
    )
    for run in zone_runs
]
warehouse_dfs = [
    load_df(
        f"runs/WarehouseEnv/{run}/eval_results_checkpoints.csv",
        smooth_radius=10,
        env_name="WarehouseEnv",
    )
    for run in runs
]

zone_df = pd.concat(zone_dfs, ignore_index=True)
warehouse_df = pd.concat(warehouse_dfs, ignore_index=True)

# Create figure with 4 columns in a single row
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# ZoneEnv - Success Rate
sns.lineplot(
    data=zone_df,
    x="timestep",
    y="smooth_sr",
    hue="Method",
    ax=axes[0],
    errorbar=("ci", 90),
    legend=False,
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
    errorbar=("ci", 90),
    legend=False,
)
axes[1].set_title("ZoneEnv")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Steps")

# WarehouseEnv - Success Rate
sns.lineplot(
    data=warehouse_df,
    x="timestep",
    y="smooth_sr",
    hue="Method",
    ax=axes[2],
    errorbar=("ci", 90),
    legend=False,
)
axes[2].set_title("WarehouseEnv")
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Success Rate")

# WarehouseEnv - Steps (Length)
sns.lineplot(
    data=warehouse_df,
    x="timestep",
    y="smooth_length",
    hue="Method",
    ax=axes[3],
    errorbar=("ci", 90),
)
axes[3].set_title("WarehouseEnv")
axes[3].set_xlabel("Timestep")
axes[3].set_ylabel("Steps")

# Move legend to bottom
handles, labels = axes[3].get_legend_handles_labels()
axes[3].get_legend().remove()
# Reorder legend: LTL2Action, DeepLTL, StructLTL
desired_order = ["LTL2Action", "DeepLTL", "StructLTL"]
order_idx = [labels.index(name) for name in desired_order]
handles = [handles[i] for i in order_idx]
labels = [labels[i] for i in order_idx]
fig.legend(
    handles, labels, loc="lower center", ncol=len(runs), bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for legend
plt.savefig("training_curves.pdf", bbox_inches="tight", dpi=300)
# plt.show()
