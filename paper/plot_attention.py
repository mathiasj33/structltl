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
    path: str | Path, smooth_radius: int = 5, env_name: str = ""
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


dnf = load_df("runs/ZoneEnvComplex/struct_ltl/main/eval_results_checkpoints.csv")
gru = load_df("runs/ZoneEnvComplex/struct_ltl/gru/eval_results_checkpoints.csv")
myopic = load_df("runs/ZoneEnvComplex/struct_ltl/myopic/eval_results_checkpoints.csv")
dnf["Method"] = "Attention"
gru["Method"] = "GRU"
myopic["Method"] = "Myopic"
warehouse_df = pd.concat([dnf, gru, myopic], ignore_index=True)

# Create figure with 4 columns in a single row
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# ZoneEnv - Success Rate
sns.lineplot(
    data=warehouse_df,
    x="timestep",
    y="smooth_sr",
    hue="Method",
    style="Method",
    ax=axes[0],
    errorbar=("ci", 95),
    legend=False,
)
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Success Rate")

# ZoneEnv - Steps (Length)
sns.lineplot(
    data=warehouse_df,
    x="timestep",
    y="smooth_length",
    hue="Method",
    style="Method",
    ax=axes[1],
    errorbar=("ci", 95),
    legend=True,
)
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Steps")


# Move legend to bottom
handles, labels = axes[1].get_legend_handles_labels()
axes[1].get_legend().remove()
# Reorder legend: LTL2Action, DeepLTL, StructLTL
desired_order = ["Attention", "GRU", "Myopic"]
order_idx = [labels.index(name) for name in desired_order]
handles = [handles[i] for i in order_idx]
labels = [labels[i] for i in order_idx]
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for legend
plt.savefig("attention_curves.pdf", bbox_inches="tight", dpi=300)
plt.show()
