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

# Skip the default red and keep a stable palette length.
base_palette = list(sns.color_palette())
if len(base_palette) > 3:
    base_palette.pop(3)  # remove red from the palette

line_styles = [
    (1, 0),  # Solid "-"
    (4, 1.5, 1, 1.5),  # Dash-dot "-."
    (1, 1),  # Dotted ":"
    (4, 1.5),  # Dashed "--"
]

with sns.color_palette(base_palette):

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
        # Extract run name from path
        method_name = Path(path).parent.parent.name
        method_map = {
            "struct_ltl": "StructLTL",
            "deep_ltl": "DeepLTL",
            "ltl2action": "LTL2Action",
            "genz_ltl": "GenZ-LTL",
        }
        if name:
            df["Method"] = name
        else:
            df["Method"] = method_map.get(method_name, method_name)
        df["Environment"] = env_name
        return df

    warehouse_runs = [
        "struct_ltl/main",
        "deep_ltl/main",
        "ltl2action/main",
        "genz_ltl/main",
    ]

    zone_runs = [
        "struct_ltl/main",
        "deep_ltl/main",
        "ltl2action/main",
        "genz_ltl/main",
    ]

    # Load data for both environments
    zone_dfs = [
        load_df(
            f"runs/ZoneEnvComplex/{run}/eval_results_checkpoints.csv",
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
        for run in warehouse_runs
    ]

    zone_df = pd.concat(zone_dfs, ignore_index=True)
    warehouse_df = pd.concat(warehouse_dfs, ignore_index=True)

    # Create figure with 4 columns in a single row
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    errorbar = ("ci", 95)
    # errorbar = "sd"

    # ZoneEnv - Success Rate
    sns.lineplot(
        data=zone_df,
        x="timestep",
        y="smooth_sr",
        hue="Method",
        style="Method",
        ax=axes[0],
        errorbar=errorbar,
        legend=True,
        palette=base_palette,
        dashes=line_styles,
    )
    axes[0].set_title("ZoneEnv-NM")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Success Rate")

    # ZoneEnv - Steps (Length)
    sns.lineplot(
        data=zone_df,
        x="timestep",
        y="smooth_length",
        hue="Method",
        style="Method",
        ax=axes[1],
        errorbar=errorbar,
        legend=False,
        palette=base_palette,
        dashes=line_styles,
    )
    axes[1].set_title("ZoneEnv-NM")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Steps")

    # WarehouseEnv - Success Rate
    sns.lineplot(
        data=warehouse_df,
        x="timestep",
        y="smooth_sr",
        hue="Method",
        style="Method",
        ax=axes[2],
        errorbar=errorbar,
        legend=False,
        palette=base_palette,
        dashes=line_styles,
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
        style="Method",
        ax=axes[3],
        errorbar=errorbar,
        legend=False,
        palette=base_palette,
        dashes=line_styles,
    )
    axes[3].set_title("WarehouseEnv")
    axes[3].set_xlabel("Timestep")
    axes[3].set_ylabel("Steps")

    # Move legend to bottom
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    # Reorder legend: LTL2Action, DeepLTL, StructLTL
    desired_order = ["LTL2Action", "DeepLTL", "GenZ-LTL", "StructLTL"]
    order_idx = [labels.index(name) for name in desired_order if name in labels]
    handles = [handles[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]
    for handle, label in zip(handles, labels, strict=False):
        if label in line_styles and hasattr(handle, "set_linestyle"):
            handle.set_linestyle(line_styles[label])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(warehouse_runs),
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.savefig("training_curves.pdf", bbox_inches="tight", dpi=300)
    # plt.show()
