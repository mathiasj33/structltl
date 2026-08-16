import io

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 1. Publication-ready settings
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

# 2. Palette Setup
base_palette = list(sns.color_palette())
if len(base_palette) > 3:
    base_palette.pop(3)  # remove red

# Define color mapping explicitly so it perfectly matches the other script's
# default color assignments (based on its loading order)
default_load_order = ["StructLTL", "DeepLTL", "LTL2Action", "GenZ-LTL"]
color_mapping = {method: base_palette[i] for i, method in enumerate(default_load_order)}

# 3. Data Preparation - Environment 1
x_values = [12, 8, 4, 2]
methods = ["StructLTL", "GenZ-LTL", "DeepLTL", "LTL2Action"]

raw_data = [
    [(0.771, 0.084), (0.775, 0.018), (0.635, 0.155), (0.581, 0)],  # 12
    [(0.868, 0.052), (0.861, 0.011), (0.765, 0.115), (0.741, 0)],  # 8
    [(0.930, 0.025), (0.884, 0.009), (0.867, 0.055), (0.834, 0)],  # 4
    [(0.978, 0.010), (0.964, 0.004), (0.989, 0.009), (0.953, 0)],  # 2
]

rows = []
for i, x in enumerate(x_values):
    for j, method in enumerate(methods):
        mean, std = raw_data[i][j]
        rows.append(
            {"Complexity": x, "Method": method, "Success Rate": mean, "std": std}
        )

df1 = pd.DataFrame(rows)
df1 = df1.sort_values("Complexity")

# 4. Data Preparation - Environment 2
csv_data = """Method,Depth,Mean Return
struct_ltl,2,0.9886067708333334
struct_ltl,4,0.9804036458333334
struct_ltl,8,0.9249348958333333
struct_ltl,12,0.9427083333333334
genz_ltl,2,0.9142578124999999
genz_ltl,4,0.8771484374999999
genz_ltl,8,0.8497395833333333
genz_ltl,12,0.8468098958333333
deep_ltl,2,0.9886067708333334
deep_ltl,4,0.9867838541666666
deep_ltl,8,0.9166666666666666
deep_ltl,12,0.8654947916666668
ltl2action,2,0.986421130952381
ltl2action,4,0.9812127976190476
ltl2action,8,0.8570498511904763
ltl2action,12,0.8187313988095237"""

df2 = pd.read_csv(io.StringIO(csv_data))
method_mapping = {
    "struct_ltl": "StructLTL",
    "genz_ltl": "GenZ-LTL",
    "deep_ltl": "DeepLTL",
    "ltl2action": "LTL2Action",
}
df2["Method"] = df2["Method"].map(method_mapping)
df2 = df2.rename(columns={"Depth": "Complexity", "Mean Return": "Success Rate"})
df2 = df2.sort_values("Complexity")
desired_order = ["LTL2Action", "DeepLTL", "GenZ-LTL", "StructLTL"]

# 5. Plotting
with sns.color_palette(base_palette):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Subplot 1 (Environment 1) ---
    sns.barplot(
        data=df1,
        x="Complexity",
        y="Success Rate",
        hue="Method",
        palette=color_mapping,  # Apply strict color mapping
        hue_order=desired_order,  # Ensure grouped bars appear in consistent order
        linewidth=1,
        ax=axes[0],
    )
    axes[0].set_title("ZoneEnv-NM")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xlabel("Formula Depth")

    # --- Subplot 2 (Environment 2) ---
    sns.barplot(
        data=df2,
        x="Complexity",
        y="Success Rate",
        hue="Method",
        palette=color_mapping,  # Apply strict color mapping
        hue_order=desired_order,  # Ensure grouped bars appear in consistent order
        linewidth=1,
        ax=axes[1],
    )
    axes[1].set_title("WarehouseEnv")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Formula Depth")

    # 6. Global Legend Formatting
    # Remove individual legends
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    # Get handles/labels from one of the plots
    handles, labels = axes[0].get_legend_handles_labels()

    # Reorder legend to match the lineplot script: LTL2Action, DeepLTL, GenZ-LTL, StructLTL
    order_idx = [labels.index(name) for name in desired_order if name in labels]

    handles = [handles[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]

    # Anchor the global legend beneath the subplots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the centralized bottom legend
    plt.savefig("gen.pdf", bbox_inches="tight", dpi=300)
