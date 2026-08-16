import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

with sns.color_palette(base_palette):
    # 3. Data Preparation
    # Mapping rows to [12, 8, 4, 2] based on your update
    x_values = [12, 8, 4, 2]
    methods = ["StructLTL", "GenZ-LTL", "DeepLTL", "LTL2Action"]

    # Raw data: (mean, std)
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

    df = pd.DataFrame(rows)
    # Sorting by Complexity so the bar chart flows 2 -> 12
    df = df.sort_values("Complexity")

    # 4. Plotting
    plt.figure(figsize=(8, 4))

    # Create the barplot
    ax = sns.barplot(
        data=df,
        x="Complexity",
        y="Success Rate",
        hue="Method",
        palette=base_palette,
        # edgecolor="black",
        linewidth=1,
    )

    # # 5. Add Error Bars manually (since they are pre-calculated)
    # # Extract the x-locations of the bars
    # x_coords = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
    # # Seaborn sorts bars by hue, then by x-category
    # # The order of patches in ax.patches matches the 'hue' groups
    # df_sorted_for_patches = df.sort_values(by=["Method", "Complexity"])

    # for i, patch in enumerate(ax.patches):
    #     # Get the corresponding error value
    #     std_val = df_sorted_for_patches.iloc[i]["std"]
    #     mean_val = df_sorted_for_patches.iloc[i]["Success Rate"]
    #     x_pos = patch.get_x() + patch.get_width() / 2

    #     ax.errorbar(
    #         x_pos, mean_val, yerr=std_val, fmt="none", c="black", capsize=4, elinewidth=1.2
    #     )

    # Formatting
    ax.set_title("Performance by Sequence Length")
    ax.set_ylim(0, 1.1)  # Room for error bars and legend
    ax.legend(title="Method", loc="lower left")

    plt.tight_layout()
    plt.show()
