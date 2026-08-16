"""Plot reach-task performance directly from downloaded evaluation results."""

from pathlib import Path

import numpy as np
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
RED_PALETTE_INDEX = 3
base_palette = list(sns.color_palette())
if len(base_palette) > RED_PALETTE_INDEX:
    base_palette.pop(RED_PALETTE_INDEX)  # remove red

# Define color mapping explicitly so it perfectly matches the other script's
# default color assignments (based on its loading order)
default_load_order = ["StructLTL", "DeepLTL", "LTL2Action", "GenZ-LTL"]
color_mapping = {method: base_palette[i] for i, method in enumerate(default_load_order)}

METHOD_MAPPING = {
    "struct_ltl": "StructLTL",
    "genz_ltl": "GenZ-LTL",
    "deep_ltl": "DeepLTL",
    "ltl2action": "LTL2Action",
}
DEPTHS = [2, 4, 8, 12]
DESIRED_ORDER = ["LTL2Action", "DeepLTL", "GenZ-LTL", "StructLTL"]
ENVIRONMENTS = (
    ("ZoneEnvComplex", "ZoneEnv-NM"),
    ("WarehouseEnv", "WarehouseEnv"),
)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "runs"
BOOTSTRAP_CONFIDENCE_LEVEL = 95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def load_environment_results(environment: str) -> pd.DataFrame:
    """Return one row per evaluation seed for an environment."""
    results = []
    for method, display_name in METHOD_MAPPING.items():
        for depth in DEPTHS:
            path = (
                RESULTS_ROOT
                / environment
                / method
                / "main"
                / "results"
                / f"results_reach-{depth}.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing results file: {path}")

            data = pd.read_csv(path)
            if "return" not in data:
                raise ValueError(f"Results file has no 'return' column: {path}")

            results.append(
                pd.DataFrame(
                    {
                        "Complexity": depth,
                        "Method": display_name,
                        "Success Rate": data["return"],
                    }
                )
            )

    return pd.concat(results, ignore_index=True).sort_values("Complexity")


def bootstrap_mean_confidence_interval(
    values: pd.Series, rng: np.random.Generator
) -> tuple[float, float]:
    """Calculate a percentile bootstrap confidence interval for a sample mean."""
    sample = values.to_numpy(dtype=float)
    resamples = rng.choice(sample, size=(BOOTSTRAP_RESAMPLES, len(sample)), replace=True)
    bootstrap_means = resamples.mean(axis=1)
    tail_probability = (100 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2
    lower, upper = np.percentile(
        bootstrap_means, [tail_probability, 100 - tail_probability]
    )
    return float(lower), float(upper)


def summarize_results(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate means and bootstrap intervals in the displayed plot order."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for depth in DEPTHS:
        for method in DESIRED_ORDER:
            values = data.loc[
                (data["Complexity"] == depth) & (data["Method"] == method),
                "Success Rate",
            ]
            lower, upper = bootstrap_mean_confidence_interval(values, rng)
            rows.append(
                {
                    "Complexity": depth,
                    "Method": method,
                    "Mean": values.mean(),
                    "CI Lower": lower,
                    "CI Upper": upper,
                }
            )
    return pd.DataFrame(rows)


def print_markdown_table(title: str, summary: pd.DataFrame) -> None:
    """Print a depth-by-method Markdown table of means and 95% confidence intervals."""
    print(f"\n### {title}")
    print("Mean success rate [bootstrap 95% CI]")
    print(f"| Formula depth | {' | '.join(DESIRED_ORDER)} |")
    print(f"|---:|{'|'.join(':---:' for _ in DESIRED_ORDER)}|")
    for depth in DEPTHS:
        row = summary.loc[summary["Complexity"] == depth].set_index("Method")
        cells = [
            f"{row.loc[method, 'Mean']:.2f} "
            f"[{row.loc[method, 'CI Lower']:.2f}, {row.loc[method, 'CI Upper']:.2f}]"
            for method in DESIRED_ORDER
        ]
        print(f"| {depth} | {' | '.join(cells)} |")


def add_confidence_intervals(axis: plt.Axes, summary: pd.DataFrame) -> None:
    """Add the precomputed intervals at the centres of Seaborn's grouped bars."""
    bar_containers = list(axis.containers)
    for method, bars in zip(DESIRED_ORDER, bar_containers, strict=True):
        method_summary = summary.loc[
            summary["Method"] == method
        ].sort_values("Complexity")
        means = method_summary["Mean"].to_numpy()
        lower_errors = means - method_summary["CI Lower"].to_numpy()
        upper_errors = method_summary["CI Upper"].to_numpy() - means
        centres = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        axis.errorbar(
            centres,
            means,
            yerr=(lower_errors, upper_errors),
            color=".26",
            fmt="none",
            capsize=0,
        )


# 3. Data Preparation
environment_data = [
    (
        data := load_environment_results(environment),
        title,
        summarize_results(data),
    )
    for environment, title in ENVIRONMENTS
]
for _, title, summary in environment_data:
    print_markdown_table(title, summary)

# 4. Plotting
with sns.color_palette(base_palette):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for axis, (data, title, summary) in zip(axes, environment_data, strict=True):
        sns.barplot(
            data=data,
            x="Complexity",
            y="Success Rate",
            hue="Method",
            palette=color_mapping,
            hue_order=DESIRED_ORDER,
            linewidth=1,
            errorbar=None,
            ax=axis,
        )
        add_confidence_intervals(axis, summary)
        axis.set_title(title)
        axis.set_ylim(0, 1.1)
        axis.set_xlabel("Formula Depth")

    axes[1].set_ylabel("")

    # 5. Global Legend Formatting
    # Remove individual legends
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    # Get handles/labels from one of the plots
    handles, labels = axes[0].get_legend_handles_labels()

    # Reorder legend to match the lineplot script:
    # LTL2Action, DeepLTL, GenZ-LTL, StructLTL
    order_idx = [labels.index(name) for name in DESIRED_ORDER if name in labels]
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
    plt.savefig("gen_from_results.pdf", bbox_inches="tight", dpi=300)
