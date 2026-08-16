import numpy as np
import pandas as pd
import yaml


def get_latex_header():
    return r"""\begin{table*}[t]
    \caption{Caption}
    \label{tab:main-results}
    \begin{center}
        \begin{small}

            \newcommand{\metric}[3]{%
                $#1_{\scriptscriptstyle #2}^{\scriptscriptstyle #3}$%
            }

            % Columns adjusted for new GenZ-LTL entries
            % 1: Multirow label
            % 2: \phi label
            % 3-6: Success Rate (L2A, Deep, GenZ, Struct)
            % 7-10: Mean Steps (L2A, Deep, GenZ, Struct)
            % 11: Psi Label
            % 12-14: Average Visits (Deep, GenZ, Struct)
            \resizebox{\textwidth}{!}{%
                \begin{tabular}{ll rrrr rrrr l rrr}
                    \toprule
                     & \multicolumn{9}{c}{\textbf{Finite Horizon}} & \multicolumn{4}{c}{\textbf{Infinite Horizon}}                                                                                                                                                                                                                                                  \\
                    \cmidrule(lr){2-10} \cmidrule(l){11-14}

                     &                                             & \multicolumn{4}{c}{{Success Rate} ($\uparrow$)} & \multicolumn{4}{c}{{Average Steps} ($\downarrow$)} &                                             & \multicolumn{3}{c}{{Average Visits ($\uparrow$)}}                                                                                         \\
                    \cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(l){12-14}

                     & $\varphi$                                   & LTL2Action                                      & DeepLTL                                            & GenZ-LTL                                    & StructLTL                                         & LTL2Action & DeepLTL & GenZ-LTL & StructLTL & $\psi$ & DeepLTL & GenZ-LTL & StructLTL \\
                    \midrule"""


def get_latex_footer():
    return r"""                    \bottomrule
                \end{tabular}
            }
        \end{small}
    \end{center}
\end{table*}"""


def format_metric(
    mean,
    lower,
    upper,
    is_best,
    is_zero=False,
    round_to_int=False,
    pad_ci=False,
):
    if is_zero or (mean == 0.0 and lower == 0.0 and upper == 0.0):
        return r"\metric{0.00}{-0.00}{+0.00}"

    diff_lower = lower - mean
    diff_upper = upper - mean

    # Format mean, handling bolding for best results
    mean_str = f"{mean:.1f}" if round_to_int else f"{mean:.2f}"
    if is_best:
        mean_str = f"\\textbf{{{mean_str}}}"

    # Enforce strict - and + signs based on the example
    if round_to_int:
        diff_lower_int = int(round(abs(diff_lower)))
        diff_upper_int = int(round(abs(diff_upper)))
        if pad_ci:
            lower_str = f"-{diff_lower_int:03d}"
            upper_str = f"+{diff_upper_int:03d}"
        else:
            lower_str = f"-{diff_lower_int:d}"
            upper_str = f"+{diff_upper_int:d}"
    elif pad_ci:
        lower_str = f"-{abs(diff_lower):05.2f}"
        upper_str = f"+{abs(diff_upper):05.2f}"
    else:
        lower_str = f"-{abs(diff_lower):.2f}"
        upper_str = f"+{abs(diff_upper):.2f}"

    return f"\\metric{{{mean_str}}}{{{lower_str}}}{{{upper_str}}}"


def get_stats(df, env, finite, formula, method, metric):
    row = df[
        (df["env"] == env)
        & (df["finite"] == finite)
        & (df["formula"] == formula)
        & (df["method"] == method)
        & (df["metric"] == metric)
    ]
    if row.empty:
        return 0.0, 0.0, 0.0
    return row.iloc[0]["mean"], row.iloc[0]["ci_lower"], row.iloc[0]["ci_upper"]


def process_block(df, env, finite, formula, metric, methods, higher_is_better):
    stats = []
    for m in methods:
        stats.append(get_stats(df, env, finite, formula, m, metric))

    # Determine the best value for bolding
    valid_means = [s[0] for s in stats if s[0] > 0.0]
    best_val = None
    if valid_means:
        best_val = max(valid_means) if higher_is_better else min(valid_means)

    latex_strs = []
    for mean, lower, upper in stats:
        is_best = (best_val is not None) and np.isclose(mean, best_val, atol=1e-4)
        latex_strs.append(
            format_metric(
                mean,
                lower,
                upper,
                is_best,
                round_to_int=(metric == "length"),
                pad_ci=(metric == "length" or not finite),
            )
        )

    return " & ".join(latex_strs)


def generate_env_rows(
    df, csv_env_name, label_env_name, finite_formulas, infinite_formulas, start_idx
):
    methods_finite = ["ltl2action", "deep_ltl", "genz_ltl", "struct_ltl"]
    methods_infinite = ["deep_ltl", "genz_ltl", "struct_ltl"]

    n_rows = len(finite_formulas)
    lines = [
        f"                    % ================= {label_env_name.upper()} =================",
        f"                    \\multirow{{{n_rows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{label_env_name}}}}}",
    ]

    for i, (f_fin, f_inf) in enumerate(
        zip(finite_formulas, infinite_formulas, strict=False)
    ):
        idx = start_idx + i
        lines.append("")

        # 1. Success Rate block (Finite, Return)
        success_str = process_block(
            df,
            csv_env_name,
            True,
            f_fin,
            "return",
            methods_finite,
            higher_is_better=True,
        )

        # 2. Average Steps block (Finite, Length)
        steps_str = process_block(
            df,
            csv_env_name,
            True,
            f_fin,
            "length",
            methods_finite,
            higher_is_better=False,
        )

        # 3. Average Visits block (Infinite, Return)
        if f_inf != "asd":
            visits_str = process_block(
                df,
                csv_env_name,
                False,
                f_inf,
                "return",
                methods_infinite,
                higher_is_better=True,
            )
            psi_label = f"$\\psi_{{{idx}}}$"
        else:
            # Fallback for padded missing formulas
            visits_str = " & ".join([format_metric(0, 0, 0, False, True)] * 3)
            psi_label = ""

        phi_label = f"$\\varphi_{{{idx}}}$"

        # Format the row snippet
        row_str = (
            f"                     & {phi_label}\n"
            f"                     & {success_str}\n"
            f"                     & {steps_str}\n"
            f"                     & {psi_label} & {visits_str} \\\\"
        )
        lines.append(row_str)

    return "\n".join(lines), start_idx + n_rows


def main():
    # 1. Load Formulas (Simulated missing file handling for robust usage)
    try:
        with open("conf/formulas/zones_complex/finite.yaml") as f:
            zones_finite = yaml.safe_load(f)
        with open("conf/formulas/warehouse/finite.yaml") as f:
            warehouse_finite = yaml.safe_load(f)
        with open("conf/formulas/zones_complex/infinite.yaml") as f:
            zones_infinite = yaml.safe_load(f)
        with open("conf/formulas/warehouse/infinite.yaml") as f:
            warehouse_infinite = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Warning: Ensure your yaml config files exist. {e}")
        return

    # Pad missing infinite formulas
    if len(zones_infinite) < len(zones_finite):
        zones_infinite = zones_infinite + ["asd"] * (
            len(zones_finite) - len(zones_infinite)
        )
    if len(warehouse_infinite) < len(warehouse_finite):
        warehouse_infinite = warehouse_infinite + ["asd"] * (
            len(warehouse_finite) - len(warehouse_infinite)
        )

    # 2. Load Data
    results = pd.read_csv("paper/main_table_results.csv")

    # 3. Build Table
    latex_table = []
    latex_table.append(get_latex_header())

    # Track the formula index globally
    current_idx = 1

    # Generate ZoneEnv Rows
    # Assuming the env name in CSV is "ZoneEnvComplex" based on your example
    zone_rows, current_idx = generate_env_rows(
        results,
        "ZoneEnvComplex",
        "ZoneEnv-NM",
        zones_finite,
        zones_infinite,
        current_idx,
    )
    latex_table.append(zone_rows)
    latex_table.append(r"                    \midrule")

    # Generate Warehouse Rows
    # Assuming the env name in CSV is "Warehouse" or similar. Update if needed.
    warehouse_rows, current_idx = generate_env_rows(
        results,
        "WarehouseEnv",
        "WarehouseEnv",
        warehouse_finite,
        warehouse_infinite,
        current_idx,
    )
    latex_table.append(warehouse_rows)

    latex_table.append(get_latex_footer())

    # 4. Output Result
    final_latex = "\n".join(latex_table)
    print(final_latex)

    # Optional: Save to file
    with open("paper/generated_main_table.tex", "w") as f:
        f.write(final_latex)
    with open(
        "/home/mathias/work/dphil/structltl-neurips/tables/main_table.tex", "w"
    ) as f:
        f.write(final_latex)
    print(
        "\nTable successfully generated and saved to 'paper/generated_main_table.tex'"
    )


if __name__ == "__main__":
    main()
