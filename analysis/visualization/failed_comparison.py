import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import TOPICS_PATH

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
RESULTS_DIR = (TOPICS_PATH / "_polarization_results").resolve()
INPUT_CSV_MAIN = RESULTS_DIR / "polarization_simulations_n1000.csv"
INPUT_CSV_FAILED = RESULTS_DIR / "polarization_simulations_failed_n100.csv"

# Save directly into a subfolder or your RESULTS_DIR
OUTPUT_FILES = {
    "leiden_meta": RESULTS_DIR / "delta_leiden_meta.png",
    "kern": RESULTS_DIR / "delta_kern.png",
    "leiden": RESULTS_DIR / "delta_leiden.png",
    "spectral": RESULTS_DIR / "delta_spectral.png",
}

METRICS = {
    "ac": "AC",
    "aei": "AEI",
    "rwc": "RWC",
    "r_pb_seed_only": "OPC",
}

GROUP_POL = "Political"
GROUP_CTRL = "Non-Controversial (Control)"

SHOW_SE_BARS_ON_DELTA = True


def compute_simple_differences(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for m_key, m_label in METRICS.items():
        if m_key not in df.columns:
            continue

        # 1. Average across the simulation runs per topic first
        topic_df = df.groupby(["group", "topic"], as_index=False)[m_key].mean()
        pol_vals = topic_df[topic_df["group"] == GROUP_POL][m_key].dropna().values
        ctrl_vals = topic_df[topic_df["group"] == GROUP_CTRL][m_key].dropna().values

        pol_mean = float(np.mean(pol_vals))
        ctrl_mean = float(np.mean(ctrl_vals))
        raw_diff = pol_mean - ctrl_mean

        # Standard Error of the difference between two means (Welch's formula)
        pol_var = np.var(pol_vals, ddof=1) if len(pol_vals) > 1 else 0.0
        ctrl_var = np.var(ctrl_vals, ddof=1) if len(ctrl_vals) > 1 else 0.0
        se_diff = float(np.sqrt(pol_var / len(pol_vals) + ctrl_var / len(ctrl_vals)))

        records.append(
            {
                "metric_key": m_key,
                "Metric": m_label,
                "pol_mean": pol_mean,
                "ctrl_mean": ctrl_mean,
                "raw_diff": raw_diff,
                "se_diff": se_diff,
            }
        )

    return pd.DataFrame(records)


def plot_mean_difference_bar(
    diff_df: pd.DataFrame,
    output_path,
    xlim: tuple[float, float] | None = None,
):
    """
    Plots the Mean Difference (Delta = Political Mean - Control Mean)
    for a single algorithm and saves it as a standalone PNG.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    y_pos = np.arange(len(diff_df))
    bar_colors = [
        "#d95f02" if row["metric_key"] == "r_pb_seed_only" else "#888888"
        for _, row in diff_df.iterrows()
    ]

    xerr = diff_df["se_diff"] if SHOW_SE_BARS_ON_DELTA else None

    ax.barh(
        y_pos,
        diff_df["raw_diff"],
        xerr=xerr,
        color=bar_colors,
        edgecolor="black",
        height=0.52,
        alpha=0.9,
        capsize=4 if SHOW_SE_BARS_ON_DELTA else 0,
    )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diff_df["Metric"], fontweight="bold")
    ax.invert_yaxis()

    xlabel = (
        r"$\Delta = \mu_{\mathrm{political}} - \mu_{\mathrm{nonpolitical}}$ ($\pm$ SE)"
        if SHOW_SE_BARS_ON_DELTA
        else r"$\Delta = \mu_{\mathrm{political}} - \mu_{\mathrm{nonpolitical}}$"
    )
    ax.set_xlabel(xlabel, fontweight="bold")

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        min_x = min(0.0, (diff_df["raw_diff"] - diff_df["se_diff"]).min())
        max_x = max(0.35, (diff_df["raw_diff"] + diff_df["se_diff"]).max())
        ax.set_xlim(min_x - 0.14, max_x + 0.14)

    for i, row in diff_df.iterrows():
        val = row["raw_diff"]
        se = row["se_diff"] if SHOW_SE_BARS_ON_DELTA else 0.0
        offset = (se + 0.015) if val >= 0 else -(se + 0.015)
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            i,  # type: ignore
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if not INPUT_CSV_MAIN.exists():
        print(f"Error: {INPUT_CSV_MAIN} not found.")
        return
    if not INPUT_CSV_FAILED.exists():
        print(f"Error: {INPUT_CSV_FAILED} not found.")
        return

    df_main = pd.read_csv(INPUT_CSV_MAIN)
    df_failed = pd.read_csv(INPUT_CSV_FAILED)

    # 1. Compute difference DataFrames for all 4 algorithms
    diff_dfs = {
        "leiden_meta": compute_simple_differences(df_main),
    }
    for algo in ["kern", "leiden", "spectral"]:
        sub_df = df_failed[df_failed["algorithm"] == algo]
        if sub_df.empty:
            print(
                f"[WARNING] No rows found for algorithm '{algo}' in {INPUT_CSV_FAILED}"
            )
            continue
        diff_dfs[algo] = compute_simple_differences(sub_df)

    # 2. Compute shared global x-axis limits for direct visual comparability
    global_min = min(
        0.0,
        min((d["raw_diff"] - d["se_diff"]).min() for d in diff_dfs.values()),
    )
    global_max = max(
        0.35,
        max((d["raw_diff"] + d["se_diff"]).max() for d in diff_dfs.values()),
    )
    shared_xlim = (global_min - 0.14, global_max + 0.14)

    # 3. Generate and save each of the 4 plots
    for algo_key, d_df in diff_dfs.items():
        plot_mean_difference_bar(
            d_df,
            output_path=OUTPUT_FILES[algo_key],
            xlim=shared_xlim,
        )


if __name__ == "__main__":
    main()
