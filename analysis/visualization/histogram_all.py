import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from config.paths import TOPICS_PATH

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
TOPIC_GROUPS = {
    "Political": [
        "capitalism",
        "climate change",
        "covid-19",
        "immigration",
        "gaza",
        "ukraine",
    ],
    "Non-Controversial (Control)": [
        "0 basketball",
        "0 bg3",
        "0 coffee",
        "0 game development",
        "0 photography",
        "0 pokemon",
    ],
}

MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
INTERPOLATE = True


def plot_opinion_distribution(topic: str, is_interpolated_run: bool = True):
    interp_str = "interpolated" if is_interpolated_run else "filtered"

    graphml_path = (
        TOPICS_PATH
        / f"{topic}/graph/with_opinions/{topic}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.graphml"
    )
    output_plot = TOPICS_PATH / f"{topic}/eda/{topic}.png"

    if not graphml_path.exists():
        print(f"[WARNING] Graph not found for '{topic}': {graphml_path}")
        return

    print(f"Loading network from {graphml_path}...")
    G = nx.read_graphml(graphml_path)

    orig_opinions = []
    interp_opinions = []

    # Extract finalized opinions and metadata directly from the graph
    for node, data in G.nodes(data=True):
        score = data.get("url_bias_score")
        if score is None:
            continue

        # Handle GraphML boolean serialization
        is_interp = data.get("is_interpolated")
        if isinstance(is_interp, str):
            is_interp = is_interp.lower() == "true"

        if is_interp:
            interp_opinions.append(score)
        else:
            orig_opinions.append(score)

    fig, ax_hist = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("coolwarm")

    # Corrected normalization to prevent artificial right-wing saturation
    norm = mcolors.Normalize(vmin=-1.5, vmax=0.5, clip=True)

    bins = 30

    if is_interpolated_run:
        # plt.hist returns [Original_Patches, Interpolated_Patches] when stacked
        _, _, patches = ax_hist.hist(
            [orig_opinions, interp_opinions],
            bins=bins,
            range=(-1.5, 0.5),
            stacked=True,
            edgecolor="black",
            alpha=0.8,
        )

        orig_patches = patches[0]  # type: ignore
        interp_patches = patches[1]  # type: ignore

        # Apply colormap to Original data
        for patch in orig_patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))

        # Apply colormap AND hatching to Interpolated data
        for patch in interp_patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))
            patch.set_hatch("////")

        # Build custom proxy artists for an accurate legend
        legend_elements = [
            Patch(facecolor="lightgray", edgecolor="black", label="Original"),
            Patch(
                facecolor="lightgray",
                edgecolor="black",
                hatch="////",
                label="Interpolated",
            ),
        ]
        ax_hist.legend(handles=legend_elements, loc="upper right", fontsize=14)

    else:
        _, _, patches = ax_hist.hist(
            orig_opinions, bins=bins, range=(-1.5, 0.5), edgecolor="black", alpha=0.8
        )
        for patch in patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))

    ax_hist.set_title(
        f"Estimated User Opinion Distribution (Min {MIN_URL} URLs)",
        pad=12,
        fontsize=20,
    )
    ax_hist.set_xlabel("Estimated Opinions (Left -1.5 to Right 0.5)", fontsize=18)
    ax_hist.set_ylabel("Number of Users", fontsize=18)
    ax_hist.tick_params(axis="both", which="major", labelsize=14)
    ax_hist.set_xlim(-1.6, 0.6)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    os.makedirs(output_plot.parent, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    plt.close(fig)

    print(f"Visualization saved to {output_plot}")


def main():
    total = sum(len(v) for v in TOPIC_GROUPS.values())
    print(f"Starting batch opinion distribution plotting for {total} topics...\n")

    for group, topics in TOPIC_GROUPS.items():
        for topic in topics:
            plot_opinion_distribution(topic, is_interpolated_run=INTERPOLATE)

    print("\nBatch plotting complete!")


if __name__ == "__main__":
    main()
