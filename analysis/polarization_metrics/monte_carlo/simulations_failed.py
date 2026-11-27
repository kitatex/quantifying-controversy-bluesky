import warnings
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from scipy.sparse.linalg import eigsh
import igraph as ig
import leidenalg
from tqdm import tqdm

from config.paths import TOPICS_PATH

from analysis.polarization_metrics.welch_t_test import compute_welch_arrays
from analysis.partition.leiden_meta import (
    get_opinion_graph_path,
    extract_giant_component,
)
from analysis.polarization_metrics.rwc_vectorized import (
    precompute_adjacency,
    compute_rwc_from_adj,
)
from analysis.polarization_metrics.point_biserial import (
    is_node_interpolated,
    compute_point_biserial_arrays,
)
from analysis.polarization_metrics.assortativity import (
    compute_assortativity_from_counts,
)
from analysis.polarization_metrics.adaptive_ei import (
    compute_aei_from_counts,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# CONFIGURATION
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

ALGORITHMS = ["kern", "leiden", "spectral"]

MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
INTERPOLATED = True

LEIDEN_RESOLUTION = 1.0
N_SIMULATIONS = 100

# Since Kernighan-Lin can be slow on large graphs over 1000 runs,
# you can optionally lower N_SIMULATIONS_KERN (e.g., to 100) if needed,
# or keep it at N_SIMULATIONS for full parity.
N_SIMULATIONS_BY_ALGO = {
    "kern": N_SIMULATIONS,
    "leiden": N_SIMULATIONS,
    "spectral": N_SIMULATIONS,
}

RWC_REPETITIONS_PER_PARTITION = 10
RWC_WALKS_PER_SIDE = 1000
RWC_SEED_FRACTION = 0.10

OUTPUT_DIR = (TOPICS_PATH / "_polarization_results").resolve()
OUTPUT_CSV = OUTPUT_DIR / f"polarization_simulations_failed_n{N_SIMULATIONS}.csv"


# ==========================================
# PREPARATION & CACHING
# ==========================================
def compute_cached_spectral_sides(G_int: nx.Graph) -> np.ndarray | None:
    """
    Computes the node-level spectral bipartition once per topic graph,
    as the normalized Laplacian and Fiedler vector are deterministic.
    """
    n_total = G_int.number_of_nodes()
    if n_total < 2:
        return None
    try:
        L = nx.normalized_laplacian_matrix(G_int, nodelist=range(n_total))
        # Seed fixed or unseeded; Fiedler eigenvector direction is oriented later
        _, eigenvectors = eigsh(L, k=2, which="SM")
        fiedler = eigenvectors[:, 1]
        node_sides = np.where(fiedler < 0, 0, 1).astype(int)
        return node_sides
    except Exception as e:
        print(f"[WARNING] Spectral decomposition failed: {e}")
        return None


def prepare_topic_graph(graph_path):
    """
    Loads the topic graph once, extracts the giant component,
    and pre-computes integer node indices (0..N-1), igraph object,
    NetworkX integer graph, and cached spectral partition.
    """
    raw_G = nx.read_graphml(graph_path)
    G = extract_giant_component(raw_G)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    G_int = nx.relabel_nodes(G, node_to_idx, copy=True)

    # Ensure undirected for Kernighan-Lin & Spectral
    G_int_undirected = G_int.to_undirected()

    edges_weighted = [
        (u, v, float(G_int_undirected[u][v].get("weight", 1.0)))
        for u, v in G_int_undirected.edges()
    ]
    g_ig = ig.Graph(
        n=len(nodes),
        edges=[(u, v) for u, v, _ in edges_weighted],
        directed=False,
    )

    adj_dict = precompute_adjacency(G_int_undirected)

    all_valid_idx, all_opinions = [], []
    seed_valid_idx, seed_opinions = [], []

    for i in range(len(nodes)):
        node_attr = G_int_undirected.nodes[i]
        score = node_attr.get("url_bias_score")
        if score is not None:
            val = float(score)
            all_valid_idx.append(i)
            all_opinions.append(val)
            if not is_node_interpolated(node_attr):
                seed_valid_idx.append(i)
                seed_opinions.append(val)

    spectral_sides = compute_cached_spectral_sides(G_int_undirected)

    return {
        "G_int": G_int_undirected,
        "g_ig": g_ig,
        "edges_weighted": edges_weighted,
        "adj_dict": adj_dict,
        "all_valid_idx": np.array(all_valid_idx, dtype=int),
        "all_opinions": np.array(all_opinions, dtype=float),
        "seed_valid_idx": np.array(seed_valid_idx, dtype=int),
        "seed_opinions": np.array(seed_opinions, dtype=float),
        "spectral_sides": spectral_sides,
    }


# ==========================================
# PARTITIONING DISPATCHER
# ==========================================
def get_partition_for_algorithm(
    prepped: dict, algorithm: str, seed: int
) -> tuple[np.ndarray | None, int]:
    """
    Returns (node_sides_array, n_communities) for the requested algorithm:
      - 'kern': Kernighan-Lin bisection (n_communities = 2)
      - 'leiden': Modularity Leiden, Largest Community vs. Rest (n_communities = m)
      - 'spectral': Node-level Normalized Laplacian Fiedler split (n_communities = 2)
    """
    n_total = prepped["g_ig"].vcount()

    if algorithm == "kern":
        try:
            side_0, _ = kernighan_lin_bisection(prepped["G_int"], seed=seed)
        except Exception:
            return None, 0
        node_sides = np.ones(n_total, dtype=int)
        node_sides[list(side_0)] = 0
        return node_sides, 2

    elif algorithm == "leiden":
        g_ig = prepped["g_ig"]
        result = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            seed=seed,
        )
        n_comms = len(result)
        if n_comms < 2:
            return None, n_comms

        # Sort communities by size descending; largest is Side 0, rest is Side 1
        communities = sorted(list(result), key=len, reverse=True)
        largest_comm = communities[0]

        node_sides = np.ones(n_total, dtype=int)
        node_sides[list(largest_comm)] = 0
        return node_sides, n_comms

    elif algorithm == "spectral":
        if prepped["spectral_sides"] is None:
            return None, 0
        return prepped["spectral_sides"].copy(), 2

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ==========================================
# SINGLE SIMULATION & METRICS STEP
# ==========================================
def simulate_single_partition_and_measure(
    prepped: dict, algorithm: str, resolution: float, seed: int
) -> dict | None:
    """
    Executes 1 partition using `algorithm` and evaluates all polarization metrics:
      1. Point-Biserial Correlation (r_pb & r_pb_seed_only)
      2. Welch's t-test (welch_t & opinion_diff)
      3. Random Walk Controversy (rwc)
      4. Assortativity Coefficient (ac)
      5. Adaptive EI Index (aei & aei_2x_cross)
    """
    n_total = prepped["g_ig"].vcount()

    node_sides, n_comms = get_partition_for_algorithm(prepped, algorithm, seed)
    if node_sides is None:
        return None

    # Canonically orient so Side 0 always has the lower mean opinion score
    all_idx = prepped["all_valid_idx"]
    all_ops = prepped["all_opinions"]
    if len(all_idx) > 0:
        s0_mask = node_sides[all_idx] == 0
        s1_mask = node_sides[all_idx] == 1
        if np.any(s0_mask) and np.any(s1_mask):
            if np.mean(all_ops[s0_mask]) > np.mean(all_ops[s1_mask]):
                node_sides = 1 - node_sides

    side_0_idx = np.where(node_sides == 0)[0].tolist()
    side_1_idx = np.where(node_sides == 1)[0].tolist()
    n_0 = len(side_0_idx)
    n_1 = len(side_1_idx)

    if n_0 == 0 or n_1 == 0:
        return None

    # 2. Count internal (e_00, e_11) and crossing (e_01) edges
    e_00 = 0.0
    e_11 = 0.0
    e_01 = 0.0
    for u, v, w in prepped["edges_weighted"]:
        su = node_sides[u]
        sv = node_sides[v]
        if su == 0 and sv == 0:
            e_00 += w
        elif su == 1 and sv == 1:
            e_11 += w
        else:
            e_01 += w

    # 3. Compute Structural Polarization Measures (AC, AEI, RWC)
    ac_score = compute_assortativity_from_counts(e_00, e_11, e_01)
    aei_score = compute_aei_from_counts(
        n_0, n_1, e_00, e_11, e_01, double_cross_density=False
    )
    aei_2x_score = compute_aei_from_counts(
        n_0, n_1, e_00, e_11, e_01, double_cross_density=True
    )

    rwc_stats = compute_rwc_from_adj(
        prepped["adj_dict"],
        side_0_idx,
        side_1_idx,
        seed_fraction=RWC_SEED_FRACTION,
        repetitions=RWC_REPETITIONS_PER_PARTITION,
        walks_per_side=RWC_WALKS_PER_SIDE,
        base_seed=seed,
    )
    rwc_score = rwc_stats["mean"]
    rwc_internal_std = rwc_stats["std"]

    # 4. Compute Opinion-Structural Polarization Measures (Point-Biserial & Welch's t-test)
    r_pb_all, p_val_all = compute_point_biserial_arrays(
        node_sides[all_idx], all_ops, canonical_orient=True
    )
    r_pb_seed, p_val_seed = compute_point_biserial_arrays(
        node_sides[prepped["seed_valid_idx"]],
        prepped["seed_opinions"],
        canonical_orient=True,
    )

    welch_t_all, welch_p_all, opinion_diff_all = compute_welch_arrays(
        node_sides[all_idx], all_ops, canonical_orient=True
    )
    welch_t_seed, welch_p_seed, opinion_diff_seed = compute_welch_arrays(
        node_sides[prepped["seed_valid_idx"]],
        prepped["seed_opinions"],
        canonical_orient=True,
    )

    return {
        "r_pb": r_pb_all,
        "p_value": p_val_all,
        "r_pb_seed_only": r_pb_seed,
        "p_value_seed_only": p_val_seed,
        "welch_t": welch_t_all,
        "welch_p": welch_p_all,
        "opinion_diff": opinion_diff_all,
        "welch_t_seed_only": welch_t_seed,
        "welch_p_seed_only": welch_p_seed,
        "opinion_diff_seed_only": opinion_diff_seed,
        "rwc": rwc_score,
        "rwc_mc_std": rwc_internal_std,
        "ac": ac_score,
        "aei": aei_score,
        "aei_2x_cross": aei_2x_score,
        "n_communities": n_comms,
        "n_side_0": n_0,
        "n_side_1": n_1,
        "minority_ratio": min(n_0, n_1) / n_total,
        "e_00": e_00,
        "e_11": e_11,
        "e_01_cut": e_01,
    }


def run_all_simulations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for group_name, topic_list in TOPIC_GROUPS.items():
        for topic in topic_list:
            graph_path = get_opinion_graph_path(
                topic, URL_SOURCE, MIN_RP, MIN_URL, INTERPOLATED
            )
            if not graph_path.exists():
                print(f"[WARNING] Skipping '{topic}': {graph_path} not found")
                continue

            print(f"\nPreparing '{topic}' [{group_name}]...")
            prepped = prepare_topic_graph(graph_path)

            for algo in ALGORITHMS:
                n_runs = N_SIMULATIONS_BY_ALGO[algo]
                desc = f"Simulating {topic} [{algo}]"

                for sim_idx in tqdm(range(n_runs), desc=desc):
                    seed = 1000 + sim_idx
                    metrics = simulate_single_partition_and_measure(
                        prepped,
                        algorithm=algo,
                        resolution=LEIDEN_RESOLUTION,
                        seed=seed,
                    )
                    if metrics is not None:
                        records.append(
                            {
                                "algorithm": algo,
                                "group": group_name,
                                "topic": topic,
                                "simulation": sim_idx,
                                "seed": seed,
                                "resolution": LEIDEN_RESOLUTION,
                                **metrics,
                            }
                        )

    df = pd.DataFrame(records)
    with open(OUTPUT_CSV.resolve(), "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)

    print("\n" + "=" * 65)
    print(
        f"Completed {len(df)} total simulation runs across "
        f"{df['topic'].nunique()} topics and {df['algorithm'].nunique()} algorithms."
    )
    print(f"Saved results CSV to:\n{OUTPUT_CSV.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_simulations()
