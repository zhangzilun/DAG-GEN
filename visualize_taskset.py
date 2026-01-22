# visualize_taskset_fixed.py


from pathlib import Path
import json
import random
import pickle
import gzip
from typing import Any, Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



ROOT_DIR = Path(r"E:\DAG-GNN\src1\exp_ratio_v2_timehead_lp300_tmax299999")   # your dag 
OUT_DIR  = Path(r"E:\DAG-GNN\src1\exp_ratio_v2_timehead_lp300_tmax299999_viz")
FORMAT   = "png"   

TIME_KEY = "label"
FALLBACK_KEYS = ["label_300", "critical_time", "time", "C", "weight", "t"]

MAX_GRAPHS = 0     
SEED = 42

SHOW_EDGE_LABELS = True
MAX_EDGE_LABELS = 120
NODE_SIZE = 700
FONT_SIZE = 8
EDGE_ALPHA = 0.85



def load_graph_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".gpickle", ".pkl", ".pickle", ".gz") or p.name.lower().endswith(".gpickle.gz"):
            files.append(p)
    return sorted(files)


def read_graph_any(p: Path) -> nx.DiGraph:
    op = gzip.open if str(p).lower().endswith(".gz") else open
    with op(p, "rb") as f:
        G = pickle.load(f)
    return G if isinstance(G, nx.DiGraph) else G.to_directed()


def edge_time(d: Dict[str, Any], default: float = 1.0) -> float:
    if TIME_KEY in d:
        try:
            return float(d[TIME_KEY])
        except:
            pass
    for k in FALLBACK_KEYS:
        if k in d:
            try:
                return float(d[k])
            except:
                pass
    return default


def compute_layers(G: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(G):
        nodes = list(G.nodes())
        return [nodes], {n: 0 for n in nodes}

    topo = list(nx.topological_sort(G))
    dist = {n: 0 for n in topo}
    for u in topo:
        for v in G.successors(u):
            dist[v] = max(dist[v], dist[u] + 1)

    layers = {}
    for n, d in dist.items():
        layers.setdefault(d, []).append(n)

    return [layers[k] for k in sorted(layers)], dist


def draw_one(G: nx.DiGraph, title: str, out_path: Path):
    layers, _ = compute_layers(G)

    pos = {}
    for li, nodes in enumerate(layers):
        for i, n in enumerate(nodes):
            pos[n] = (li * 2.0, -i)

    fig = plt.figure(figsize=(10, 6), dpi=160)
    ax = plt.gca()
    ax.axis("off")

    N, E, L = G.number_of_nodes(), G.number_of_edges(), len(layers)

    lp = 0.0
    for u, v in G.edges():
        lp += edge_time(G.edges[u, v])

    ax.set_title(f"{title}\nN={N} E={E} L={L} LP≈{lp:.1f}", fontsize=11)

    nx.draw_networkx_nodes(G, pos, node_size=NODE_SIZE, node_color="white",
                           edgecolors="black", linewidths=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=FONT_SIZE, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=EDGE_ALPHA, ax=ax)

    if SHOW_EDGE_LABELS and E <= MAX_EDGE_LABELS:
        labels = {(u, v): f"{edge_time(G.edges[u, v]):.0f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=FONT_SIZE - 1, ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = load_graph_files(ROOT_DIR)
    print(f"[INFO] graphs_found={len(files)}")

    if MAX_GRAPHS > 0 and len(files) > MAX_GRAPHS:
        random.Random(SEED).shuffle(files)
        files = files[:MAX_GRAPHS]

    index = []

    if FORMAT == "png":
        for p in files:
            G = read_graph_any(p)
            rel = p.relative_to(ROOT_DIR)
            out_png = OUT_DIR / rel.with_suffix(".png")
            draw_one(G, str(rel), out_png)
            index.append({"file": str(rel), "out": str(out_png)})

    else:
        pdf_path = OUT_DIR / "taskset_viz.pdf"
        with PdfPages(pdf_path) as pdf:
            for p in files:
                G = read_graph_any(p)
                rel = p.relative_to(ROOT_DIR)
                fig = plt.figure()
                draw_one(G, str(rel), None)
                pdf.savefig(fig)
                plt.close(fig)

    with open(OUT_DIR / "viz_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Visualization saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

