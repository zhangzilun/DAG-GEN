# run_support10_generate5_fixed.py
# 固定参数：用10个support图（可未见过） -> 生成5个图 -> 输出参数统计 -> 可视化+保存
# 运行：python run_support10_generate5_fixed.py

from __future__ import annotations

import os
import json
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

# 依赖你现有工程文件（确保同目录/可import）
import infer_fewshot_structure1_ratio_v2 as INF
import infer_fewshot_structure1_ratio_v2_timehead as TH


# =========================
# 1) 参数全部写死在这里
# =========================
DEVICE = INF.DEVICE if hasattr(INF, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu")

SUPPORT_DIR = Path(r"E:\DAG-GNN\src1\test10") # 这里放你的“10个样本DAG”（未见过也行）
CKPT_TIME   = Path(r"..\src1\checkpoints\best.pt")  # 你的 time-head 全家桶pt
OUT_DIR     = Path(r"..\src1\exp_unseen_support10_generate5")  # 输出目录

# support数量 & 生成数量
K_SHOT = 10
N_GEN  = 5

# 结构采样超参（ratio_v2）
TEMP = 0.20                 # 结构采样温度
PICK = "topk"               # "topk" 或 "thr"
THR  = 1.00                 # pick="thr"时有意义；pick="topk"也会用到一些阈值判定（按你ratio_v2实现）
TOPK_SCALE  = 1.00          # E_budget = E_ref * scale
TOPK_GUMBEL = 0.0
SVEC_MODE   = "no_E_T"      # 你的ratio_v2里常用：不把E/T作为硬约束（更稳）

ALLOW_SKIP = True           # True: 允许跳过极少数失败case（一般更稳）
MAX_TRIES_PER_GEN = 2000    # 每张图最多尝试次数（失败就换seed继续）

# 时间头参数
LP_BINS   = 8
LABEL_KEY = "label"         # 你 lp_report.py 里用的时间key（例如 label/critical_time）
INTEGERIZE = 0              # 1: 四舍五入到整数
TMAX_LP   = 300.0           # 最长路径上限
LP_MIN    = 240.0           # 抬底（你timehead脚本里默认）

BASE_SEED = 42


# =========================
# 2) 工具函数：读图 / 写图 / 统计 / 可视化
# =========================
TIME_KEYS_FALLBACK = ["label", "critical_time", "time", "weight", "t", "C"]

def read_graph_any(p: Path) -> nx.DiGraph:
    return INF.read_graph_any(p)

def list_graph_files(d: Path) -> List[Path]:
    exts = (".gpickle", ".gpickle.gz", ".gz", ".json")
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def edge_time(d: dict, default: float = 1.0) -> float:
    for k in TIME_KEYS_FALLBACK:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def graph_total_time(G: nx.DiGraph, key: str = LABEL_KEY) -> float:
    s = 0.0
    for _, _, d in G.edges(data=True):
        if key in d:
            try:
                s += float(d[key])
                continue
            except Exception:
                pass
        s += edge_time(d, 1.0)
    return float(s)

def longest_path_time(G: nx.DiGraph, key: str = LABEL_KEY) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    if not nx.is_directed_acyclic_graph(G):
        return float("nan")
    topo = list(nx.topological_sort(G))
    dp = {n: 0.0 for n in topo}
    for u in topo:
        for v in G.successors(u):
            t = float(edge_time(G.edges[u, v], 1.0)) if key not in G.edges[u, v] else float(G.edges[u, v][key])
            dp[v] = max(dp[v], dp[u] + t)
    return float(max(dp.values()) if dp else 0.0)

def layer_count_and_max_width(G: nx.DiGraph) -> Tuple[int, int]:
    layers = INF.topological_layers(G)
    widths = [len(x) for x in layers]
    return int(len(widths)), int(max(widths) if widths else 0)

def graph_stats(G: nx.DiGraph, name: str, key: str = LABEL_KEY) -> Dict[str, Any]:
    L, W = layer_count_and_max_width(G) if nx.is_directed_acyclic_graph(G) else (-1, -1)
    src = sum(1 for n in G.nodes if G.in_degree(n) == 0)
    sink = sum(1 for n in G.nodes if G.out_degree(n) == 0)
    iso = sum(1 for n in G.nodes if G.degree(n) == 0)
    lp = longest_path_time(G, key=key)
    tt = graph_total_time(G, key=key)
    return {
        "name": name,
        "dag": bool(nx.is_directed_acyclic_graph(G)),
        "N": int(G.number_of_nodes()),
        "E": int(G.number_of_edges()),
        "L": int(L),
        "W": int(W),
        "sources": int(src),
        "sinks": int(sink),
        "isolated": int(iso),
        "total_time": float(tt),
        "LP": float(lp) if np.isfinite(lp) else None,
        "time_key": key,
    }

def ensure_node_attrs_layer_pos(G: nx.DiGraph) -> None:
    # 若节点没有layer/pos，补上，方便层级绘图
    if any("layer" not in G.nodes[n] for n in G.nodes):
        topo = list(nx.topological_sort(G))
        dist = {n: 0 for n in topo}
        for u in topo:
            for v in G.successors(u):
                dist[v] = max(dist[v], dist[u] + 1)
        for n in G.nodes:
            G.nodes[n]["layer"] = int(dist[n])
    if any("pos" not in G.nodes[n] for n in G.nodes):
        layers: Dict[int, List[int]] = {}
        for n, d in G.nodes(data=True):
            layers.setdefault(int(d.get("layer", 0)), []).append(int(n))
        for li, nodes in layers.items():
            for i, n in enumerate(sorted(nodes)):
                G.nodes[n]["pos"] = int(i)

def layered_positions(G: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
    ensure_node_attrs_layer_pos(G)
    return {int(n): (float(G.nodes[n]["layer"]), -float(G.nodes[n]["pos"])) for n in G.nodes}

def save_dag_plot(G: nx.DiGraph, png_path: Path, title: str, key: str = LABEL_KEY) -> None:
    plt.figure(figsize=(9, 4.5))
    pos = layered_positions(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=450, font_size=7)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if key in d:
            edge_labels[(u, v)] = f"{float(d[key]):.1f}"
        elif "critical_time" in d:
            edge_labels[(u, v)] = f"{float(d['critical_time']):.1f}"
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(title)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=180)
    plt.close()

def save_adjacency_grid_png(A: np.ndarray, png_path: Path, title: str) -> None:
    """
    点阵可视化：每个格子都有边框方格（满足你“每个格子渲染成带边框方格”的要求）
    A: NxN 0/1
    """
    N = int(A.shape[0])
    plt.figure(figsize=(6, 6))
    # 用pcolormesh画格子，并且每个格子有边框
    x = np.arange(N + 1)
    y = np.arange(N + 1)
    plt.pcolormesh(x, y, A.astype(np.float32), shading="flat", edgecolors="k", linewidth=0.25)
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=220)
    plt.close()

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@torch.no_grad()
def generate_one_strict_ratio_v2(
    INF,
    encoder,
    decoder,
    sup_graphs,      # list[nx.DiGraph]  (10-shot)
    ref_graph,       # nx.DiGraph
    temp: float,
    thr: float,
    pick: str,
    topk_scale: float,
    topk_gumbel: float,
    svec_mode: str,
    allow_skip: bool,
    seed: int,
):
    """
    返回：
      {"ok":True, "G":nx.DiGraph, "A_np":np.ndarray, "widths":list[int], "z_style":torch.Tensor}
    或：
      {"ok":False, "note": "..."}
    """

    rng = random.Random(int(seed))

    # ---- 1) support -> style z ----
    X_list = []
    for Gs in sup_graphs:
        X, _order, _widths = INF.graph_to_node_feats(Gs)
        X_list.append(X)
    z = INF.encode_style(encoder, X_list)          # [d] or [d_style]
    z = z.to(INF.DEVICE).view(1, -1)              # [1,d]

    # ---- 2) ref -> widths + layer_ids + s_vec ----
    widths = INF.graph_to_widths(ref_graph)
    if len(widths) <= 1:
        return {"ok": False, "note": "ref_graph has L<=1"}

    layer_ids = []
    for li, w in enumerate(widths):
        layer_ids += [li] * int(w)

    s_vec = INF.graph_to_svec(ref_graph)
    s_vec = INF.mask_svec(s_vec, mode=str(svec_mode)).to(INF.DEVICE)

    # ---- 3) decoder -> logits ----
    # 你的 ratio_v2 里 decoder 是这样被调用的：decoder(s_vec, widths=..., z_style=z)
    out = decoder(s_vec, widths=widths, z_style=z)
    # out 可能是 (A_logits, widths_used) 或 dict，这里做鲁棒解析
    out = decoder(s_vec, widths=widths, z_style=z)

    # ===== 修复点：decoder 直接返回 A_logits =====
    if isinstance(out, torch.Tensor):
        A_logits = out
        widths_used = widths
    elif isinstance(out, (tuple, list)):
        A_logits = out[0]
        widths_used = out[1] if len(out) > 1 else widths
    elif isinstance(out, dict):
        A_logits = out["A_logits"]
        widths_used = out.get("widths", widths)
    else:
        return {"ok": False, "note": f"unknown decoder output type: {type(out)}"}

    prob, valid_mask = INF.logits_to_prob_and_mask(A_logits, temperature=float(temp))

    # ---- 4) pick adjacency ----
    N = int(prob.size(0))
    E_ref = int(ref_graph.number_of_edges())
    if pick == "thr":
        A = ((prob > float(thr)) & valid_mask).to(torch.int32).detach().cpu().numpy()
    elif pick == "topk":
        k_total = int(max(0, round(E_ref * float(topk_scale))))
        A = INF.coverage_topk_adj_from_prob(
            prob=prob,
            valid_mask=valid_mask,
            layer_ids=layer_ids[:N],
            k_total=k_total,
            allow_skip=bool(allow_skip),
            add_gumbel=float(topk_gumbel),
            seed=int(seed),
            target_skip_frac=0.0,
        )
    else:
        return {"ok": False, "note": f"Unknown pick={pick}"}

    # ---- 5) strict validity check ----
    G = INF.build_graph_from_adj(A)
    rs = INF.strict_reasons(G)
    if not INF.is_valid_strict(rs):
        return {"ok": False, "note": f"invalid: {rs}"}

    return {"ok": True, "G": G, "A_np": A, "widths": widths_used, "z_style": z.squeeze(0)}


# =========================
# 3) 核心流程：用support生成5张图
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3.1 读取 support 图（必须>=10）
    files = list_graph_files(SUPPORT_DIR)
    if len(files) < K_SHOT:
        raise RuntimeError(f"SUPPORT_DIR={SUPPORT_DIR} 文件不足：found={len(files)} need={K_SHOT}")

    support_files = files[:K_SHOT]
    support_graphs = [read_graph_any(p) for p in support_files]

    # 3.2 加载 time-head checkpoint（包含 encoder+decoder+time_head）
    encoder, decoder, time_head, meta = TH.load_time_head_ckpt(
        ckpt_time=CKPT_TIME,
        lp_bins=LP_BINS,
        encoder=None,
        decoder=None,
    )

    # 3.3 support 统计
    support_stats = []
    for p, G in zip(support_files, support_graphs):
        support_stats.append(graph_stats(G, name=p.name, key=LABEL_KEY))

    # 保存 support 报告
    save_json(OUT_DIR / "support_stats.json", support_stats)
    save_csv(OUT_DIR / "support_stats.csv", support_stats)

    # 3.4 生成 N_GEN 张图
    gen_stats: List[Dict[str, Any]] = []
    rng = random.Random(BASE_SEED)

    for gi in range(N_GEN):
        ok = False
        last_note = ""
        # 每次生成选一个ref_graph（从support里轮流/随机选都行）
        ref_graph = support_graphs[gi % len(support_graphs)]

        # 固定support集：用于style编码（10-shot）
        sup_graphs = support_graphs

        for attempt in range(MAX_TRIES_PER_GEN):
            seed = BASE_SEED + 10000 * gi + attempt

            out = generate_one_strict_ratio_v2(
                INF=INF,
                encoder=encoder,
                decoder=decoder,
                sup_graphs=sup_graphs,
                ref_graph=ref_graph,
                temp=float(TEMP),
                thr=float(THR),
                pick=str(PICK),
                topk_scale=float(TOPK_SCALE),
                topk_gumbel=float(TOPK_GUMBEL),
                svec_mode=str(SVEC_MODE),
                allow_skip=bool(ALLOW_SKIP),
                seed=int(seed),
            )

            if not out.get("ok", False):
                last_note = out.get("note", "generate_one_strict failed")
                continue

            Ggen: nx.DiGraph = out["G"]
            A_np: np.ndarray = out["A_np"]
            widths: List[int] = out["widths"]
            z_style: torch.Tensor = out["z_style"]  # [1,d]

            # attach time labels + LP校准（<=TMAX）
            TH.attach_time_labels(
                G=Ggen,
                A=A_np,
                widths=widths,
                z_style=z_style,
                time_head=time_head,
                lp_bins=int(LP_BINS),
                label_key=str(LABEL_KEY),
                integerize=int(INTEGERIZE),
                tmax_longest_path=float(TMAX_LP),
                lp_min=float(LP_MIN),
            )

            # 最终检查：DAG + 单源单汇 + LP<=TMAX
            if not nx.is_directed_acyclic_graph(Ggen):
                last_note = "not DAG after generation"
                continue
            src = sum(1 for n in Ggen.nodes if Ggen.in_degree(n) == 0)
            sink = sum(1 for n in Ggen.nodes if Ggen.out_degree(n) == 0)
            if src != 1 or sink != 1:
                last_note = f"src/sink != 1 (src={src}, sink={sink})"
                continue
            lp = longest_path_time(Ggen, key=LABEL_KEY)
            if not np.isfinite(lp) or lp > float(TMAX_LP) + 1e-6:
                last_note = f"LP too large: {lp}"
                continue

            # 保存图
            base = f"gen_{gi:02d}"
            out_sub = OUT_DIR / base
            TH.save_graph_gpickle_json(Ggen, out_sub, base)

            # 可视化：层级DAG图 + 点阵邻接矩阵
            save_dag_plot(Ggen, out_sub / f"{base}_dag.png", title=f"{base} | LP={lp:.1f}", key=LABEL_KEY)
            save_adjacency_grid_png((A_np > 0.5).astype(np.int32), out_sub / f"{base}_adj_grid.png",
                                    title=f"{base} adjacency (grid)")

            # 统计输出（包含support条件信息）
            st = graph_stats(Ggen, name=f"{base}.gpickle", key=LABEL_KEY)
            st["ref_used"] = f"support[{gi % len(support_graphs)}]={support_files[gi % len(support_graphs)].name}"
            st["seed"] = int(seed)
            st["attempt"] = int(attempt)
            gen_stats.append(st)

            ok = True
            print(f"[OK] generated {base}: N={st['N']} E={st['E']} L={st['L']} W={st['W']} total={st['total_time']:.1f} LP={st['LP']:.1f}")
            break

        if not ok:
            print(f"[FAIL] gen_{gi:02d} failed after {MAX_TRIES_PER_GEN} tries. last_note={last_note}")
            gen_stats.append({
                "name": f"gen_{gi:02d}",
                "ok": False,
                "note": last_note,
            })

    # 3.5 保存最终报告
    report = {
        "meta": {
            "support_dir": str(SUPPORT_DIR),
            "ckpt_time": str(CKPT_TIME),
            "out_dir": str(OUT_DIR),
            "k_shot": int(K_SHOT),
            "n_gen": int(N_GEN),
            "TEMP": float(TEMP),
            "PICK": str(PICK),
            "THR": float(THR),
            "TOPK_SCALE": float(TOPK_SCALE),
            "SVEC_MODE": str(SVEC_MODE),
            "LP_BINS": int(LP_BINS),
            "LABEL_KEY": str(LABEL_KEY),
            "INTEGERIZE": int(INTEGERIZE),
            "TMAX_LP": float(TMAX_LP),
            "LP_MIN": float(LP_MIN),
            "BASE_SEED": int(BASE_SEED),
            "time_ckpt_meta": meta,
        },
        "support": support_stats,
        "generated": gen_stats,
    }
    save_json(OUT_DIR / "report.json", report)
    save_csv(OUT_DIR / "generated_stats.csv", [r for r in gen_stats if isinstance(r, dict) and r.get("name")])

    print(f"\n[DONE] Outputs written to: {OUT_DIR.resolve()}")
    print(f" - support_stats.json / support_stats.csv")
    print(f" - generated_stats.csv / report.json")
    print(f" - each gen_i saved under: {OUT_DIR}/gen_XX/\n")


if __name__ == "__main__":
    main()
