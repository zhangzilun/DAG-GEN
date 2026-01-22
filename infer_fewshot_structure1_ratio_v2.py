# infer_fewshot_structure1_ratio_v2.py  -- FULL REPLACE VERSION (per-bucket temp×thr grid search)
# Goal: Calibrate (temp,thr) per bucket to minimize density error under strict-valid constraint,
#       WITHOUT any post-fix (no add/remove edges). We only choose temp and threshold.
#
# Usage example:
# python infer_fewshot_structure1.py ^
#   --ckpt checkpoints_fewshot_structure_freeze_v3/fewshot_best.pt ^
#   --k_shot 10 ^
#   --tries_per_bucket 50 --max_tries_per_bucket 3000 ^
#   --calib_n 120 ^
#   --temps 0.5 0.7 0.9 1 1.3 1.6 2 3 4 5 6 8 ^
#   --thr_list 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.92 0.94 0.95 0.96 0.97 0.98 ^
#   --min_valid_rate 0.25 ^
#   --svec_mode no_E_T ^
#   --save_valid 1 --out_dir infer_fewshot_v3_calib_grid_out
#
from __future__ import annotations

import argparse
import json
import math
import random
import pickle
import gzip
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from config import DATA_DIR, DEVICE
from config_structure1 import BUCKETS
from models import FewShotStyleEncoder, StructureToGraphDecoder5
from utils import topological_layers


def load_graph_files(data_dir: Path) -> List[Path]:
    data_dir = Path(data_dir)
    if data_dir.is_file():
        return [data_dir]
    files: List[Path] = []
    for p in data_dir.rglob("*"):
        suf = p.suffix.lower()
        if suf in (".json", ".gpickle", ".gz") or p.name.endswith(".gpickle.gz"):
            files.append(p)
    files.sort()
    return files


def read_graph_any(p: Path) -> nx.DiGraph:
    is_gz = str(p).endswith(".gpickle.gz") or p.suffix.lower() == ".gz"
    opener = gzip.open if is_gz else open
    with opener(p, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()
    return G


# ---------------- bucketing ----------------
def bucket_id_for_L(L: int) -> int:
    for bi, (_name, lo, hi) in enumerate(BUCKETS):
        if lo <= L <= hi:
            return bi
    return len(BUCKETS) - 1


def bucket_name(bi: int) -> str:
    return BUCKETS[bi][0] if 0 <= bi < len(BUCKETS) else f"B{bi}"


_num_re = re.compile(r"(\d+)$")


def node_sort_key(n: Any) -> Tuple[int, int, str]:
    if isinstance(n, int):
        return (0, int(n), str(n))
    s = str(n)
    m = _num_re.search(s)
    if m:
        return (1, int(m.group(1)), s)
    return (2, 0, s)


def graph_to_layers_and_order(G: nx.DiGraph) -> Tuple[List[List[Any]], List[Any]]:
    layers = topological_layers(G)
    layers_sorted = [sorted(Li, key=node_sort_key) for Li in layers]
    node_order: List[Any] = []
    for Li in layers_sorted:
        node_order.extend(Li)
    return layers_sorted, node_order


def graph_to_widths(G: nx.DiGraph) -> List[int]:
    layers, _ = graph_to_layers_and_order(G)
    return [len(Li) for Li in layers]


def graph_to_adj_target(G: nx.DiGraph, node_order: List[Any]) -> torch.Tensor:
    N = len(node_order)
    idx = {node_order[i]: i for i in range(N)}
    A = torch.zeros((N, N), dtype=torch.float32)
    for u, v in G.edges():
        if u in idx and v in idx:
            iu, iv = idx[u], idx[v]
            if iu != iv:
                A[iu, iv] = 1.0
    return A


def graph_to_node_feats(G: nx.DiGraph) -> Tuple[torch.Tensor, List[Any], List[int]]:
    """
    Encoder input aligned with node_order:
      [layer_id_norm, in_deg_norm, out_deg_norm]  => [N,3]
    """
    layers, node_order = graph_to_layers_and_order(G)
    widths = [len(Li) for Li in layers]
    N = len(node_order)
    L = max(1, len(layers))

    layer_id = torch.zeros((N,), dtype=torch.float32)
    node_to_layer = {}
    for li, Li in enumerate(layers):
        for n in Li:
            node_to_layer[n] = li
    for i, n in enumerate(node_order):
        li = node_to_layer.get(n, 0)
        layer_id[i] = float(li) / float(max(1, L - 1))

    indeg = torch.tensor([float(G.in_degree(n)) for n in node_order], dtype=torch.float32)
    outdeg = torch.tensor([float(G.out_degree(n)) for n in node_order], dtype=torch.float32)
    indeg = indeg / indeg.max().clamp_min(1.0)
    outdeg = outdeg / outdeg.max().clamp_min(1.0)

    X = torch.stack([layer_id, indeg, outdeg], dim=-1)
    return X, node_order, widths


def pad_stack_graph_feats(X_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_list: list of [Ni, F]
    returns:
      X: [K, Nmax, F]
      mask: [K, Nmax] (float 0/1)
    """
    K = len(X_list)
    Fdim = int(X_list[0].shape[-1])
    Nmax = max(int(x.shape[0]) for x in X_list)
    X = torch.zeros((K, Nmax, Fdim), dtype=torch.float32)
    mask = torch.zeros((K, Nmax), dtype=torch.float32)
    for i, x in enumerate(X_list):
        n = int(x.shape[0])
        X[i, :n, :] = x
        mask[i, :n] = 1.0
    return X, mask


def graph_to_svec(G: nx.DiGraph) -> torch.Tensor:
    # normalization from your project config
    from config import NORM_N, NORM_E, NORM_L, NORM_W, NORM_T

    widths = graph_to_widths(G)
    N = int(G.number_of_nodes())
    E = int(G.number_of_edges())
    L = int(len(widths))
    W = int(max(widths)) if widths else 0
    total_T = 0.0  # structure-only
    return torch.tensor(
        [N / NORM_N, E / NORM_E, L / NORM_L, W / NORM_W, total_T / max(1e-9, NORM_T)],
        dtype=torch.float32,
    )


def mask_svec(s_vec: torch.Tensor, mode: str = "no_E_T") -> torch.Tensor:
    """
    s_vec = [N,E,L,W,T] (normalized)
    """
    x = s_vec.clone()
    if x.dim() == 2 and x.size(0) == 1:
        x = x.squeeze(0)
    if mode == "no_E_T":
        if x.numel() >= 2:
            x[1] = 0.0
        if x.numel() >= 5:
            x[4] = 0.0
    elif mode == "no_E":
        if x.numel() >= 2:
            x[1] = 0.0
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown svec_mode: {mode}")
    return x


def build_graph_from_adj(A: np.ndarray) -> nx.DiGraph:
    N = int(A.shape[0])
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    ii, jj = np.where(A > 0)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i != j:
            G.add_edge(int(i), int(j))
    return G


def strict_reasons(G: nx.DiGraph) -> Dict[str, Any]:
    reasons: Dict[str, Any] = {}
    reasons["N"] = int(G.number_of_nodes())
    reasons["E"] = int(G.number_of_edges())
    reasons["has_cycle"] = (not nx.is_directed_acyclic_graph(G))
    src = [n for n in G.nodes() if G.in_degree(n) == 0]
    sink = [n for n in G.nodes() if G.out_degree(n) == 0]
    iso = [n for n in G.nodes() if G.degree(n) == 0]
    reasons["num_sources"] = int(len(src))
    reasons["num_sinks"] = int(len(sink))
    reasons["num_isolated"] = int(len(iso))
    reasons["multi_source"] = (len(src) != 1)
    reasons["multi_sink"] = (len(sink) != 1)
    reasons["has_isolated"] = (len(iso) > 0)
    return reasons


def is_valid_strict(reasons: Dict[str, Any]) -> bool:
    return (not reasons["has_cycle"]) and (not reasons["multi_source"]) and (not reasons["multi_sink"]) and (not reasons["has_isolated"])


def logits_to_prob_and_mask(A_logits: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert logits -> prob only on finite positions, using stable clamp.
    valid_mask defines allowed positions (mask-space).
    """
    if A_logits.dim() == 3 and A_logits.size(0) == 1:
        A_logits = A_logits.squeeze(0)

    valid_mask = torch.isfinite(A_logits)

    x = torch.nan_to_num(A_logits, nan=0.0, posinf=10.0, neginf=-10.0)

    # Keep consistent with your training/infer scaling (project has LOGIT_TEMPERATURE often).
    # If not exists, default 6.0.
    try:
        import config_structure1 as cfg
        logit_temp = float(getattr(cfg, "LOGIT_TEMPERATURE", 6.0))
    except Exception:
        logit_temp = 6.0

    x = x / max(1e-6, logit_temp)
    x = x.clamp(-8.0, 8.0)
    x = x / max(1e-9, float(temperature))

    prob = torch.zeros_like(x)
    if valid_mask.any():
        prob[valid_mask] = torch.sigmoid(x[valid_mask])
    return prob, valid_mask

def _hist_int(values, bins):
    # bins: list of (lo, hi) inclusive, last can be (k, 10**9)
    h = [0] * len(bins)
    for v in values:
        for i, (lo, hi) in enumerate(bins):
            if lo <= v <= hi:
                h[i] += 1
                break
    s = sum(h)
    if s <= 0:
        return [0.0] * len(h)
    return [x / s for x in h]

def graph_dist_stats_from_adj(A_bin: torch.Tensor, layer_ids: list[int]) -> dict:
    """
    A_bin: [N,N] 0/1 float tensor
    layer_ids: len N, int layer index for each node
    returns: distribution-like stats (histograms + key scalars)
    """
    N = int(A_bin.size(0))
    # degrees
    in_deg = A_bin.sum(dim=0).to(torch.int64).tolist()
    out_deg = A_bin.sum(dim=1).to(torch.int64).tolist()

    # degree hist bins: 0,1,2,3,4,5+
    deg_bins = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,10**9)]
    in_hist = _hist_int(in_deg, deg_bins)
    out_hist = _hist_int(out_deg, deg_bins)

    # edge span hist: span=1,2,3,4,5+
    ii, jj = torch.nonzero(A_bin > 0.5, as_tuple=True)
    spans = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        spans.append(int(layer_ids[j] - layer_ids[i]))
    span_bins = [(1,1),(2,2),(3,3),(4,4),(5,10**9)]
    span_hist = _hist_int(spans, span_bins)

    # widths dist
    L = int(max(layer_ids) + 1) if layer_ids else 0
    widths = [0] * max(1, L)
    for li in layer_ids:
        widths[int(li)] += 1
    # width bins: 1-2,3-4,5-6,7-8,9+
    width_bins = [(1,2),(3,4),(5,6),(7,8),(9,10**9)]
    width_hist = _hist_int(widths, width_bins)

    # layer-pair ratio vector (already in your code)
    v_lp, neigh_ratio = layer_pair_ratios_from_adj(A_bin, layer_ids=layer_ids, L=L)

    return {
        "N": N,
        "E": int(A_bin.sum().item()),
        "L": int(L),
        "widths": widths,
        "width_hist": width_hist,
        "in_deg_hist": in_hist,
        "out_deg_hist": out_hist,
        "span_hist": span_hist,
        "neighbor_ratio": float(neigh_ratio),
        "layerpair_vec": v_lp.detach().cpu().tolist(),  # length L*L
    }


def threshold_adj(prob: torch.Tensor, valid_mask: torch.Tensor, thr: float) -> np.ndarray:
    p = prob.detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    N = p.shape[0]
    A = np.zeros((N, N), dtype=np.int32)
    A[m] = (p[m] > float(thr)).astype(np.int32)
    return A


def dens_in_mask_space_from_adj(A: np.ndarray, allowed: float) -> float:
    return float(A.sum()) / max(1.0, float(allowed))


def dens_in_mask_space_from_target(A_tgt: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[float, float]:
    """
    returns dens_gt, allowed
    """
    if A_tgt.dim() == 3 and A_tgt.size(0) == 1:
        A_tgt = A_tgt.squeeze(0)
    allowed = float(valid_mask.sum().item())
    E = float(A_tgt[valid_mask].sum().item()) if valid_mask.any() else 0.0
    dens = E / max(1.0, allowed)
    return dens, allowed


def index_files_by_bucket(files: List[Path]) -> Dict[int, List[int]]:
    bucket_to_idx: Dict[int, List[int]] = {i: [] for i in range(len(BUCKETS))}
    for i, p in enumerate(files):
        try:
            G = read_graph_any(p)
            if not nx.is_directed_acyclic_graph(G):
                continue
            L = len(graph_to_widths(G))
            if L <= 1:
                continue
            bi = bucket_id_for_L(L)
            bucket_to_idx[bi].append(i)
        except Exception:
            continue
    return bucket_to_idx


def print_bucket_counts(bucket_to_idx: Dict[int, List[int]]) -> None:
    print("========== Reference bucket counts (by L) ==========")
    for bi, (name, lo, hi) in enumerate(BUCKETS):
        c = len(bucket_to_idx.get(bi, []))
        print(f"[{bi}] {name:>12}  L∈[{lo},{hi}] : {c}")
    print("====================================================\n")


def sample_k_support(
    bucket_to_idx: Dict[int, List[int]],
    files: List[Path],
    bi: int,
    k: int,
    rng: random.Random,
    exclude: Optional[int] = None,
) -> List[torch.Tensor]:
    pool = bucket_to_idx.get(bi, [])
    if len(pool) == 0:
        raise RuntimeError(f"Empty bucket: {bucket_name(bi)}")
    cand = pool[:]
    if exclude is not None and exclude in cand and len(cand) > 1:
        cand.remove(exclude)
    if len(cand) >= k:
        picks = rng.sample(cand, k=k)
    else:
        picks = [rng.choice(cand) for _ in range(k)]

    X_list: List[torch.Tensor] = []
    for idx in picks:
        G = read_graph_any(files[idx])
        X, _order, _widths = graph_to_node_feats(G)
        X_list.append(X)
    return X_list

def encode_style(encoder: torch.nn.Module, X_list: List[torch.Tensor]) -> torch.Tensor:
    X, M = pad_stack_graph_feats(X_list)
    X = X.to(DEVICE)
    M = M.to(DEVICE)
    z = encoder(X, M)
    # robust aggregation to [d]
    if isinstance(z, torch.Tensor) and z.dim() >= 2:
        z = z.mean(dim=tuple(range(z.dim() - 1))).view(-1)
    else:
        z = z.view(-1)
    return z

#Top-K
def topk_adj_from_prob(prob: torch.Tensor, valid_mask: torch.Tensor, k: int, add_gumbel: float = 0.0, seed: int = 0) -> np.ndarray:
    """
    prob: [N,N] in [0,1]
    valid_mask: [N,N] bool (allowed edges)
    k: number of edges to select in mask-space
    add_gumbel: >0 adds gumbel noise for diversity (0 means deterministic top-k)
    """
    if prob.dim() == 3 and prob.size(0) == 1:
        prob = prob.squeeze(0)
    if valid_mask.dim() == 3 and valid_mask.size(0) == 1:
        valid_mask = valid_mask.squeeze(0)

    N = prob.size(0)
    A = np.zeros((N, N), dtype=np.int32)

    # collect candidate indices
    idx = torch.nonzero(valid_mask, as_tuple=False)  # [M,2]
    M = idx.size(0)
    if M == 0:
        return A

    # clamp k
    k = int(max(0, min(int(k), int(M))))
    if k == 0:
        return A

    scores = prob[valid_mask]  # [M]
    scores = torch.clamp(scores, 1e-6, 1 - 1e-6)

    if add_gumbel and add_gumbel > 0:
        g = torch.Generator(device=scores.device)
        g.manual_seed(int(seed))
        u = torch.rand_like(scores, generator=g)
        gumbel = -torch.log(-torch.log(torch.clamp(u, 1e-6, 1 - 1e-6)))
        scores = scores + float(add_gumbel) * gumbel

 
    topk = torch.topk(scores, k=k, largest=True)
    chosen = idx[topk.indices]  # [k,2]
    ii = chosen[:, 0].detach().cpu().numpy()
    jj = chosen[:, 1].detach().cpu().numpy()
    A[ii, jj] = 1
    return A


def k_from_ref_dens(mean_ref_dens: float, allowed_edges: float) -> int:
    # allowed_edges = valid_mask.sum()
    return int(round(float(mean_ref_dens) * float(allowed_edges)))

def load_fewshot_ckpt(ckpt_path: Path) -> Tuple[FewShotStyleEncoder, StructureToGraphDecoder5, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported ckpt format: expected dict")

    d_style = int(ckpt.get("d_style", 32))
    k_shot = int(ckpt.get("k_shot", 10))

    encoder = FewShotStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style).to(DEVICE)
    decoder = StructureToGraphDecoder5(d_style=d_style).to(DEVICE)

    # load state dicts
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"], strict=True)
    else:
        raise RuntimeError("ckpt missing key: encoder")

    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"], strict=True)
    elif "model" in ckpt:
        # legacy
        decoder.load_state_dict(ckpt["model"], strict=True)
    else:
        raise RuntimeError("ckpt missing key: decoder/model")

    encoder.eval()
    decoder.eval()

    meta = {
        "d_style": d_style,
        "ckpt_k_shot": k_shot,
        "epoch": int(ckpt.get("epoch", -1)),
        "best_loss": float(ckpt.get("best_loss", float("nan"))),
    }
    return encoder, decoder, meta

def srcsink_profile(A_bin: torch.Tensor) -> Dict[str, int]:
    """
    Count number of source and sink nodes from a binary adjacency matrix.
    A_bin: (N, N) torch.Tensor with 0/1 entries
    """
    # out-degree = row sum, in-degree = col sum
    out_deg = A_bin.sum(dim=1)
    in_deg = A_bin.sum(dim=0)

    sources = int((in_deg == 0).sum().item())
    sinks = int((out_deg == 0).sum().item())

    return {
        "sources": sources,
        "sinks": sinks,
    }

def layer_pair_ratios_from_adj(
    A_bin: torch.Tensor,
    layer_ids: List[int],
    L: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Compute layer-pair edge ratio vector and neighbor-layer ratio.

    Returns:
        v : Tensor of shape [L*L], normalized layer-pair ratios
        neighbor_ratio : fraction of edges that go to adjacent layers
    """
    N = A_bin.size(0)
    if L is None:
        L = int(max(layer_ids)) + 1

    mat = torch.zeros((L, L), device=A_bin.device)
    total_edges = float(A_bin.sum().item())

    if total_edges <= 0:
        return torch.zeros(L * L, device=A_bin.device), 0.0

    for i in range(N):
        li = layer_ids[i]
        row = A_bin[i]
        if row.sum() == 0:
            continue
        js = torch.nonzero(row, as_tuple=False).view(-1)
        for j in js.tolist():
            lj = layer_ids[j]
            mat[li, lj] += 1.0

    v = mat.flatten() / total_edges

    # neighbor ratio: edges between adjacent layers only
    neigh_edges = 0.0
    for li in range(L - 1):
        neigh_edges += mat[li, li + 1].item()

    neighbor_ratio = neigh_edges / total_edges

    return v, float(neighbor_ratio)

def l1_ratio_distance(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    L1 distance between two ratio vectors.
    """
    if v1 is None or v2 is None:
        return float("inf")
    if v1.numel() != v2.numel():
        return float("inf")
    return float(torch.sum(torch.abs(v1 - v2)).item())

def pick_and_store_calib_choice(rep: Dict[str, Any], bi: int, pick: str) -> Tuple[float, float, str]:
    """
    Return (best_temp, best_value, tag_str)
      - pick=="thr"  -> best_value = best_thr
      - pick=="topk" -> best_value = best_topk_scale
    """
    if rep.get("best_temp", None) is None:
        raise RuntimeError(f"Empty calibration for bucket={bucket_name(bi)}")

    t = float(rep["best_temp"])

    if pick == "thr":
        if rep.get("best_thr", None) is None:
            raise RuntimeError(f"Missing best_thr for bucket={bucket_name(bi)}")
        val = float(rep["best_thr"])
        tag = f"thr={val:.3f}"
    elif pick == "topk":
        if rep.get("best_topk_scale", None) is None:
            raise RuntimeError(f"Missing best_topk_scale for bucket={bucket_name(bi)}")
        val = float(rep["best_topk_scale"])
        tag = f"topk_scale={val:.3f}"
    else:
        raise ValueError(f"Unknown pick: {pick}")

    return t, val, tag

def coverage_topk_adj_from_prob(
    prob: torch.Tensor,
    valid_mask: torch.Tensor,
    layer_ids: List[int],
    k_total: int,
    allow_skip: bool = True,
    add_gumbel: float = 0.0,
    seed: int = 0,
    target_skip_frac: float = 0.0,   # <<< NEW: fraction of non-neighbor edges (span>=2) in fill stage
) -> np.ndarray:
    """
    One-shot edge picking rule (NOT post-fix):
      (1) coverage-in  : each node j with layer>0 gets >=1 incoming edge
      (2) coverage-out : each node i with layer<L-1 gets >=1 outgoing edge
      (3) fill remaining budget with TopK, BUT split into:
            - skip edges (span>=2) quota
            - neighbor edges (span==1) quota
    """
    if prob.dim() == 3 and prob.size(0) == 1:
        prob = prob.squeeze(0)
    if valid_mask.dim() == 3 and valid_mask.size(0) == 1:
        valid_mask = valid_mask.squeeze(0)

    N = int(prob.size(0))
    A = torch.zeros((N, N), dtype=torch.int32, device=prob.device)

    idx = torch.nonzero(valid_mask, as_tuple=False)  # [M,2]
    if seed % 100000 == 0:
        layer_ids_t = torch.tensor(layer_ids, device=prob.device, dtype=torch.long)
        src = idx[:, 0];
        dst = idx[:, 1]
        span = layer_ids_t[dst] - layer_ids_t[src]
        print(
            f"[CAND] M={idx.size(0)} neigh={(span == 1).sum().item()} skip={(span >= 2).sum().item()} allow_skip={allow_skip}")

    M = int(idx.size(0))
    if M == 0:
        return A.detach().cpu().numpy()

    scores_all = prob[valid_mask].clamp(1e-6, 1 - 1e-6)

    # optional gumbel noise
    if add_gumbel and float(add_gumbel) > 0.0:
        g = torch.Generator(device=scores_all.device)
        g.manual_seed(int(seed))
        u = torch.rand_like(scores_all, generator=g)
        gumbel = -torch.log(-torch.log(u.clamp(1e-6, 1 - 1e-6)))
        scores_all = scores_all + float(add_gumbel) * gumbel

    src = idx[:, 0]
    dst = idx[:, 1]

    layer_ids_t = torch.tensor(layer_ids, device=prob.device, dtype=torch.long)
    srcL = layer_ids_t[src]
    dstL = layer_ids_t[dst]
    L = int(layer_ids_t.max().item()) + 1

    if allow_skip:
        cov_mask = (srcL < dstL)
    else:
        cov_mask = (dstL == (srcL + 1))

    chosen_set = set()

    def _choose(ii: int, jj: int):
        A[ii, jj] = 1
        chosen_set.add((ii, jj))

    for j in range(N):
        lj = int(layer_ids[j])
        if lj <= 0:
            continue
        m = cov_mask & (dst == j)
        if not bool(m.any()):
            continue
        sc = scores_all[m]
        ii = src[m]
        kbest = int(torch.argmax(sc).item())
        i_best = int(ii[kbest].item())
        if (i_best, j) not in chosen_set:
            _choose(i_best, j)


    for i in range(N):
        li = int(layer_ids[i])
        if li >= L - 1:
            continue
        m = cov_mask & (src == i)
        if not bool(m.any()):
            continue
        sc = scores_all[m]
        jj = dst[m]
        kbest = int(torch.argmax(sc).item())
        j_best = int(jj[kbest].item())
        if (i, j_best) not in chosen_set:
            _choose(i, j_best)

    k_total = int(max(0, min(int(k_total), M)))
    already = int(A.sum().item())
    if already >= k_total:
        return A.detach().cpu().numpy()

    remain = int(k_total - already)
    if remain <= 0:
        return A.detach().cpu().numpy()

    chosen_mask = torch.zeros((N, N), dtype=torch.bool, device=prob.device)
    if chosen_set:
        ii = torch.tensor([e[0] for e in chosen_set], device=prob.device, dtype=torch.long)
        jj = torch.tensor([e[1] for e in chosen_set], device=prob.device, dtype=torch.long)
        chosen_mask[ii, jj] = True

    keep = ~chosen_mask[src, dst]
    if not bool(keep.any()):
        return A.detach().cpu().numpy()

    scores_rem = scores_all[keep]
    idx_rem = idx[keep]
    src_rem = idx_rem[:, 0]
    dst_rem = idx_rem[:, 1]
    srcL_rem = layer_ids_t[src_rem]
    dstL_rem = layer_ids_t[dst_rem]
    span_rem = (dstL_rem - srcL_rem)

    neigh_mask = (span_rem == 1)
    skip_mask = (span_rem >= 2)

    tsf = float(max(0.0, min(1.0, target_skip_frac)))
    k_skip = int(round(remain * tsf))
    k_neigh = remain - k_skip

    # pick skip first (if requested)
    if k_skip > 0 and bool(skip_mask.any()):
        sc = scores_rem[skip_mask]
        idc = idx_rem[skip_mask]
        k1 = int(min(k_skip, int(idc.size(0))))
        topk = torch.topk(sc, k=k1, largest=True)
        chosen = idc[topk.indices]
        A[chosen[:, 0], chosen[:, 1]] = 1
        remain -= k1
        k_neigh = remain  # whatever left goes to neigh

    if remain <= 0:
        return A.detach().cpu().numpy()

    # then pick neighbor edges
    if bool(neigh_mask.any()):
        sc = scores_rem[neigh_mask]
        idc = idx_rem[neigh_mask]
        k2 = int(min(remain, int(idc.size(0))))
        topk = torch.topk(sc, k=k2, largest=True)
        chosen = idc[topk.indices]
        A[chosen[:, 0], chosen[:, 1]] = 1
        return A.detach().cpu().numpy()

    # if no neighbor edges left, fall back to global
    k3 = int(min(remain, int(idx_rem.size(0))))
    topk = torch.topk(scores_rem, k=k3, largest=True)
    chosen = idx_rem[topk.indices]
    A[chosen[:, 0], chosen[:, 1]] = 1
    return A.detach().cpu().numpy()


@torch.no_grad()
def calibrate_bucket_temp_thr(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    files: List[Path],
    bucket_to_idx: Dict[int, List[int]],
    bi: int,
    k_shot: int,
    calib_n: int,
    temps: List[float],
    thr_list: List[float],
    svec_mode: str,
    min_valid_rate: float,
    seed: int,
    pick: str = "thr",
    topk_gumbel: float = 0.0,
    allow_skip: bool = True,
) -> Dict[str, Any]:
    rng = random.Random(seed + 1000 * bi)
    pool = bucket_to_idx.get(bi, [])
    if len(pool) == 0:
        return {
            "bucket": bucket_name(bi),
            "pick": pick,
            "best_temp": None,
            "best_thr": None,
            "best_topk_scale": None,
            "best_valid_rate": 0.0,
            "best_dens_mean_abs_err": float("inf"),
            "best_mean_ref": 0.0,
            "best_mean_gen": 0.0,
            "p_q10": 0.0, "p_q50": 0.0, "p_q90": 0.0, "p_mean": 0.0,
            "allowed_mean": 0.0,
            "note": "empty bucket",
        }

    if pick not in ("thr", "topk"):
        raise ValueError(f"Unknown pick: {pick}")

    calib_n = int(calib_n)
    keys = list(map(float, thr_list))  # thr list OR topk_scale list
    calib_idxs = [rng.choice(pool) for _ in range(calib_n)]
    results: List[Dict[str, Any]] = []

    for temp in temps:
        p_vals: List[float] = []
        allowed_vals: List[float] = []
        acc = {k: {"valid": 0, "sum_gen": 0.0, "sum_ref": 0.0} for k in keys}

        for idx in calib_idxs:
            Gt = read_graph_any(files[idx])
            layers, node_order = graph_to_layers_and_order(Gt)
            widths = [len(Li) for Li in layers]

            node2layer = {}
            for li, layer_nodes in enumerate(layers):
                for n in layer_nodes:
                    node2layer[n] = li
            layer_ids = [int(node2layer[n]) for n in node_order]

            s_vec = mask_svec(graph_to_svec(Gt), mode=svec_mode).to(DEVICE)
            A_tgt = graph_to_adj_target(Gt, node_order).to(DEVICE)

            X_list = sample_k_support(bucket_to_idx, files, bi, k_shot, rng, exclude=idx)
            z = encode_style(encoder, X_list)

            out = decoder(s_vec, widths=widths, z_style=z)
            A_logits = out[0] if isinstance(out, (tuple, list)) else (out["A_logits"] if isinstance(out, dict) else out)

            prob, valid_mask = logits_to_prob_and_mask(A_logits, temperature=float(temp))
            allowed = float(valid_mask.sum().item())
            allowed_vals.append(allowed)

            if valid_mask.any():
                pv = prob[valid_mask].detach().cpu().flatten().numpy()
                if pv.size > 0:
                    if pv.size > 5000:
                        pv = np.random.choice(pv, size=5000, replace=False)
                    p_vals.extend(pv.tolist())

            dens_ref, _ = dens_in_mask_space_from_target(A_tgt, valid_mask)

            Aref_bin = (A_tgt > 0.5).float()
            ref_stats = graph_dist_stats_from_adj(Aref_bin, layer_ids)
            ref_span = ref_stats["span_hist"]  # [span1, span2, span3, span4, span5+]
            target_skip_frac = float(1.0 - float(ref_span[0]))  # want span>=2 mass

            for key in keys:
                if pick == "thr":
                    A = threshold_adj(prob, valid_mask, thr=float(key))
                else:
                    scale = float(key)
                    k_edges = int(round(float(dens_ref) * float(allowed) * scale))
                    k_edges = max(1, k_edges)
                    k_edges = min(k_edges, int(allowed))

                    A = coverage_topk_adj_from_prob(
                        prob, valid_mask,
                        layer_ids=layer_ids,
                        k_total=k_edges,
                        allow_skip=bool(allow_skip),
                        add_gumbel=float(topk_gumbel),
                        seed=seed + 100000 * bi,  # ✅ calibration 阶段用 bucket-level seed
                        target_skip_frac=target_skip_frac,
                    )

                G = build_graph_from_adj(A)
                rs = strict_reasons(G)

                acc[key]["sum_ref"] += float(dens_ref)
                if not is_valid_strict(rs):
                    continue

                dens_gen = dens_in_mask_space_from_adj(A, allowed=allowed)
                acc[key]["valid"] += 1
                acc[key]["sum_gen"] += float(dens_gen)

        if len(p_vals) > 0:
            pv = np.array(p_vals, dtype=np.float32)
            p_q10 = float(np.quantile(pv, 0.10))
            p_q50 = float(np.quantile(pv, 0.50))
            p_q90 = float(np.quantile(pv, 0.90))
            p_mean = float(pv.mean())
        else:
            p_q10 = p_q50 = p_q90 = p_mean = 0.0

        allowed_mean = float(np.mean(allowed_vals)) if allowed_vals else 0.0

        for key in keys:
            v = int(acc[key]["valid"])
            valid_rate = v / max(1, calib_n)
            mean_ref = acc[key]["sum_ref"] / max(1, calib_n)

            if v > 0:
                mean_gen = acc[key]["sum_gen"] / v
                dens_mean_abs_err = abs(float(mean_gen) - float(mean_ref))
            else:
                mean_gen = 0.0
                dens_mean_abs_err = float("inf")

            row = {
                "bucket": bucket_name(bi),
                "pick": pick,
                "temp": float(temp),
                "valid_rate": float(valid_rate),
                "dens_mean_abs_err": float(dens_mean_abs_err),
                "mean_ref": float(mean_ref),
                "mean_gen": float(mean_gen),
                "p_q10": p_q10, "p_q50": p_q50, "p_q90": p_q90, "p_mean": p_mean,
                "allowed_mean": allowed_mean,
                "valid_cnt": int(v),
            }
            if pick == "thr":
                row["thr"] = float(key)
            else:
                row["topk_scale"] = float(key)
            results.append(row)

    feasible = [r for r in results if r["valid_rate"] >= float(min_valid_rate) and math.isfinite(r["dens_mean_abs_err"])]
    if feasible:
        feasible.sort(key=lambda r: (r["dens_mean_abs_err"], -r["valid_rate"]))
        best = feasible[0]
        note = f"feasible (valid_rate>={min_valid_rate})"
    else:
        results_sorted = sorted(results, key=lambda r: (-r["valid_rate"], r["dens_mean_abs_err"]))
        best = results_sorted[0]
        note = f"fallback (no combo meets valid_rate>={min_valid_rate})"

    return {
        "bucket": bucket_name(bi),
        "pick": pick,
        "best_temp": float(best["temp"]),
        "best_thr": float(best["thr"]) if pick == "thr" and "thr" in best else None,
        "best_topk_scale": float(best["topk_scale"]) if pick == "topk" and "topk_scale" in best else None,
        "best_valid_rate": float(best["valid_rate"]),
        "best_dens_mean_abs_err": float(best["dens_mean_abs_err"]),
        "best_mean_ref": float(best["mean_ref"]),
        "best_mean_gen": float(best["mean_gen"]),
        "p_q10": float(best["p_q10"]),
        "p_q50": float(best["p_q50"]),
        "p_q90": float(best["p_q90"]),
        "p_mean": float(best["p_mean"]),
        "allowed_mean": float(best["allowed_mean"]),
        "note": note,
    }


@torch.no_grad()
def generate_bucket(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    files: List[Path],
    bucket_to_idx: Dict[int, List[int]],
    bi: int,
    k_shot: int,
    tries_per_bucket: int,
    max_tries_per_bucket: int,
    temp: float,
    thr: float,
    svec_mode: str,
    save_valid: int,
    out_dir: Path,
    seed: int,
    pick: str = "thr",
    topk_scale: float = 1.0,
    topk_gumbel: float = 0.0,
    allow_skip: bool = True,
) -> Dict[str, Any]:
   
    import pickle, json
    from networkx.readwrite import json_graph

    rng = random.Random(seed + 9999 * bi)
    pool = bucket_to_idx.get(bi, [])
    if len(pool) == 0:
        return {"bucket": bucket_name(bi), "valid": 0, "invalid": 0, "tries": 0, "note": "empty bucket"}

    if pick not in ("thr", "topk"):
        raise ValueError(f"Unknown pick: {pick}")

    out_bucket_dir = out_dir / bucket_name(bi)
    if save_valid:
        out_bucket_dir.mkdir(parents=True, exist_ok=True)

    valid = 0
    invalid = 0
    tried = 0

    invalid_reasons = {"has_cycle": 0, "multi_source": 0, "multi_sink": 0, "has_isolated": 0}
    dens_ref_list: List[float] = []
    dens_gen_list: List[float] = []
    uniq_signatures: set = set()
    picked_hist: Dict[str, int] = {}

    # distribution logs (valid only)
    ref_dist_list: List[dict] = []
    gen_dist_list: List[dict] = []

    def _json_default(o):
        import numpy as np
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    while valid < tries_per_bucket and tried < max_tries_per_bucket:
        tried += 1
        tidx = rng.choice(pool)

        Gt = read_graph_any(files[tidx])
        layers, node_order = graph_to_layers_and_order(Gt)
        widths = [len(Li) for Li in layers]

        node2layer = {}
        for li, layer_nodes in enumerate(layers):
            for n in layer_nodes:
                node2layer[n] = li
        layer_ids = [int(node2layer[n]) for n in node_order]

        s_vec = mask_svec(graph_to_svec(Gt), mode=svec_mode).to(DEVICE)
        A_tgt = graph_to_adj_target(Gt, node_order).to(DEVICE)

        X_list = sample_k_support(bucket_to_idx, files, bi, k_shot, rng, exclude=tidx)
        z = encode_style(encoder, X_list)

        out = decoder(s_vec, widths=widths, z_style=z)
        A_logits = out[0] if isinstance(out, (tuple, list)) else (out["A_logits"] if isinstance(out, dict) else out)

        prob, valid_mask = logits_to_prob_and_mask(A_logits, temperature=float(temp))
        dens_gt, allowed = dens_in_mask_space_from_target(A_tgt, valid_mask)


        Aref_bin = (A_tgt > 0.5).float()
        ref_stats = graph_dist_stats_from_adj(Aref_bin, layer_ids)
        span_hist = ref_stats.get("span_hist", [])
        span1 = float(span_hist[0]) if isinstance(span_hist, list) and len(span_hist) > 0 else 1.0
        target_skip_frac = float(max(0.0, min(1.0, 1.0 - span1)))

        if pick == "thr":
            A = threshold_adj(prob, valid_mask, thr=float(thr))
            key_str = f"{float(thr):.3f}"
            picked_hist[key_str] = picked_hist.get(key_str, 0) + 1
        else:
            scale = float(topk_scale)
            k_edges = int(round(float(dens_gt) * float(allowed) * scale))
            k_edges = max(1, k_edges)
            k_edges = min(k_edges, int(allowed))

            if tried <= 3:
                print(
                    f"[TOPK] bucket={bucket_name(bi)} allowed={int(allowed)} dens_ref={dens_gt:.5f} "
                    f"scale={scale:.3f} k_edges={k_edges} allow_skip={bool(allow_skip)}"
                )

            A = coverage_topk_adj_from_prob(
                prob, valid_mask,
                layer_ids=layer_ids,
                k_total=k_edges,
                target_skip_frac=target_skip_frac,
                allow_skip=bool(allow_skip),
                add_gumbel=float(topk_gumbel),
                seed=seed + 100000 * bi + tried,
            )

            key_str = f"{scale:.3f}"
            picked_hist[key_str] = picked_hist.get(key_str, 0) + 1

        G = build_graph_from_adj(A)
        rs = strict_reasons(G)

        if not is_valid_strict(rs):
            invalid += 1
            if rs["has_cycle"]: invalid_reasons["has_cycle"] += 1
            if rs["multi_source"]: invalid_reasons["multi_source"] += 1
            if rs["multi_sink"]: invalid_reasons["multi_sink"] += 1
            if rs["has_isolated"]: invalid_reasons["has_isolated"] += 1
            continue

        valid += 1

        dens = dens_in_mask_space_from_adj(A, allowed=allowed)
        dens_ref_list.append(float(dens_gt))
        dens_gen_list.append(float(dens))

        A_bin = torch.from_numpy((A > 0).astype(np.float32)).to(A_tgt.device)

        # save distribution stats (valid only)
        ref_dist_list.append(ref_stats)
        gen_dist_list.append(graph_dist_stats_from_adj(A_bin, layer_ids))

        sig = tuple(map(int, A_bin.flatten().tolist()))
        uniq_signatures.add(hash(sig))

        if save_valid:
            tag = f"thr={thr:.2f}" if pick == "thr" else f"topk_scale={topk_scale:.2f}"
            stem = f"valid_{valid:04d}_N{A.shape[0]}_E{int(A_bin.sum())}_dens{dens:.3f}"

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(A_bin.detach().cpu().numpy(), interpolation="nearest")
            plt.title(f"{bucket_name(bi)} temp={temp:.2f} {tag}\nN={A.shape[0]} E={int(A_bin.sum())} dens={dens:.3f}")
            plt.axis("off")
            plt.tight_layout()
            fig.savefig(out_bucket_dir / f"{stem}.png", dpi=160)
            plt.close(fig)

            with open(out_bucket_dir / f"{stem}.gpickle", "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

            data = json_graph.node_link_data(G, edges="links")
            with open(out_bucket_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=_json_default)

    dist_path = out_dir / f"{bucket_name(bi)}_dist_ref_gen.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump({"bucket": bucket_name(bi), "ref": ref_dist_list, "gen": gen_dist_list}, f, ensure_ascii=False, indent=2)

    def _mean(x): return float(np.mean(x)) if x else 0.0
    def _std(x): return float(np.std(x)) if x else 0.0

    return {
        "bucket": bucket_name(bi),
        "tries": int(tried),
        "valid": int(valid),
        "invalid": int(invalid),
        "valid_rate_over_tried": float(valid / max(1, tried)),
        "pick": str(pick),
        "picked_hist": picked_hist,
        "invalid_reasons": invalid_reasons,
        "ref_dens_mean": _mean(dens_ref_list),
        "ref_dens_std": _std(dens_ref_list),
        "gen_dens_mean": _mean(dens_gen_list),
        "gen_dens_std": _std(dens_gen_list),
        "unique_valid_ratio": float(len(uniq_signatures) / max(1, valid)) if valid > 0 else 0.0,
        "dist_json": str(dist_path),
        "temp": float(temp),
        "thr": float(thr) if pick == "thr" else None,
        "topk_scale": float(topk_scale) if pick == "topk" else None,
        "topk_gumbel": float(topk_gumbel) if pick == "topk" else None,
        "svec_mode": str(svec_mode),
        "allow_skip": bool(allow_skip),
    }




def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="few-shot freeze ckpt (fewshot_best.pt)")
    ap.add_argument("--data_dir", type=str, default="", help="Override DATA_DIR")
    ap.add_argument("--out_dir", type=str, default="infer_fewshot_out")

    ap.add_argument("--k_shot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tries_per_bucket", type=int, default=50)
    ap.add_argument("--max_tries_per_bucket", type=int, default=3000)

    # calibration
    ap.add_argument("--calib_n", type=int, default=120)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1, 1.3, 1.6, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--thr_list", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98])
    ap.add_argument("--min_valid_rate", type=float, default=0.25, help="Calibration feasibility gate. Allow low valid_rate because we can retry.")

    ap.add_argument("--svec_mode", type=str, default="no_E_T", choices=["no_E_T", "no_E", "none"])
    ap.add_argument("--save_valid", type=int, default=1)

    ap.add_argument("--pick", choices=["thr", "topk"], default="thr",
                    help="thr: threshold prob; topk: edge-budget by (dens_ref * allowed * topk_scale)")
    ap.add_argument("--topk_gumbel", type=float, default=0.0,
                    help="optional gumbel noise factor for topk (0=deterministic)")
    ap.add_argument("--topk_scale", type=float, default=1.0,
                    help="edge budget multiplier for topk mode (k = dens_ref * allowed * topk_scale)")

    # NEW: allow_skip (so your CLI works)
    ap.add_argument("--allow_skip", type=int, default=1,
                    help="topk coverage: 1 allow cross-layer edges in coverage (srcL<dstL), 0 only adjacent (dstL=srcL+1)")

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    encoder, decoder, meta = load_fewshot_ckpt(ckpt_path)

    print(f"[OK] Loaded ckpt: {ckpt_path} (d_style={meta['d_style']} ckpt_k_shot={meta['ckpt_k_shot']} run_k_shot={args.k_shot})")
    print(f"[OK] DEVICE={DEVICE}  svec_mode={args.svec_mode}  pick={args.pick}  allow_skip={bool(args.allow_skip)}")
    print(f"[OK] temps={list(map(float, args.temps))}")
    print(f"[OK] thr_list={list(map(float, args.thr_list))}")
    print(f"[OK] calib_n={args.calib_n}  min_valid_rate={args.min_valid_rate}")

    data_dir = Path(args.data_dir) if args.data_dir else Path(DATA_DIR)
    files = load_graph_files(data_dir)
    if not files:
        raise RuntimeError(f"No graph files found under {data_dir}")

    bucket_to_idx = index_files_by_bucket(files)
    print_bucket_counts(bucket_to_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== Per-bucket calibration ==========")
    calib_report: Dict[str, Any] = {"meta": meta, "args": vars(args), "per_bucket": {}}
    chosen: Dict[int, Tuple[float, float]] = {}

    for bi in range(len(BUCKETS)):
        rep = calibrate_bucket_temp_thr(
            encoder=encoder,
            decoder=decoder,
            files=files,
            bucket_to_idx=bucket_to_idx,
            bi=bi,
            k_shot=int(args.k_shot),
            calib_n=int(args.calib_n),
            temps=list(map(float, args.temps)),
            thr_list=list(map(float, args.thr_list)),
            svec_mode=str(args.svec_mode),
            min_valid_rate=float(args.min_valid_rate),
            seed=int(args.seed),
            pick=str(args.pick),
            topk_gumbel=float(args.topk_gumbel),
            allow_skip=bool(args.allow_skip),
        )
        calib_report["per_bucket"][bucket_name(bi)] = rep

        if rep.get("best_temp", None) is None:
            print(f"[CALIB] {bucket_name(bi)} -> empty bucket")
            continue

        t = float(rep["best_temp"])
        if args.pick == "thr":
            val = float(rep["best_thr"])
            print(f"[CALIB] {bucket_name(bi)} -> temp={t:.3f} thr={val:.3f} valid_rate≈{rep['best_valid_rate']:.3f} note={rep['note']}")
        else:
            val = float(rep["best_topk_scale"]) if rep.get("best_topk_scale", None) is not None else float(args.topk_scale)
            print(f"[CALIB] {bucket_name(bi)} -> temp={t:.3f} topk_scale={val:.3f} valid_rate≈{rep['best_valid_rate']:.3f} note={rep['note']}")

        chosen[bi] = (t, val)

    calib_path = out_dir / "infer_fewshot_calibration_report.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved calibration report: {calib_path}\n")

    all_stats: Dict[str, Any] = {"meta": meta, "args": vars(args), "per_bucket": {}}

    for bi in range(len(BUCKETS)):
        if bi not in chosen:
            continue

        bname = bucket_name(bi)
        temp, val = chosen[bi]

        print(f"===== Bucket {bname} ===== tries={args.tries_per_bucket} k_shot={args.k_shot}")

        if args.pick == "thr":
            print(f"[USE] temp={temp:.3f} thr={val:.3f}  (from calibration)")
            use_thr = float(val)
            use_topk_scale = float(args.topk_scale)
        else:
            print(f"[USE] temp={temp:.3f} topk_scale={val:.3f} allow_skip={bool(args.allow_skip)}  (from calibration)")
            use_thr = 0.0
            use_topk_scale = float(val)

        stat = generate_bucket(
            encoder=encoder,
            decoder=decoder,
            files=files,
            bucket_to_idx=bucket_to_idx,
            bi=int(bi),
            k_shot=int(args.k_shot),
            tries_per_bucket=int(args.tries_per_bucket),
            max_tries_per_bucket=int(args.max_tries_per_bucket),
            temp=float(temp),
            thr=float(use_thr),
            svec_mode=str(args.svec_mode),
            save_valid=int(args.save_valid),
            out_dir=out_dir,
            seed=int(args.seed),
            pick=str(args.pick),
            topk_scale=float(use_topk_scale),
            topk_gumbel=float(args.topk_gumbel),
            allow_skip=bool(args.allow_skip),
        )

        all_stats["per_bucket"][bname] = stat

        print(f"[DONE] {bname} tried={stat['tries']} valid={stat['valid']} invalid={stat['invalid']} valid_rate={stat['valid']/max(1,stat['tries']):.3f}")
        print(f"       picked_hist={stat['picked_hist']}")
        print(f"       invalid_reasons={stat['invalid_reasons']}\n")

    stats_path = out_dir / "infer_fewshot_structure1_calibrated_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved stats: {stats_path}")



if __name__ == "__main__":
    main()
