# utils.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math

import networkx as nx
import numpy as np
import torch

import os


# ------------------------------------------------------------
#  DAG 层次化 & 拓扑工具
# ------------------------------------------------------------
def topological_layers(G: nx.DiGraph) -> List[List]:
    """
    按拓扑排序把 DAG 分成多层：
        第一层：没有入边的点
        后面每一层：入边都来自前面层的点
    如果有环会直接抛异常。
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph is not a DAG (cycle detected).")

    in_deg = {n: d for n, d in G.in_degree()}
    remaining = set(G.nodes())
    layers: List[List] = []

    while remaining:
        this_layer = [n for n in remaining if in_deg[n] == 0]
        if not this_layer:
            raise ValueError("Cycle detected while constructing layers.")
        layers.append(this_layer)
        for n in this_layer:
            remaining.remove(n)
            for _, v in G.out_edges(n):
                in_deg[v] -= 1

    return layers


def build_order_from_widths(widths):
    """
    Build node order from per-layer widths.
    Robust to torch tensors / floats.
    """
    # --- coerce widths to python list[int] ---
    try:
        if hasattr(widths, "detach"):
            widths = widths.detach().cpu()
        if hasattr(widths, "dim") and widths.dim() > 1:
            widths = widths[0]
        if hasattr(widths, "view"):
            widths = widths.view(-1)
        if hasattr(widths, "tolist"):
            widths = widths.tolist()
    except Exception:
        pass

    if not isinstance(widths, (list, tuple)):
        widths = [widths]

    widths = [int(round(float(w))) for w in widths]
    widths = [max(0, w) for w in widths]

    order = []
    for li, w in enumerate(widths):
        for pos in range(w):
            order.append((li, pos))
    return order



def distribute_nodes_across_layers(total_nodes: int, num_layers: int) -> List[int]:
    """
    给定节点数和层数，把节点平均分配到各层，至少每层 1 个。
    """
    num_layers = max(2, int(num_layers))
    total_nodes = max(num_layers, int(total_nodes))  # 至少每层 1 个

    base = total_nodes // num_layers
    rem = total_nodes % num_layers

    widths = [base] * num_layers
    for i in range(rem):
        widths[i] += 1

    # 保证不会出现 0
    widths = [max(1, w) for w in widths]
    return widths


def mask_allowed_pairs(widths: List[int]) -> np.ndarray:
    """
    返回 NxN 的 mask，1 表示允许的边 (前一层 -> 后一层)，否则 0。
    """
    order = build_order_from_widths(widths)
    N = len(order)
    mask = np.zeros((N, N), dtype=np.float32)
    for i, (li, _pi) in enumerate(order):
        for j, (lj, _pj) in enumerate(order):
            if lj > li:  # 只能连到更靠后的层
                mask[i, j] = 1.0
    return mask


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


# ------------------------------------------------------------
#  从矩阵计算最长路径时间（用于 loss / 评估）
# ------------------------------------------------------------
def longest_path_time_from_mats(
    A: np.ndarray,
    T: np.ndarray,
    widths: List[int],
) -> float:
    """
    给定邻接矩阵 A (0/1) 和时间矩阵 T，假设节点顺序已经按照
    build_order_from_widths(widths) 排好（也就是拓扑顺序）。
    返回整张图的最长路径时间。
    """
    N = A.shape[0]
    if N == 0:
        return 0.0

    # 简单 DP：拓扑顺序就是 0..N-1
    dp = np.zeros(N, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                cand = dp[i] + T[i, j]
                if cand > dp[j]:
                    dp[j] = cand
    return float(dp.max())

# ------------------------------------------------------------
#  Checkpoint utilities (for training / finetuning)
# ------------------------------------------------------------


def save_checkpoint(
    model_dict,
    optimizer,
    epoch: int,
    best_loss: float,
    ckpt_dir: str,
    is_best: bool = False,
):
    """
    model_dict: dict of torch.nn.Module, e.g.
        {"encoder": encoder, "decoder": decoder}
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "models": {k: v.state_dict() for k, v in model_dict.items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }

    last_path = os.path.join(ckpt_dir, "fewshot_last.pt")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, "fewshot_best.pt")
        torch.save(state, best_path)


def load_checkpoint(
    model_dict,
    ckpt_path: str,
    device: torch.device = "cpu",
):
    """
    Load checkpoint into model_dict.
    Compatible with BOTH:
    - old format: encoder / decoder at top-level
    - new format: ckpt["models"][name]

    Returns (start_epoch, best_loss)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # -------- detect format --------
    if "models" in ckpt:
        # new format (v4.1+)
        model_states = ckpt["models"]
    else:
        # old format (v3 / v4)
        model_states = ckpt

    for name, model in model_dict.items():
        if name in model_states:
            model.load_state_dict(model_states[name])
        else:
            raise KeyError(
                f"Model '{name}' not found in checkpoint. "
                f"Available keys: {list(model_states.keys())}"
            )

    start_epoch = ckpt.get("epoch", -1) + 1
    best_loss = ckpt.get("best_loss", float("inf"))

    return start_epoch, best_loss
