# utils.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math

import networkx as nx
import numpy as np
import torch

import os


def topological_layers(G: nx.DiGraph) -> List[List]:
    
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
   
    num_layers = max(2, int(num_layers))
    total_nodes = max(num_layers, int(total_nodes))  

    base = total_nodes // num_layers
    rem = total_nodes % num_layers

    widths = [base] * num_layers
    for i in range(rem):
        widths[i] += 1

    
    widths = [max(1, w) for w in widths]
    return widths


def mask_allowed_pairs(widths: List[int]) -> np.ndarray:
    
    order = build_order_from_widths(widths)
    N = len(order)
    mask = np.zeros((N, N), dtype=np.float32)
    for i, (li, _pi) in enumerate(order):
        for j, (lj, _pj) in enumerate(order):
            if lj > li:  
                mask[i, j] = 1.0
    return mask


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)



def longest_path_time_from_mats(
    A: np.ndarray,
    T: np.ndarray,
    widths: List[int],
) -> float:
    
    N = A.shape[0]
    if N == 0:
        return 0.0

    dp = np.zeros(N, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                cand = dp[i] + T[i, j]
                if cand > dp[j]:
                    dp[j] = cand
    return float(dp.max())



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

    
    if "models" in ckpt:
        
        model_states = ckpt["models"]
    else:
        
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

