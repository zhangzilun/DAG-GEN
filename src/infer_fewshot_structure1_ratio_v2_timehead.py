# infer_fewshot_structure1_ratio_v2_timehead.py  -- FULL REPLACE
# STRUCTURE FIRST, THEN TIME, ENFORCE LP<=tmax
from __future__ import annotations

import argparse
import json
import math
import random
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE
from models import FewShotStyleEncoder, StructureToGraphDecoder5

import infer_fewshot_structure1_ratio_v2 as INF


class TimeHead(nn.Module):
    def __init__(self, d_style: int, lp_bins: int = 8, hidden: int = 256):
        super().__init__()
        edge_dim = 3 + 3 + 1 + 5 + (lp_bins * lp_bins)
        self.lp_bins = int(lp_bins)
        self.mlp = nn.Sequential(
            nn.Linear(d_style + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, e_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, e_feat], dim=-1)
        return self.mlp(x).squeeze(-1)  


class GNNStyleEncoder(nn.Module):
   
    def __init__(self, in_dim: int = 3, d_hidden: int = 64, d_style: int = 32, n_mp: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_mp = int(n_mp)
        self.lin_in = nn.Linear(in_dim, d_hidden)
        self.lin_self = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(self.n_mp)])
        self.lin_in_nei = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(self.n_mp)])
        self.lin_out_nei = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(self.n_mp)])
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_hidden, d_style)

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        X = X * m.unsqueeze(-1)
        A = (A > 0.5).float()
        A = A * (m.unsqueeze(1) * m.unsqueeze(2))

        h = F.relu(self.lin_in(X))
        h = self.drop(h)

        out_deg = A.sum(dim=2).clamp_min(1.0)
        in_deg = A.sum(dim=1).clamp_min(1.0)

        for k in range(self.n_mp):
            h_out = torch.bmm(A, h) / out_deg.unsqueeze(-1)
            h_in = torch.bmm(A.transpose(1, 2), h) / in_deg.unsqueeze(-1)
            h_new = self.lin_self[k](h) + self.lin_out_nei[k](h_out) + self.lin_in_nei[k](h_in)
            h = F.relu(h_new)
            h = self.drop(h)
            h = h * m.unsqueeze(-1)

        h_sum = (h * m.unsqueeze(-1)).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        h_mean = h_sum / denom
        return self.proj(h_mean)


class StyleEncoderAdapter(nn.Module):
    
    def __init__(self, gnn: GNNStyleEncoder):
        super().__init__()
        self.gnn = gnn

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        K, N, _ = X.shape
        A = torch.zeros((K, N, N), dtype=X.dtype, device=X.device)
        eye = torch.eye(N, dtype=X.dtype, device=X.device).unsqueeze(0).expand(K, -1, -1)
        A = A + eye
        return self.gnn(X, A, M)




import inspect
import torch
import torch.nn as nn
from typing import List

class DecoderAdapter(nn.Module):
  
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        try:
            self._sig = inspect.signature(base.forward)
            self._param_names = [p.name for p in self._sig.parameters.values() if p.name != "self"]
        except Exception:
            self._sig = None
            self._param_names = None

    @staticmethod
    def _norm(name: str) -> str:
        return name.replace("-", "_").lower()

    @staticmethod
    def _expand_to_nodes(z: torch.Tensor, widths: List[int]) -> torch.Tensor:
       
        N = int(sum(int(w) for w in widths))
        if N <= 0:
            return z
        if z.dim() == 2:
            return z.unsqueeze(1).expand(z.size(0), N, z.size(1))
        if z.dim() == 1:
            return z.view(1, 1, -1).expand(1, N, z.numel())
        return z

    @staticmethod
    def _pool_to_global(z: torch.Tensor) -> torch.Tensor:
        
        if z.dim() == 3:
            return z.mean(dim=1)
        return z

    def _call_base(self, s_vec: torch.Tensor, widths: List[int], z_style: torch.Tensor):
     
        if not self._param_names:
            try:
                return self.base(s_vec=s_vec, widths=widths, z_style=z_style)
            except Exception:
                return self.base(s_vec, z_style, widths)

        args = []
        for n in self._param_names:
            nnm = self._norm(n)
            if nnm in ("s_vec", "svec", "s_vector", "s", "struct", "structure", "svec_in"):
                args.append(s_vec)
            elif nnm in ("widths", "w", "ws", "layer_widths", "layerwidths"):
                args.append(widths)
            elif nnm in ("z_style", "z", "style", "zsty", "zstyle", "style_vec", "stylevec"):
                args.append(z_style)
            else:
                p = self._sig.parameters[n]
                if p.default is not inspect._empty:
                    continue
                raise TypeError(f"DecoderAdapter: unknown required param '{n}' in {self._sig}")

        try:
            return self.base(*args)
        except TypeError:

            kwargs = {}
            for n in self._param_names:
                nnm = self._norm(n)
                if nnm in ("s_vec", "svec", "s_vector", "s", "struct", "structure", "svec_in"):
                    kwargs[n] = s_vec
                elif nnm in ("widths", "w", "ws", "layer_widths", "layerwidths"):
                    kwargs[n] = widths
                elif nnm in ("z_style", "z", "style", "zsty", "zstyle", "style_vec", "stylevec"):
                    kwargs[n] = z_style
            return self.base(**kwargs)

    def forward(self, s_vec: torch.Tensor, widths: List[int], z_style: torch.Tensor):
        z1 = z_style
        z2 = self._expand_to_nodes(z_style, widths)
        z3 = self._pool_to_global(z2)

        last_err = None
        for zi in (z1, z2, z3):
            try:
                return self._call_base(s_vec, widths, zi)
            except RuntimeError as e:
                msg = str(e)
                last_err = e
                continue

        raise last_err

def widths_to_layer_ids(widths: List[int]) -> List[int]:
    layer_ids: List[int] = []
    for li, w in enumerate(widths):
        layer_ids += [li] * int(w)
    return layer_ids


def node_feats_from_generated(A: np.ndarray, widths: List[int]) -> torch.Tensor:
    
    N = int(A.shape[0])
    layer_ids = widths_to_layer_ids(widths)
    L = max(1, len(widths))

    layer_norm = torch.tensor(
        [float(layer_ids[i]) / float(max(1, L - 1)) for i in range(N)],
        dtype=torch.float32,
    )

    indeg = torch.tensor(A.sum(axis=0), dtype=torch.float32)
    outdeg = torch.tensor(A.sum(axis=1), dtype=torch.float32)
    indeg = indeg / indeg.max().clamp_min(1.0)
    outdeg = outdeg / outdeg.max().clamp_min(1.0)

    return torch.stack([layer_norm, indeg, outdeg], dim=-1)  


def build_edge_feat_from_generated(
    A: np.ndarray,
    widths: List[int],
    lp_bins: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    
    N = int(A.shape[0])
    layer_ids = widths_to_layer_ids(widths)
    L = max(1, len(widths))
    X = node_feats_from_generated(A, widths)

    edges = [(int(i), int(j)) for i, j in zip(*np.where(A > 0.5))]
    feats: List[torch.Tensor] = []

    for (u, v) in edges:
        li = int(layer_ids[u])
        lj = int(layer_ids[v])

        span = max(1, lj - li)
        span_norm = float(span) / float(max(1, L - 1))
        span_bin = min(5, span)  
        span_oh = torch.zeros((5,), dtype=torch.float32)
        span_oh[span_bin - 1] = 1.0

        a = int(math.floor((li / max(1, L - 1)) * lp_bins))
        b = int(math.floor((lj / max(1, L - 1)) * lp_bins))
        a = max(0, min(lp_bins - 1, a))
        b = max(0, min(lp_bins - 1, b))
        lp_idx = a * lp_bins + b
        lp_oh = torch.zeros((lp_bins * lp_bins,), dtype=torch.float32)
        lp_oh[lp_idx] = 1.0

        feats.append(torch.cat([X[u], X[v], torch.tensor([span_norm], dtype=torch.float32), span_oh, lp_oh], dim=0))

    edge_dim = 3 + 3 + 1 + 5 + (lp_bins * lp_bins)
    if not feats:
        return torch.zeros((0, edge_dim), dtype=torch.float32), edges
    return torch.stack(feats, dim=0), edges


def longest_path_time_from_edges(N: int, edges: List[Tuple[int, int]], w: np.ndarray) -> float:
   
    Gtmp = nx.DiGraph()
    Gtmp.add_nodes_from(range(N))
    for (u, v), t in zip(edges, w.tolist()):
        Gtmp.add_edge(int(u), int(v), w=float(t))
    topo = list(nx.topological_sort(Gtmp))
    dp = {n: 0.0 for n in topo}
    for u in topo:
        for v in Gtmp.successors(u):
            t = float(Gtmp.edges[u, v]["w"])
            if dp[v] < dp[u] + t:
                dp[v] = dp[u] + t
    return float(max(dp.values()) if dp else 0.0)


def predict_lp_raw_only(
    A: np.ndarray,
    widths: List[int],
    z_style: torch.Tensor,
    time_head: nn.Module,
    lp_bins: int,
) -> float:
    e_feat, edges = build_edge_feat_from_generated(A, widths, lp_bins)
    if e_feat.numel() == 0 or len(edges) == 0:
        return 0.0

    e_feat = e_feat.to(DEVICE)
    z = z_style if z_style.dim() == 2 else z_style.unsqueeze(0)
    z_rep = z.expand(e_feat.size(0), -1)

    with torch.no_grad():
        y_hat = time_head(z_rep, e_feat)
        y = torch.expm1(y_hat).clamp_min(1.0).detach().cpu().numpy().astype(np.float32)

    N = int(A.shape[0])
    return longest_path_time_from_edges(N, edges, y)


def attach_time_labels(
    G: nx.DiGraph,
    A: np.ndarray,
    widths: List[int],
    z_style: torch.Tensor,
    time_head: nn.Module,
    lp_bins: int,
    label_key: str,
    integerize: int,
    tmax_longest_path: float,
    lp_min: float = 0.0,
) -> None:

    e_feat, edges = build_edge_feat_from_generated(A, widths, lp_bins)
    if e_feat.numel() == 0 or len(edges) == 0:
        return

    e_feat = e_feat.to(DEVICE)
    z = z_style if z_style.dim() == 2 else z_style.unsqueeze(0)
    z_rep = z.expand(e_feat.size(0), -1)

    with torch.no_grad():
        y_hat = time_head(z_rep, e_feat)
        y = torch.expm1(y_hat).clamp_min(1.0).cpu().numpy().astype(np.float32)

    N = int(A.shape[0])
    lp = longest_path_time_from_edges(N, edges, y)

    eps = 1e-6
    if lp > eps:
        if lp_min and lp < float(lp_min):
            y *= (float(lp_min) / lp)
        if tmax_longest_path and lp > float(tmax_longest_path):
            y *= ((float(tmax_longest_path) - 1e-3) / lp)

    if int(integerize):
        y = np.round(y)

    for (u, v), t in zip(edges, y.tolist()):
        if G.has_edge(u, v):
            G.edges[u, v][label_key] = float(t)

def _copy_partial(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    out = dst.clone()
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    out[slices] = src[slices]
    return out

def _remap_decoder_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
   
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("s_proj."):
            continue

        if k.startswith("style_proj.0."):
            out[k.replace("style_proj.0.", "style_proj.")] = v
            continue

        if k.startswith("pair.0."):
            out[k.replace("pair.0.", "edge_mlp.0.")] = v
            continue
        if k.startswith("pair.2."):
            out[k.replace("pair.2.", "edge_mlp.3.")] = v
            continue

        out[k] = v
    return out

def smart_load_decoder_v2(decoder: nn.Module, sd_raw: Dict[str, torch.Tensor]) -> None:
   
    sd = _remap_decoder_keys(sd_raw)
    cur = decoder.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        if k not in cur:
            continue
        if cur[k].shape == v.shape:
            new_sd[k] = v
        else:
            if ("rank_emb.weight" in k) or ("pos_emb.weight" in k):
                new_sd[k] = _copy_partial(cur[k], v)
            else:
                continue

    cur.update(new_sd)
    decoder.load_state_dict(cur, strict=True)

def load_structure_ckpt(ckpt_struct: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_struct), map_location="cpu")

    enc_sd = ckpt.get("encoder", {}) or {}
    is_gnn = any(k.startswith("lin_in.") for k in enc_sd.keys()) or ("lin_in.weight" in enc_sd)

    d_style = int(ckpt.get("d_style", 32))
    if is_gnn:
        gnn = GNNStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style, n_mp=int(ckpt.get("n_mp", 2))).to(DEVICE)
        gnn.load_state_dict(enc_sd, strict=True)
        enc = StyleEncoderAdapter(gnn).to(DEVICE) 
        enc_kind = "GNNStyleEncoder(+Adapter)"
    else:
        enc = FewShotStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style).to(DEVICE)
        if "encoder" in ckpt:
            enc.load_state_dict(enc_sd, strict=True)
        enc_kind = "FewShotStyleEncoder"

    dec = StructureToGraphDecoder5(max_rank=64).to(DEVICE)
    if "decoder" in ckpt:
        try:
            dec.load_state_dict(ckpt["decoder"], strict=True)
        except Exception:
            smart_load_decoder_v2(dec, ckpt["decoder"])

    meta = ckpt.get("meta", {}) or {}
    meta["loaded_with"] = "smart_load_decoder_v2(max_rank=64)"
    meta["enc_kind"] = enc_kind
    meta["d_style"] = d_style
    return enc, dec, meta


def load_time_head_ckpt(ckpt_path: Path, lp_bins: int):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    d_style = int(ckpt.get("d_style", 32))
    k_shot = int(ckpt.get("k_shot", 10))

    enc_sd = ckpt.get("encoder", {}) or {}
    is_gnn = any(k.startswith("lin_in.") for k in enc_sd.keys()) or ("lin_in.weight" in enc_sd)

    if is_gnn:
        enc = GNNStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style, n_mp=2).to(DEVICE)
        enc.load_state_dict(enc_sd, strict=True)
        enc = StyleEncoderAdapter(enc).to(DEVICE)
        enc_kind = "GNNStyleEncoder(+Adapter)"
    else:
        enc = FewShotStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style).to(DEVICE)
        enc.load_state_dict(enc_sd, strict=True)
        enc_kind = "FewShotStyleEncoder"

    dec = StructureToGraphDecoder5().to(DEVICE)
    if "decoder" in ckpt:
        try:
            dec.load_state_dict(ckpt["decoder"], strict=True)
        except Exception:
            dec.load_state_dict(ckpt["decoder"], strict=False)
    dec = DecoderAdapter(dec).to(DEVICE)

    time_head = TimeHead(d_style=d_style, lp_bins=int(lp_bins), hidden=256).to(DEVICE)
    time_head.load_state_dict(ckpt["time_head"], strict=True)

    meta = ckpt.get("meta", {}) or {}
    meta["enc_kind"] = enc_kind
    meta["d_style"] = d_style
    meta["ckpt_k_shot"] = k_shot
    return enc, dec, time_head, meta


def save_graph_gpickle_json(G: nx.DiGraph, out_dir: Path, base: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_gpk = out_dir / f"{base}.gpickle"
    p_json = out_dir / f"{base}.json"

    try:
        from networkx.readwrite.gpickle import write_gpickle as _w
        _w(G, p_gpk)
    except Exception:
        import pickle
        with open(p_gpk, "wb") as f:
            pickle.dump(G, f)

    data = nx.node_link_data(G, edges="links")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"..\data\gpickle2")

    ap.add_argument("--ckpt_struct", type=str, default="", help="structure ckpt (optional)")
    ap.add_argument("--ckpt_time", type=str, default="", help="time ckpt with time_head (+ encoder)")

    ap.add_argument("--k_shot", type=int, default=10)
    ap.add_argument("--pick", type=str, default="topk", choices=["thr", "topk"])
    ap.add_argument("--allow_skip", type=int, default=1)
    ap.add_argument("--tries_per_bucket", type=int, default=50)
    ap.add_argument("--max_tries_per_bucket", type=int, default=12000)
    ap.add_argument("--calib_n", type=int, default=600)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30])
    ap.add_argument("--thr_list", type=float, nargs="+", default=[0.80, 0.90, 1.00, 1.10, 1.20, 1.30])
    ap.add_argument("--topk_gumbel", type=float, default=0.0)
    ap.add_argument("--min_valid_rate", type=float, default=0.70)
    ap.add_argument("--svec_mode", type=str, default="no_E_T")

    ap.add_argument("--lp_bins", type=int, default=8)
    ap.add_argument("--label_key", type=str, default="label")
    ap.add_argument("--integerize", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=300.0)

    ap.add_argument("--save_valid", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="exp_ratio_v2_timehead_Tmax")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--config", type=str, default="", help="path to json config (optional)")

    def _load_json(p: str) -> Dict[str, Any]:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _apply_cfg(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        return args

    def _dump_resolved(out_dir: Path, args: argparse.Namespace) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = {k: getattr(args, k) for k in vars(args).keys()}
        with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(resolved, f, indent=2, ensure_ascii=False)
        try:
            import sys
            with open(out_dir / "run_cmd.txt", "w", encoding="utf-8") as f:
                f.write(" ".join(sys.argv))
        except Exception:
            pass

    args = ap.parse_args()
    if getattr(args, "config", ""):
        cfg = _load_json(args.config)
        args = _apply_cfg(args, cfg)

    out_dir = Path(getattr(args, "out_dir"))
    _dump_resolved(out_dir, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = INF.load_graph_files(data_dir)
    print(f"[INFO] data_dir={data_dir} files={len(files)}")
    bucket_to_idx = INF.index_files_by_bucket(files)
    INF.print_bucket_counts(bucket_to_idx)

    if args.ckpt_struct:
        struct_encoder, struct_decoder, meta_struct = load_structure_ckpt(Path(args.ckpt_struct))
        struct_decoder = DecoderAdapter(struct_decoder).to(DEVICE)
        print(f"[OK] Loaded structure-ckpt: {args.ckpt_struct}")
    else:
        struct_encoder = None
        struct_decoder = None
        meta_struct = {}

    time_encoder, time_decoder_unused, time_head, meta_time = load_time_head_ckpt(Path(args.ckpt_time), lp_bins=int(args.lp_bins))
    print(f"[OK] Loaded time-ckpt: {args.ckpt_time} (d_style={meta_time['d_style']} ckpt_k_shot={meta_time['ckpt_k_shot']} run_k_shot={args.k_shot})")
    print(f"[OK] time-encoder-kind: {meta_time['enc_kind']}")
    print(f"[OK] structure-encoder-kind: {meta_struct['enc_kind']}")
    if struct_encoder is None:
        struct_encoder = time_encoder
        struct_decoder = time_decoder_unused
        print("[WARN] ckpt_struct not provided -> using time-ckpt encoder/decoder for structure stage (best-effort).")

    for bi in range(len(INF.BUCKETS)):
        bname = INF.bucket_name(bi)
        pool = bucket_to_idx.get(bi, [])
        print(f"\n===== Bucket {bname} ===== tries={args.tries_per_bucket} k_shot={args.k_shot}")
        if len(pool) < args.k_shot + 1:
            print("[SKIP] not enough graphs in this bucket.")
            continue

        calib = INF.calibrate_bucket_temp_thr(
            encoder=struct_encoder,
            decoder=struct_decoder,
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
            allow_skip=bool(int(args.allow_skip)),
        )

        best_temp = calib.get("best_temp", None)
        if best_temp is None:
            print("[SKIP] calibration returned no best_temp")
            continue

        if str(args.pick) == "thr":
            best_thr = float(calib.get("best_thr", args.thr_list[0]))
            best_topk_scale = 1.0
            print(f"[CALIB] pick=thr best_temp={best_temp:.3f} best_thr={best_thr:.4f} best_valid_rate={calib.get('best_valid_rate',0):.3f} dens_err={calib.get('best_dens_mean_abs_err',0):.4f}")
        else:
            best_thr = 0.0
            best_topk_scale = float(calib.get("best_topk_scale", args.thr_list[0]))
            print(f"[CALIB] pick=topk best_temp={best_temp:.3f} best_topk_scale={best_topk_scale:.3f} best_valid_rate={calib.get('best_valid_rate',0):.3f} dens_err={calib.get('best_dens_mean_abs_err',0):.4f}")

        lp_q = 0.20
        lp_calib_n = 200
        lp_list: List[float] = []
        lp_rng = random.Random(int(args.seed) + 12345 * bi)

        lp_tried = 0
        while len(lp_list) < lp_calib_n and lp_tried < lp_calib_n * 50:
            lp_tried += 1
            tidx = lp_rng.choice(pool)

            Gt = INF.read_graph_any(files[tidx])
            layers, node_order = INF.graph_to_layers_and_order(Gt)
            widths = [len(Li) for Li in layers]

            node2layer = {}
            for li, layer_nodes in enumerate(layers):
                for n in layer_nodes:
                    node2layer[n] = li
            layer_ids = [int(node2layer[n]) for n in node_order]

            s_vec = INF.mask_svec(INF.graph_to_svec(Gt), mode=str(args.svec_mode)).to(DEVICE)
            A_tgt = INF.graph_to_adj_target(Gt, node_order).to(DEVICE)

            X_list = INF.sample_k_support(bucket_to_idx, files, bi, int(args.k_shot), lp_rng, exclude=tidx)
            z_struct = INF.encode_style(struct_encoder, X_list)  
            out_dec = struct_decoder(s_vec, widths=widths, z_style=z_struct)
            A_logits = out_dec[0] if isinstance(out_dec, (tuple, list)) else (out_dec["A_logits"] if isinstance(out_dec, dict) else out_dec)

            prob, valid_mask = INF.logits_to_prob_and_mask(A_logits, temperature=float(best_temp))
            dens_ref, allowed = INF.dens_in_mask_space_from_target(A_tgt, valid_mask)

            if str(args.pick) == "thr":
                A = INF.threshold_adj(prob, valid_mask, thr=float(best_thr))
            else:
                Aref_bin = (A_tgt > 0.5).float()
                ref_stats = INF.graph_dist_stats_from_adj(Aref_bin, layer_ids)
                ref_span = ref_stats["span_hist"]
                target_skip_frac = float(1.0 - float(ref_span[0]))

                k_edges = int(round(float(dens_ref) * float(allowed) * float(best_topk_scale)))
                k_edges = max(1, min(k_edges, int(allowed)))

                A = INF.coverage_topk_adj_from_prob(
                    prob, valid_mask,
                    layer_ids=layer_ids,
                    k_total=k_edges,
                    allow_skip=bool(int(args.allow_skip)),
                    add_gumbel=float(args.topk_gumbel),
                    seed=int(args.seed) + 888888 * bi + lp_tried,
                    target_skip_frac=target_skip_frac,
                )

            Gtmp = INF.build_graph_from_adj(A)
            rs = INF.strict_reasons(Gtmp)
            if not INF.is_valid_strict(rs):
                continue

            A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else np.asarray(A, dtype=np.float32)

            z_time = INF.encode_style(time_encoder, X_list)
            lp_raw = predict_lp_raw_only(
                A=A_np,
                widths=[int(w) for w in widths],
                z_style=z_time,
                time_head=time_head,
                lp_bins=int(args.lp_bins),
            )
            if math.isfinite(lp_raw) and lp_raw > 0:
                lp_list.append(float(lp_raw))

        if len(lp_list) == 0:
            lp_min_bucket = 0.0
            print("[LP-CALIB] no valid samples, lp_min_bucket=0")
        else:
            lp_arr = np.asarray(lp_list, dtype=np.float32)
            lp_min_bucket = float(np.quantile(lp_arr, lp_q))
            print(f"[LP-CALIB] n={len(lp_list)} q={lp_q:.2f} lp_min_bucket={lp_min_bucket:.2f} p50={float(np.quantile(lp_arr,0.5)):.2f} p90={float(np.quantile(lp_arr,0.9)):.2f} max={float(lp_arr.max()):.2f}")

        rng = random.Random(int(args.seed) + 9999 * bi)
        valid = 0
        invalid = 0
        tried = 0
        invalid_reasons = {"has_cycle": 0, "multi_source": 0, "multi_sink": 0, "has_isolated": 0}
        saved_id = 0

        out_bucket_dir = out_dir / bname
        if int(args.save_valid) == 1:
            out_bucket_dir.mkdir(parents=True, exist_ok=True)

        while valid < int(args.tries_per_bucket) and tried < int(args.max_tries_per_bucket):
            tried += 1
            tidx = rng.choice(pool)

            Gt = INF.read_graph_any(files[tidx])
            layers, node_order = INF.graph_to_layers_and_order(Gt)
            widths = [len(Li) for Li in layers]

            node2layer = {}
            for li, layer_nodes in enumerate(layers):
                for n in layer_nodes:
                    node2layer[n] = li
            layer_ids = [int(node2layer[n]) for n in node_order]

            s_vec = INF.mask_svec(INF.graph_to_svec(Gt), mode=str(args.svec_mode)).to(DEVICE)
            A_tgt = INF.graph_to_adj_target(Gt, node_order).to(DEVICE)

            X_list = INF.sample_k_support(bucket_to_idx, files, bi, int(args.k_shot), rng, exclude=tidx)
            z_struct = INF.encode_style(struct_encoder, X_list)

            out_dec = struct_decoder(s_vec, widths=widths, z_style=z_struct)
            A_logits = out_dec[0] if isinstance(out_dec, (tuple, list)) else (out_dec["A_logits"] if isinstance(out_dec, dict) else out_dec)

            prob, valid_mask = INF.logits_to_prob_and_mask(A_logits, temperature=float(best_temp))
            dens_ref, allowed = INF.dens_in_mask_space_from_target(A_tgt, valid_mask)

            if str(args.pick) == "thr":
                A = INF.threshold_adj(prob, valid_mask, thr=float(best_thr))
            else:
                Aref_bin = (A_tgt > 0.5).float()
                ref_stats = INF.graph_dist_stats_from_adj(Aref_bin, layer_ids)
                ref_span = ref_stats["span_hist"]
                target_skip_frac = float(1.0 - float(ref_span[0]))

                k_edges = int(round(float(dens_ref) * float(allowed) * float(best_topk_scale)))
                k_edges = max(1, min(k_edges, int(allowed)))

                A = INF.coverage_topk_adj_from_prob(
                    prob, valid_mask,
                    layer_ids=layer_ids,
                    k_total=k_edges,
                    allow_skip=bool(int(args.allow_skip)),
                    add_gumbel=float(args.topk_gumbel),
                    seed=int(args.seed) + 100000 * bi + tried,
                    target_skip_frac=target_skip_frac,
                )

            Ggen_struct = INF.build_graph_from_adj(A)
            rs = INF.strict_reasons(Ggen_struct)
            if not INF.is_valid_strict(rs):
                invalid += 1
                invalid_reasons["has_cycle"] += int(rs.get("has_cycle", 0))
                invalid_reasons["multi_source"] += int(rs.get("multi_source", 0))
                invalid_reasons["multi_sink"] += int(rs.get("multi_sink", 0))
                invalid_reasons["has_isolated"] += int(rs.get("has_isolated", 0))
                continue

            A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else np.asarray(A, dtype=np.float32)

           
            z_time = INF.encode_style(time_encoder, X_list)

            attach_time_labels(
                G=Ggen_struct,
                A=A_np,
                widths=[int(w) for w in widths],
                z_style=z_time,
                time_head=time_head,
                lp_bins=int(args.lp_bins),
                label_key=str(args.label_key),
                integerize=int(args.integerize),
                tmax_longest_path=float(args.tmax),
                lp_min=float(lp_min_bucket),
            )

            valid += 1
            if int(args.save_valid) == 1:
                saved_id += 1
                base = f"{bname}_valid_{saved_id:04d}"
                save_graph_gpickle_json(Ggen_struct, out_bucket_dir, base)

        rate = float(valid / max(1, tried))
        print(f"[DONE] {bname} tried={tried} valid={valid} invalid={invalid} valid_rate={rate:.3f}")
        print(f"       invalid_reasons={invalid_reasons}")


if __name__ == "__main__":
    main()
