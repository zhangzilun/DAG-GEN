# infer_fewshot_structure1_ratio_v2_timehead.py  -- FULL REPLACE (STRUCTURE FIRST, THEN TIME, ENFORCE LP<=300)

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

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
        return self.mlp(x).squeeze(-1)  # predicts log1p(time)


def pool_style(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 1:
        return z.unsqueeze(0)
    if z.dim() == 2 and z.size(0) > 1:
        return z.mean(dim=0, keepdim=True)
    return z


def widths_to_layer_ids(widths: List[int]) -> List[int]:
    layer_ids = []
    for li, w in enumerate(widths):
        layer_ids += [li] * int(w)
    return layer_ids


def node_feats_from_generated(A: np.ndarray, widths: List[int]) -> torch.Tensor:
    """node feat: [layer_norm, indeg_norm, outdeg_norm]"""
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

    return torch.stack([layer_norm, indeg, outdeg], dim=-1)  # [N,3]


def build_edge_feat_from_generated(
    A: np.ndarray,
    widths: List[int],
    lp_bins: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """edge feat = [xi(3), xj(3), span_norm(1), span_onehot(5), layerpair_onehot(lp_bins^2)]"""
    N = int(A.shape[0])
    layer_ids = widths_to_layer_ids(widths)
    L = max(1, len(widths))
    X = node_feats_from_generated(A, widths)  # [N,3]

    edges = [(int(i), int(j)) for i, j in zip(*np.where(A > 0.5))]
    feats: List[torch.Tensor] = []

    for (u, v) in edges:
        li = int(layer_ids[u])
        lj = int(layer_ids[v])

        span = max(1, lj - li)
        span_norm = float(span) / float(max(1, L - 1))
        span_bin = min(5, span)  # 1..5
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
    """Compute longest path time on a DAG with given edge weights (assumes acyclic)."""
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
    G,
    A,
    widths,
    z_style,
    time_head,
    lp_bins,
    label_key,
    integerize,
    tmax_longest_path,
    lp_min=240.0,
):
    import numpy as np
    import networkx as nx
    import torch

    # === build edge features  ===
    e_feat, edges = build_edge_feat_from_generated(A, widths, lp_bins)
    if e_feat.numel() == 0:
        return

    e_feat = e_feat.to(DEVICE)
    z = z_style if z_style.dim() == 2 else z_style.unsqueeze(0)
    z_rep = z.expand(e_feat.size(0), -1)

    with torch.no_grad():
        y_hat = time_head(z_rep, e_feat)
        y = torch.expm1(y_hat).clamp_min(1.0).cpu().numpy().astype(np.float32)

    N = int(A.shape[0])
    Gtmp = nx.DiGraph()
    Gtmp.add_nodes_from(range(N))
    for (u, v), t in zip(edges, y.tolist()):
        Gtmp.add_edge(int(u), int(v), w=float(t))

    topo = list(nx.topological_sort(Gtmp))
    dp = {n: 0.0 for n in topo}
    for u in topo:
        for v in Gtmp.successors(u):
            nv = dp[u] + float(Gtmp.edges[u, v]["w"])
            if dp[v] < nv:
                dp[v] = nv
    lp = max(dp.values()) if dp else 0.0

    
    eps = 1e-6
    if lp > eps:
        if lp < lp_min:
            y *= (lp_min / lp)
        elif lp > tmax_longest_path:
            y *= ((tmax_longest_path - 1e-3) / lp)

    if int(integerize):
        y = np.round(y)

    # === write back ===
    for (u, v), t in zip(edges, y.tolist()):
        if G.has_edge(u, v):
            G.edges[u, v][label_key] = float(t)


def load_structure_ckpt(ckpt_struct: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    enc, dec, meta = INF.load_fewshot_ckpt(ckpt_struct)
    return enc, dec, meta


def load_time_head_ckpt(
    ckpt_time: Path,
    lp_bins: int,
    encoder: Optional[Any] = None,
    decoder: Optional[Any] = None,
) -> Tuple[Any, Any, TimeHead, Dict[str, Any]]:
    """
    Supports:
      A) ckpt_time has encoder/decoder/time_head
      B) ckpt_time only has time_head -> you must provide --ckpt_struct for encoder/decoder
    """
    ckpt = torch.load(str(ckpt_time), map_location=DEVICE)
    if not isinstance(ckpt, dict):
        raise RuntimeError("ckpt_time must be a dict checkpoint")

    d_style = int(ckpt.get("d_style", 32))
    ckpt_k = int(ckpt.get("k_shot", 10))

    # If ckpt_time contains encoder/decoder, load them
    if "encoder" in ckpt and "decoder" in ckpt:
        enc = FewShotStyleEncoder(in_dim=3, d_hidden=64, d_style=d_style).to(DEVICE)
        dec = StructureToGraphDecoder5(d_style=d_style).to(DEVICE)
        enc.load_state_dict(ckpt["encoder"], strict=True)
        dec.load_state_dict(ckpt["decoder"], strict=True)
        encoder = enc
        decoder = dec
    else:
        if encoder is None or decoder is None:
            raise RuntimeError("ckpt_time has no encoder/decoder; please provide --ckpt_struct")

    if "time_head" not in ckpt:
        raise RuntimeError("ckpt_time missing key: time_head")

    time_head = TimeHead(d_style=d_style, lp_bins=int(lp_bins), hidden=256).to(DEVICE)
    time_head.load_state_dict(ckpt["time_head"], strict=True)

    encoder.eval()
    decoder.eval()
    time_head.eval()

    meta = {"d_style": d_style, "ckpt_k_shot": ckpt_k, "time_ckpt": str(ckpt_time)}
    return encoder, decoder, time_head, meta


def save_graph_gpickle_json(G: nx.DiGraph, out_dir: Path, base: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_gpk = out_dir / f"{base}.gpickle"
    p_json = out_dir / f"{base}.json"

    # robust gpickle
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



def calibrate_lp_min_for_bucket(
    files,
    pool,
    encoder,
    decoder,
    time_head,
    bi,
    k_shot,
    calib_n,
    best_temp,
    best_thr,
    best_topk_scale,
    args,
    seed,
    q=0.20,   
):
    import numpy as np
    import random

    rng = random.Random(int(seed) + 7777 * int(bi))
    lp_list = []

    
    for _ in range(int(calib_n)):
        pick = rng.sample(pool, int(k_shot) + 1)
        sup_idx = pick[: int(k_shot)]
        ref_idx = pick[-1]

        sup_graphs = [INF.read_graph_any(files[i]) for i in sup_idx]
        ref_graph = INF.read_graph_any(files[ref_idx])

       
        out = INF.generate_one_strict(
            encoder=encoder,
            decoder=decoder,
            sup_graphs=sup_graphs,
            ref_graph=ref_graph,
            temp=float(best_temp),
            thr=float(best_thr),
            pick=str(args.pick),
            topk_scale=float(best_topk_scale),
            topk_gumbel=float(args.topk_gumbel),
            svec_mode=str(args.svec_mode),
            allow_skip=bool(int(args.allow_skip)),
            seed=int(seed) + 100000 * int(bi) + _,
        )
        if not out.get("ok", False):
            continue

        Ggen = out["G"]           # nx.DiGraph
        A_np = out["A_np"]        # numpy adjacency
        widths = out["widths"]    # list[int]
        z = out["z_style"]        # [1,d]

        lp_raw = predict_lp_raw_only(
            G=Ggen,
            A=A_np,
            widths=widths,
            z_style=z,
            time_head=time_head,
            lp_bins=int(args.lp_bins),
            label_key=str(args.label_key),
            integerize=0,
        )
        if np.isfinite(lp_raw) and lp_raw > 0:
            lp_list.append(float(lp_raw))

    if len(lp_list) == 0:
        return 0.0, {"n": 0, "lp_min": 0.0}

    lp_arr = np.asarray(lp_list, dtype=np.float32)
    lp_min = float(np.quantile(lp_arr, float(q)))
    stats = {
        "n": int(lp_arr.size),
        "q": float(q),
        "lp_min": lp_min,
        "p50": float(np.quantile(lp_arr, 0.50)),
        "p90": float(np.quantile(lp_arr, 0.90)),
        "max": float(lp_arr.max()),
    }
    return lp_min, stats



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"..\data\gpickle2")

    ap.add_argument("--ckpt_struct", type=str, default="", help="structure ckpt (needed if ckpt_time lacks encoder/decoder)")
    ap.add_argument("--ckpt_time", type=str, help="time ckpt with time_head (+ optionally encoder/decoder)")

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
    ap.add_argument("--integerize", type=int, default=1)
    ap.add_argument("--tmax", type=float, default=300.0, help="enforce longest path time <= tmax by scaling all edge times")

    ap.add_argument("--save_valid", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="exp_ratio_v2_timehead_Tmax")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--config", type=str, default="", help="path to json config (optional)")

    -
    def _load_json(p: str) -> Dict[str, Any]:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _apply_cfg(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
        # cfg -> args (only keys that exist in args)
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


    # if --config provided, load and apply it first
    if getattr(args, "config", ""):
        cfg = _load_json(args.config)
        args = _apply_cfg(args, cfg)
    # dump final resolved config for reproducibility
    # NOTE: out_dir may come from cfg
    _dump_resolved(Path(getattr(args, "out_dir")), args)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = INF.load_graph_files(data_dir)
    print(f"[INFO] data_dir={data_dir} files={len(files)}")
    bucket_to_idx = INF.index_files_by_bucket(files)
    INF.print_bucket_counts(bucket_to_idx)

    enc_struct = dec_struct = None
    meta_struct: Dict[str, Any] = {}
    if args.ckpt_struct:
        enc_struct, dec_struct, meta_struct = load_structure_ckpt(Path(args.ckpt_struct))
        print(f"[OK] Loaded structure-ckpt: {args.ckpt_struct} (d_style={meta_struct.get('d_style')} ckpt_k_shot={meta_struct.get('ckpt_k_shot')})")

    encoder, decoder, time_head, meta_time = load_time_head_ckpt(Path(args.ckpt_time), lp_bins=int(args.lp_bins), encoder=enc_struct, decoder=dec_struct)
    print(f"[OK] Loaded time-ckpt: {args.ckpt_time} (d_style={meta_time['d_style']} ckpt_k_shot={meta_time['ckpt_k_shot']} run_k_shot={args.k_shot})")

    if args.ckpt_struct and meta_struct.get("d_style", meta_time["d_style"]) != meta_time["d_style"]:
        print(f"[WARN] d_style mismatch: struct={meta_struct.get('d_style')} time={meta_time['d_style']}")

    for bi in range(len(INF.BUCKETS)):
        bname = INF.bucket_name(bi)
        pool = bucket_to_idx.get(bi, [])
        print(f"\n===== Bucket {bname} ===== tries={args.tries_per_bucket} k_shot={args.k_shot}")
        if len(pool) < args.k_shot + 1:
            print("[SKIP] not enough graphs in this bucket.")
            continue

        calib = INF.calibrate_bucket_temp_thr(
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

        # -------- B2: dynamic lp_min per bucket (from raw LP quantile) --------
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
            z = INF.encode_style(encoder, X_list)

            out_dec = decoder(s_vec, widths=widths, z_style=z)
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
            lp_raw = predict_lp_raw_only(
                A=A_np,
                widths=[int(w) for w in widths],
                z_style=z,
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
            z = INF.encode_style(encoder, X_list)  # [1,d]

            out_dec = decoder(s_vec, widths=widths, z_style=z)
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

            if isinstance(A, torch.Tensor):
                A_np = A.detach().cpu().numpy()
            else:
                A_np = np.asarray(A, dtype=np.float32)

            attach_time_labels(
                G=Ggen_struct,
                A=A_np,
                widths=[int(w) for w in widths],
                z_style=z,
                time_head=time_head,
                lp_bins=int(args.lp_bins),
                label_key=str(args.label_key),
                integerize=int(args.integerize),
                tmax_longest_path=float(args.tmax),
                lp_min = float(lp_min_bucket)
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

