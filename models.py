# models.py (STRUCTURE v2: Few-shot Encoder + Style-conditioned Decoder)
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_order_from_widths, mask_allowed_pairs



# Few-shot style encoder (minimal)

class FewShotStyleEncoder(nn.Module):
    """
    Minimal graph encoder:
      Input per-graph: padded node feature tensor X [B, Nmax, F], with mask [B, Nmax] (True for valid nodes).
      Output: z [B, d_style]
    """
    def __init__(self, in_dim: int = 3, d_hidden: int = 64, d_style: int = 32, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_hidden, d_style)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        X:    [B, Nmax, F]
        mask: [B, Nmax] boolean
        """
        h = self.mlp(X)  # [B, Nmax, d_hidden]
        m = mask.unsqueeze(-1).float()
        h_sum = (h * m).sum(dim=1)                 # [B, d_hidden]
        denom = m.sum(dim=1).clamp_min(1.0)        # [B, 1]
        h_mean = h_sum / denom
        z = self.proj(h_mean)                      # [B, d_style]
        return z



# Structure decoder

class StructureToGraphDecoder5(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_rank: int = 64,
        max_pos: int = 128,
        d_style: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_rank = max_rank
        self.max_pos = max_pos

        self.rank_emb = nn.Embedding(max_rank + 1, d_model)
        self.pos_emb = nn.Embedding(max_pos + 1, d_model)

        self.s_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.style_proj = nn.Sequential(
            nn.Linear(d_style, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pair = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _clamp_long(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        return torch.clamp(x, min=lo, max=hi).long()

    def forward(
        self,
        s_vec: torch.Tensor,
        widths: List[int],
        z_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        s_vec:   [5] or [1,5]
        widths:  list of ints
        z_style: [d_style] or [1, d_style] (optional)

        return:
          A_logits: [N, N]
        """
        if s_vec.dim() == 1:
            s_vec_b = s_vec.unsqueeze(0)  # [1,5]
        else:
            s_vec_b = s_vec
        if s_vec_b.size(0) != 1:
            raise ValueError("This minimal decoder expects batch size 1 (B=1).")

        order = build_order_from_widths(widths)  # [(li, pos), ...]
        N = len(order)

        ranks = torch.tensor([li for (li, _pos) in order], device=s_vec_b.device)
        pos = torch.tensor([_pos for (_li, _pos) in order], device=s_vec_b.device)

        ranks = self._clamp_long(ranks, 0, self.max_rank)
        pos = self._clamp_long(pos, 0, self.max_pos)

        x = self.rank_emb(ranks) + self.pos_emb(pos)  # [N, d_model]
        x = x.unsqueeze(0)  # [1, N, d_model]

        x = x + self.s_proj(s_vec_b).unsqueeze(1)  # [1,1,d]

        if z_style is not None:
            if z_style.dim() == 1:
                z_b = z_style.unsqueeze(0)  # [1,d_style]
            else:
                z_b = z_style
            x = x + self.style_proj(z_b).unsqueeze(1)

        h = self.encoder(x).squeeze(0)  # [N, d_model]

        hi = h.unsqueeze(1).expand(N, N, self.d_model)
        hj = h.unsqueeze(0).expand(N, N, self.d_model)
        pair_in = torch.cat([hi, hj], dim=-1)        # [N,N,2d]
        logits = self.pair(pair_in).squeeze(-1)      # [N,N]

        allow_np = mask_allowed_pairs(widths)        # [N,N] float 0/1
        allow = torch.tensor(allow_np, device=logits.device).bool()
        logits = logits.masked_fill(~allow, float("-inf"))
        logits.fill_diagonal_(float("-inf"))

        return logits

