# models.py (STRUCTURE v2: Few-shot Encoder + Style-conditioned Decoder)
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_order_from_widths, mask_allowed_pairs



class FewShotStyleEncoder(nn.Module):
    
    def __init__(self, in_dim: int = 3, d_hidden: int = 64, d_style: int = 32, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_hidden, d_style)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.mlp(X)  
        m = mask.unsqueeze(-1).float()
        h_sum = (h * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return self.proj(h_sum / denom)


class GNNStyleEncoder(nn.Module):
    
    def __init__(
        self,
        in_dim: int = 3,
        d_hidden: int = 64,
        d_style: int = 32,
        n_mp: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mp = n_mp
        self.lin_in = nn.Linear(in_dim, d_hidden)

      
        self.lin_self = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_mp)])
        self.lin_in_nei = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_mp)])
        self.lin_out_nei = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(n_mp)])

        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_hidden, d_style)

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = X.shape
        m = mask.float()  

        X = X * m.unsqueeze(-1)

        h = F.relu(self.lin_in(X))
        h = self.drop(h)


        A = (A > 0.5).float()
        A = A * (m.unsqueeze(1) * m.unsqueeze(2)) 

        out_deg = A.sum(dim=2).clamp_min(1.0)    
        in_deg  = A.sum(dim=1).clamp_min(1.0)       

        for k in range(self.n_mp):
            
            h_out = torch.bmm(A, h) / out_deg.unsqueeze(-1)

            h_in = torch.bmm(A.transpose(1, 2), h) / in_deg.unsqueeze(-1)

            h_new = (
                self.lin_self[k](h) +
                self.lin_out_nei[k](h_out) +
                self.lin_in_nei[k](h_in)
            )
            h = F.relu(h_new)
            h = self.drop(h)
            h = h * m.unsqueeze(-1)  

        
        h_sum = (h * m.unsqueeze(-1)).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        h_mean = h_sum / denom
        z = self.proj(h_mean)
        return z


class StructureToGraphDecoder5(nn.Module):
   
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_rank: int = 32,
        d_style: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_style = d_style
        self.max_rank = max_rank

        self.rank_emb = nn.Embedding(max_rank + 1, d_model)
        self.pos_emb = nn.Embedding(512, d_model)   safe upper bound

        self.style_proj = nn.Linear(d_style, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

       
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, widths: List[int], z_style: torch.Tensor) -> torch.Tensor:
        order = build_order_from_widths(widths)
        N = len(order)

      
        if z_style.dim() == 3:
            
            z_style = z_style.flatten(0, 1)
        elif z_style.dim() == 1:
           
            z_style = z_style.unsqueeze(0)
        elif z_style.dim() != 2:
            raise ValueError(f"z_style must be 1D/2D/3D, got shape={tuple(z_style.shape)}")

        B2 = z_style.shape[0]

      
        ranks = torch.tensor(
            [li for (li, _pj) in order],
            device=z_style.device,
            dtype=torch.long
        ).clamp(0, self.max_rank)

        pos = torch.arange(N, device=z_style.device, dtype=torch.long)

        x0 = self.rank_emb(ranks) + self.pos_emb(pos)  

        x = x0.unsqueeze(0).expand(B2, N, self.d_model).contiguous()
        style = self.style_proj(z_style).unsqueeze(1) 
        x = x + style 

        h = self.encoder(x)  

        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([hi, hj], dim=-1)
        logits = self.edge_mlp(pair).squeeze(-1) 

        allowed = torch.from_numpy(mask_allowed_pairs(widths)).to(logits.device)  
        logits = logits + (allowed.unsqueeze(0) - 1.0) * 1e9
        return logits
