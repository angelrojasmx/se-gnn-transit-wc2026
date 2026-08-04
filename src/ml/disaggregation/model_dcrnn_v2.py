"""
ml/disaggregation/model_dcrnn_v2.py - EventConditionedDiffusionConvGNN (B4 follow-up).

R3.3/R6.4 ask for (a) a DCRNN/STGCN-class competitive baseline AND (b) an
event-conditioned method built on it, so the reviewer can see the FiLM
conditioning idea is not tied to one specific (and possibly favorable) GCN
backbone. model_dcrnn.py already provides (a): DiffusionConvGNN, a bidirectional
K-hop diffusion-convolution backbone evaluated against the production GCN
backbone (see eval_dcrnn_vs_backbone.py, 9 seeds, event-day MAE reported in
R3.3). This module provides (b): the IDENTICAL FiLM mechanism used in
EventConditionedGNN (model_v2.py) - same identity initialization, same
"h = h * gamma + beta right after input_proj, before graph layers" placement,
same from_pretrained()/freeze-backbone/film_parameters() pattern - applied on
top of DiffusionConvGNN instead of the plain GCN. Only the graph propagation
operator differs (P_f/P_b diffusion convolution vs. A_hat symmetric GCN); the
FiLM module itself, its initialization, and its training protocol are unchanged,
so any difference in FiLM's measured benefit can be attributed to the backbone,
not to a different conditioning mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from model_dcrnn import DiffusionConvLayer


class EventConditionedDiffusionConvGNN(nn.Module):
    """DiffusionConvGNN + FiLM event conditioning (mirrors EventConditionedGNN)."""

    N_FEATURES = 7
    HORIZON = 24

    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.2,
        n_layers: int = 2,
        K: int = 2,
        event_emb_dim: int = 384,
        film_hidden: int = 64,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.K = K
        self.event_emb_dim = event_emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.diff_layers = nn.ModuleList([DiffusionConvLayer(hidden, K=K) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.HORIZON),
        )

        # FiLM - identical structure/init to EventConditionedGNN
        self.event_proj_scale = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, hidden),
        )
        self.event_proj_shift = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, hidden),
        )
        nn.init.zeros_(self.event_proj_scale[-1].weight)
        nn.init.ones_(self.event_proj_scale[-1].bias)
        nn.init.zeros_(self.event_proj_shift[-1].weight)
        nn.init.zeros_(self.event_proj_shift[-1].bias)

    def forward(self, x, P_f, P_b, event_emb: torch.Tensor | None = None):
        h = self.input_proj(x)
        if event_emb is not None:
            if event_emb.dim() == 1:
                event_emb = event_emb.unsqueeze(0)
            gamma = self.event_proj_scale(event_emb)
            beta = self.event_proj_shift(event_emb)
            h = h * gamma + beta
        for layer, norm in zip(self.diff_layers, self.norms):
            h_new = torch.relu(layer(h, P_f, P_b))
            h_new = self.dropout(h_new)
            h = norm(h + h_new)
        logits = self.decoder(h)
        return torch.softmax(logits, dim=-1)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        event_emb_dim: int = 384,
        film_hidden: int = 64,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ) -> "EventConditionedDiffusionConvGNN":
        ck = torch.load(checkpoint_path, weights_only=False, map_location=device)
        cfg = ck.get("cfg", {})
        hidden = cfg.get("hidden", 64)
        dropout = cfg.get("dropout", 0.2)
        n_layers = cfg.get("n_layers", 2)
        K = cfg.get("K", 2)

        model = cls(hidden=hidden, dropout=dropout, n_layers=n_layers, K=K,
                    event_emb_dim=event_emb_dim, film_hidden=film_hidden)
        state = ck["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        other_missing = [k for k in missing if "event_proj" not in k]
        if other_missing:
            print(f"  WARNING: non-FiLM missing keys: {other_missing}")

        if freeze_backbone:
            for name, param in model.named_parameters():
                if "event_proj" not in name:
                    param.requires_grad = False

        return model.to(device)

    def film_parameters(self):
        return [p for name, p in self.named_parameters() if "event_proj" in name]
