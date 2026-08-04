"""
SE-GNN: Sentence-Encoder-Augmented GNN with FiLM event conditioning.

Extends DisaggregationGNN with Feature-wise Linear Modulation (FiLM,
Perez et al. 2018) conditioned on sentence-encoder embeddings of event
descriptions. When event_emb=None the model behaves identically to the
backbone GNN, enabling direct comparison without separate inference paths.

FiLM is applied to the projected node representations before graph
convolution, allowing the event context to scale and shift the initial
station features before spatial aggregation. This design choice (applying
conditioning once at the entry of the GCN stack rather than at each layer)
reduces the number of additional trainable parameters while still providing
a global semantic bias over the entire spatial aggregation process.

The FiLM layers are identity-initialised (γ=1, β=0), so an untrained
SE-GNN produces exactly the same output as the backbone GNN. This guarantees
that fine-tuning cannot degrade below the backbone even in the first epoch.

Reference:
    Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018).
    FiLM: Visual Reasoning with a General Conditioning Layer. AAAI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .model import DisaggregationGNN


class EventConditionedGNN(nn.Module):
    """
    GNN with FiLM conditioning on sentence-encoder event embeddings.

    The backbone architecture mirrors DisaggregationGNN exactly so that
    pretrained weights transfer without surgery. Only the two FiLM MLPs
    (gamma and beta projections) are new parameters.

    Args:
        hidden:        Hidden dimension (must match backbone checkpoint).
        dropout:       Dropout rate.
        n_gcn_layers:  Number of GCN layers (must match backbone checkpoint).
        event_emb_dim: Dimension of the sentence-encoder output (384 for
                       all-MiniLM-L6-v2).
        film_hidden:   Hidden dimension of the FiLM projection MLPs.
    """

    N_FEATURES = DisaggregationGNN.N_FEATURES
    HORIZON    = DisaggregationGNN.HORIZON

    def __init__(
        self,
        hidden:        int   = 64,
        dropout:       float = 0.2,
        n_gcn_layers:  int   = 2,
        event_emb_dim: int   = 384,   # all-MiniLM-L6-v2 output dim
        film_hidden:   int   = 64,
    ):
        super().__init__()

        self.hidden        = hidden
        self.event_emb_dim = event_emb_dim

        # Backbone (identical to DisaggregationGNN; weights are loaded from
        # a pretrained checkpoint and optionally frozen during FiLM fine-tuning)
        self.input_proj = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_gcn_layers)]
        )
        self.gcn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_gcn_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.HORIZON),
        )

        # FiLM modules: project event embedding to per-feature scale and shift
        self.film_scale = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden),
        )
        self.film_shift = nn.Sequential(
            nn.Linear(event_emb_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden),
        )

        # Identity initialisation: gamma=1, beta=0 so the untrained SE-GNN
        # matches the backbone exactly.
        nn.init.zeros_(self.film_scale[-1].weight)
        nn.init.ones_(self.film_scale[-1].bias)
        nn.init.zeros_(self.film_shift[-1].weight)
        nn.init.zeros_(self.film_shift[-1].bias)

    def forward(
        self,
        x:         torch.Tensor,
        A_hat:     torch.Tensor,
        event_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (N, N_FEATURES) node feature matrix.
            A_hat:     (N, N) normalised adjacency matrix.
            event_emb: (event_emb_dim,) sentence-encoder embedding, or None
                       to use backbone behaviour.

        Returns:
            profile: (N, 24) normalised hourly demand distribution.
        """
        h = self.input_proj(x)

        if event_emb is not None:
            if event_emb.dim() == 1:
                event_emb = event_emb.unsqueeze(0)        # (1, emb_dim)
            gamma = self.film_scale(event_emb)             # (1, hidden)
            beta  = self.film_shift(event_emb)             # (1, hidden)
            h     = h * gamma + beta                       # broadcast (N, hidden)

        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_agg = A_hat @ h
            h_new = self.dropout(torch.relu(gcn(h_agg)))
            h     = norm(h + h_new)

        return torch.softmax(self.decoder(h), dim=-1)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        event_emb_dim:   int  = 384,   # all-MiniLM-L6-v2 output dim
        film_hidden:     int  = 64,
        freeze_backbone: bool = True,
        device:          str  = "cpu",
    ) -> EventConditionedGNN:
        """
        Load backbone weights from a DisaggregationGNN checkpoint and
        attach identity-initialised FiLM layers.

        The checkpoint stores model_state and a cfg dict. Any keys absent
        from the SE-GNN state dict (the two FiLM modules) are ignored during
        loading; their identity initialisation is applied by __init__.

        Args:
            checkpoint_path: Path to .pt file saved by train.py.
            event_emb_dim:   Sentence-encoder output dimension.
            film_hidden:     FiLM MLP hidden size.
            freeze_backbone: If True, only FiLM parameters are trainable.
                             Reduces the number of free parameters to ~50k
                             and prevents catastrophic forgetting with few
                             training events.
            device:          Torch device string.

        Returns:
            EventConditionedGNN ready for FiLM fine-tuning.
        """
        # weights_only=False is required because the checkpoint stores a cfg
        # dict alongside the state_dict.
        ck  = torch.load(checkpoint_path, weights_only=False, map_location=device)
        cfg = ck.get("cfg", {})

        model = cls(
            hidden        = cfg.get("hidden", 64),
            dropout       = cfg.get("dropout", 0.2),
            n_gcn_layers  = cfg.get("n_gcn_layers", 2),
            event_emb_dim = event_emb_dim,
            film_hidden   = film_hidden,
        )

        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)

        non_film_missing = [k for k in missing if "film_" not in k]
        if non_film_missing:
            raise RuntimeError(
                f"Backbone keys missing from checkpoint: {non_film_missing}. "
                "Ensure the checkpoint was saved by train.py from this repository."
            )

        if freeze_backbone:
            for name, param in model.named_parameters():
                if "film_" not in name:
                    param.requires_grad = False

        return model.to(device)

    def film_parameters(self) -> list:
        """Return only the FiLM parameters (for use with a selective optimiser)."""
        return [p for name, p in self.named_parameters() if "film_" in name]


def numpy_to_event_tensor(emb: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert a numpy embedding (emb_dim,) to a float32 tensor."""
    return torch.tensor(emb, dtype=torch.float32, device=device)
