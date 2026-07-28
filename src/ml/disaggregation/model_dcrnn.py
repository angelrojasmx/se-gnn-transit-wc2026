"""
ml/disaggregation/model_dcrnn.py -- B4 competitive baseline (R3.3/R6.4)

Reviewers #3 and #6 asked for at least one DCRNN/STGCN-class competitive baseline,
since the paper positions itself against Noursalehi (2018), Santanam (2024), Wu (2025),
and Zhang (2025), all of which use spatio-temporal architectures of that family.

IMPORTANT FRAMING NOTE (disclosed honestly in the response letter, not hidden):
DCRNN (Li et al. 2018) and STGCN (Yu et al. 2018) are both designed for MULTI-STEP
forecasting from a ROLLING WINDOW of historical graph signals (e.g. 12 timesteps in,
12 out), using a recurrent (DCRNN) or temporal-gated-conv (STGCN) component along that
time axis. This paper's disaggregation task has NO such axis: each day is an
i.i.d. sample (daily total + static features -> 24-hour profile), with no cross-day
history fed to the model at all (confirmed in model.py / train.py: A_hat is static,
DailyProfileDataset returns one full-graph day per sample, sin/cos(day-of-week) is
the only "time" signal). A literal reproduction of DCRNN/STGCN would require
inventing a sliding-window historical-sequence task that does not exist in the data
as currently used -- a materially different framing, not just a model swap.

We instead implement the architectural contribution that is actually comparable
apples-to-apples: DCRNN's core SPATIAL operator, bidirectional diffusion convolution
over a K-hop random walk on the station graph, as a drop-in replacement for the
existing 2-layer symmetric-normalized GCN, on the IDENTICAL single-step task, data
split, training loop, and loss as the production backbone. This isolates the
question "does a more expressive spatial-diffusion operator help this task" while
being transparent that the recurrent/sequential half of DCRNN does not apply here.

Diffusion convolution (Li et al. 2018, Eq. 2-3):
    Z = sum_{k=0}^{K} ( P_f^k @ H @ W_{f,k} + P_b^k @ H @ W_{b,k} )
  where P_f = D_out^-1 A       (forward random-walk transition matrix)
        P_b = D_in^-1  A^T     (backward random-walk transition matrix)
  K is the number of diffusion steps (K=2 here, matching typical DCRNN configs).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import BallTree


def build_diffusion_matrices(stations_df, k=8):
    """Build forward/backward random-walk transition matrices from the SAME k-NN
    geographic graph used by build_adjacency() in model.py, for a fair comparison
    (same graph, different propagation operator)."""
    N = len(stations_df)
    coords = stations_df[["latitude", "longitude"]].to_numpy()
    coords_rad = np.deg2rad(coords)

    k = min(k, N - 1)
    tree = BallTree(coords_rad, metric="haversine")
    _, ind = tree.query(coords_rad, k=k + 1)

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in ind[i]:
            if i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0
    A = A + np.eye(N, dtype=np.float32)  # self-loops, same as build_adjacency

    out_deg = A.sum(axis=1, keepdims=True)
    in_deg = A.sum(axis=0, keepdims=True)
    P_f = A / (out_deg + 1e-8)             # D_out^-1 A
    P_b = A.T / (in_deg.T + 1e-8)          # D_in^-1 A^T

    return torch.tensor(P_f, dtype=torch.float32), torch.tensor(P_b, dtype=torch.float32)


class DiffusionConvLayer(nn.Module):
    """One DCRNN-style bidirectional diffusion convolution layer (no recurrence,
    since the task has no time axis -- see module docstring)."""

    def __init__(self, hidden, K=2):
        super().__init__()
        self.K = K
        # separate learnable weight per hop, per direction (as in DCRNN's DCGRU spatial filter)
        self.W_f = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(K + 1)])
        self.W_b = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(K + 1)])
        self.bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, h, P_f, P_b):
        out = self.W_f[0](h) + self.W_b[0](h)  # k=0 hop is just self (P^0 = I)
        h_f, h_b = h, h
        for k in range(1, self.K + 1):
            h_f = P_f @ h_f
            h_b = P_b @ h_b
            out = out + self.W_f[k](h_f) + self.W_b[k](h_b)
        return out + self.bias


class DiffusionConvGNN(nn.Module):
    """Same overall architecture as DisaggregationGNN (model.py) -- input_proj,
    N graph layers with residual+LayerNorm, decoder+softmax -- with the GCN layers
    replaced by DiffusionConvLayer. Same N_FEATURES=7, same HORIZON=24, so it is a
    drop-in swap trained with the IDENTICAL train.py loop (only the model class and
    the propagation matrices change)."""

    N_FEATURES = 7
    HORIZON = 24

    def __init__(self, hidden: int = 64, dropout: float = 0.2, n_layers: int = 2, K: int = 2):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.K = K

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

    def forward(self, x, P_f, P_b):
        h = self.input_proj(x)
        for layer, norm in zip(self.diff_layers, self.norms):
            h_new = torch.relu(layer(h, P_f, P_b))
            h_new = self.dropout(h_new)
            h = norm(h + h_new)
        logits = self.decoder(h)
        return torch.softmax(logits, dim=-1)
