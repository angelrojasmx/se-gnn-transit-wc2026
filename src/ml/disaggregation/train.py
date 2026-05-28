"""
train.py — Backbone DisaggregationGNN training
===============================================
Trains the temporal disaggregation backbone on Manhattan ridership data
(2022-2023) and validates on the 2023-2024 held-out year.

Usage:
    python src/ml/disaggregation/train.py

Requires:
    data/nyc/manhattan/daily_profiles.parquet
    data/nyc/manhattan/event_labels.csv
    data/nyc/manhattan/baseline_profile.parquet
    data/nyc/manhattan/station_lookup.csv

Outputs:
    outputs/nyc/manhattan_model.pt       — best checkpoint (val loss)
    outputs/nyc/manhattan_train_log.csv  — per-epoch loss history
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ml.disaggregation.model import DisaggregationGNN, build_adjacency, build_node_features

# Training configuration
CFG = {
    # Data — source domain: Manhattan only.
    # Bronx data is reserved for the transfer evaluation in evaluate_transfer.py.
    "data_dir":    str(ROOT / "data/nyc/manhattan"),
    "train_years": [2022, 2023],
    "val_year":    2024,      # intra-Manhattan validation for training monitoring

    # Model
    "hidden":       64,
    "dropout":      0.2,
    "n_gcn_layers": 2,
    "k_neighbors":  8,

    # Training
    "epochs":       60,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   32,
    "event_alpha":  0.4,   # extra loss weight for event days
    "patience":     8,     # early-stopping patience

    # Source venue: MetLife Stadium (Penn corridor / NFL events)
    "venue_coords": (40.8135, -74.0745),

    # Output paths
    "checkpoint": str(ROOT / "outputs/nyc/manhattan_model.pt"),
    "train_log":  str(ROOT / "outputs/nyc/manhattan_train_log.csv"),
}



class DailyProfileDataset(Dataset):
    """
    Dataset where each sample is one full day across all stations.

    Returns:
        x        : (N, 7) node feature matrix
        target   : (N, 24) normalised hourly profile (model target)
        is_event : scalar int (1 = event day in Penn corridor)
    """

    def __init__(self, profiles_df, labels_df, stations_df, baseline_df,
                 venue_coords, years=None, device="cpu", baseline_daily_df=None):
        self.stations  = stations_df.reset_index(drop=True)
        self.N         = len(self.stations)
        self.venue_coords = venue_coords
        self.device    = device
        self.baseline_daily_df = baseline_daily_df  # (sid, dow, median_daily_total)

        hour_cols = [f"h{h:02d}" for h in range(24)]

        # Filter by year if requested
        df = profiles_df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        if years:
            df = df[df["year"].isin(years)]

        # Index of unique dates in this split
        self.dates    = sorted(df["date"].unique())
        self.profiles = df  # (date, sid, h00..h23, daily_total, is_event)
        self.labels   = labels_df.set_index("date")
        self.baseline = baseline_df  # (sid, dow, hour, mean_profile)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date     = self.dates[idx]
        day_data = self.profiles[self.profiles["date"] == date]

        # Target: (N, 24) observed hourly profile
        hour_cols  = [f"h{h:02d}" for h in range(24)]
        sid_to_row = day_data.set_index("sid")
        target       = np.zeros((self.N, 24), dtype=np.float32)
        daily_totals = pd.Series(dtype=float)

        for i, sid in enumerate(self.stations["sid"].values):
            if sid in sid_to_row.index:
                target[i]       = sid_to_row.loc[sid, hour_cols].values.astype(np.float32)
                daily_totals[sid] = sid_to_row.loc[sid, "daily_total"]
            else:
                target[i] = 1.0 / 24.0   # uniform fallback for missing stations

        is_event = int(self.labels.loc[date, "is_event"]) if date in self.labels.index else 0

        # Build node feature matrix
        x = build_node_features(
            date              = date,
            stations_df       = self.stations,
            baseline_df       = self.baseline,
            daily_totals_series = daily_totals,
            venue_coords      = self.venue_coords,
            event_flag        = is_event,
            device            = self.device,
            baseline_daily_df = self.baseline_daily_df,
        )

        return (
            x,                                                    # (N, 7)
            torch.tensor(target, dtype=torch.float32),           # (N, 24)
            torch.tensor(is_event, dtype=torch.float32),         # scalar
        )


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data — source domain: Manhattan only
    data_dir = Path(CFG["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run first: python src/demand/build_dataset.py --source manhattan"
        )
    print(f"Loading Manhattan data from {data_dir}...")
    profiles  = pd.read_parquet(data_dir / "daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"]).dt.date
    labels    = pd.read_csv(data_dir / "event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    baseline  = pd.read_parquet(data_dir / "baseline_profile.parquet")
    stations  = pd.read_csv(data_dir / "station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)

    # Daily baseline totals for Feature 6 (today-vs-median ratio by sid x DOW)
    bdt_path = data_dir / "baseline_daily_totals.parquet"
    baseline_daily = pd.read_parquet(bdt_path) if bdt_path.exists() else None
    if baseline_daily is not None:
        print(f"  {len(stations)} Manhattan stations  (Feature 6: baseline_daily loaded)")
    else:
        print(f"  {len(stations)} Manhattan stations  (WARNING: baseline_daily not found)")

    # Build graph adjacency
    print("Building adjacency matrix...")
    A_hat = build_adjacency(stations, k=CFG["k_neighbors"]).to(device)

    # Datasets
    train_ds = DailyProfileDataset(
        profiles, labels, stations, baseline,
        CFG["venue_coords"], years=CFG["train_years"], device=device,
        baseline_daily_df=baseline_daily,
    )
    val_ds = DailyProfileDataset(
        profiles, labels, stations, baseline,
        CFG["venue_coords"], years=[CFG["val_year"]], device=device,
        baseline_daily_df=baseline_daily,
    )

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False)
    print(f"Train: {len(train_ds)} days | Val: {len(val_ds)} days")

    # Model and optimiser
    model = DisaggregationGNN(
        hidden=CFG["hidden"],
        dropout=CFG["dropout"],
        n_gcn_layers=CFG["n_gcn_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    loss_fn = nn.MSELoss(reduction="none")

    ckpt_path = Path(CFG["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    log_rows         = []

    print("\nTraining...")
    for epoch in range(1, CFG["epochs"] + 1):

        # Training pass
        model.train()
        train_losses = []

        for x_batch, y_batch, is_event_batch in train_loader:
            x_batch        = x_batch.to(device)        # (B, N, 7)
            y_batch        = y_batch.to(device)        # (B, N, 24)
            is_event_batch = is_event_batch.to(device)  # (B,)

            # Forward: process each day in the batch independently
            preds = torch.stack([
                model(x_batch[b], A_hat) for b in range(x_batch.shape[0])
            ])  # (B, N, 24)

            # Weighted MSE: event days get extra weight via event_alpha
            loss_elem  = loss_fn(preds, y_batch)           # (B, N, 24)
            loss_per_day = loss_elem.mean(dim=(1, 2))      # (B,)
            weights    = 1.0 + CFG["event_alpha"] * is_event_batch
            loss       = (loss_per_day * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation pass
        model.eval()
        val_losses, val_event_losses = [], []

        with torch.no_grad():
            for x_batch, y_batch, is_event_batch in val_loader:
                x_batch        = x_batch.to(device)
                y_batch        = y_batch.to(device)
                is_event_batch = is_event_batch.to(device)

                preds = torch.stack([
                    model(x_batch[b], A_hat) for b in range(x_batch.shape[0])
                ])

                loss_elem    = loss_fn(preds, y_batch)
                loss_per_day = loss_elem.mean(dim=(1, 2))
                val_losses.append(loss_per_day.mean().item())

                # Track event-day loss separately
                ev_mask = is_event_batch.bool()
                if ev_mask.any():
                    val_event_losses.append(loss_elem[ev_mask].mean().item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_ev     = np.mean(val_event_losses) if val_event_losses else float("nan")

        scheduler.step(val_loss)

        # Save checkpoint on improvement
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "best_val_loss":  float(best_val_loss),
                "cfg":            CFG,
            }, ckpt_path)
        else:
            patience_counter += 1

        log_rows.append({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "val_event_loss": val_ev,
            "best_val_loss":  best_val_loss,
        })

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"val_event={val_ev:.5f} | best={best_val_loss:.5f}"
            )

        if patience_counter >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(CFG["train_log"], index=False)
    print(f"\nBest val_loss: {best_val_loss:.5f}")
    print(f"Checkpoint -> {ckpt_path}")
    print(f"Log        -> {CFG['train_log']}")


if __name__ == "__main__":
    train()
