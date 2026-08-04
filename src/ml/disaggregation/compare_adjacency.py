"""
compare_adjacency.py - I3 (Reviewer 2.1) sensitivity analysis: geographic k-NN vs
track/line topology adjacency, trained under IDENTICAL procedure (same seed,
same train/val split, same epochs) so the comparison is apples-to-apples.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.disaggregation.model import DisaggregationGNN, build_adjacency
from ml.disaggregation.topology_adjacency import build_adjacency_topology
from ml.disaggregation.train import DailyProfileDataset

DATA_DIR = ROOT / "data" / "nyc" / "manhattan"
SEED = 42
EPOCHS = 4
LR = 1e-3
BATCH_SIZE = 32
EVENT_ALPHA = 0.4
MAX_TRAIN_DAYS = 64
MAX_VAL_DAYS = 32


def load_data():
    profiles = pd.read_parquet(DATA_DIR / "daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"]).dt.date
    labels = pd.read_csv(DATA_DIR / "event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    baseline = pd.read_parquet(DATA_DIR / "baseline_profile.parquet")
    stations = pd.read_csv(DATA_DIR / "station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)
    bdt_path = DATA_DIR / "baseline_daily_totals.parquet"
    baseline_daily = pd.read_parquet(bdt_path) if bdt_path.exists() else None
    return profiles, labels, baseline, stations, baseline_daily


def run_training(A_hat, label, profiles, labels, baseline, stations, baseline_daily):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    venue_coords = (40.8135, -74.0745)
    train_ds = DailyProfileDataset(profiles, labels, stations, baseline, venue_coords,
                                    years=[2022, 2023], baseline_daily_df=baseline_daily)
    val_ds = DailyProfileDataset(profiles, labels, stations, baseline, venue_coords,
                                  years=[2024], baseline_daily_df=baseline_daily)

    rng = np.random.RandomState(SEED)
    if len(train_ds.dates) > MAX_TRAIN_DAYS:
        keep = sorted(rng.choice(len(train_ds.dates), MAX_TRAIN_DAYS, replace=False))
        train_ds.dates = [train_ds.dates[i] for i in keep]
    if len(val_ds.dates) > MAX_VAL_DAYS:
        keep = sorted(rng.choice(len(val_ds.dates), MAX_VAL_DAYS, replace=False))
        val_ds.dates = [val_ds.dates[i] for i in keep]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DisaggregationGNN(hidden=64, dropout=0.2, n_gcn_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss(reduction="none")

    best_val = float("inf")
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch, is_event_batch in train_loader:
            preds = torch.stack([model(x_batch[b], A_hat) for b in range(x_batch.shape[0])])
            loss_elem = loss_fn(preds, y_batch)
            loss_per_day = loss_elem.mean(dim=(1, 2))
            weights = 1.0 + EVENT_ALPHA * is_event_batch
            loss = (loss_per_day * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_event_losses = [], []
        with torch.no_grad():
            for x_batch, y_batch, is_event_batch in val_loader:
                preds = torch.stack([model(x_batch[b], A_hat) for b in range(x_batch.shape[0])])
                loss_elem = loss_fn(preds, y_batch)
                val_losses.append(loss_elem.mean(dim=(1, 2)).mean().item())
                ev_mask = is_event_batch.bool()
                if ev_mask.any():
                    val_event_losses.append(loss_elem[ev_mask].mean().item())

        val_loss = float(np.mean(val_losses))
        val_ev = float(np.mean(val_event_losses)) if val_event_losses else float("nan")
        best_val = min(best_val, val_loss)
        log.append({"adjacency": label, "epoch": epoch, "train_loss": np.mean(train_losses),
                     "val_loss": val_loss, "val_event_loss": val_ev})
        print(f"  [{label}] epoch {epoch:2d}/{EPOCHS}  train={np.mean(train_losses):.5f}  "
              f"val={val_loss:.5f}  val_event={val_ev:.5f}")

    return pd.DataFrame(log), best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["geo", "topo", "both"], default="both")
    args = parser.parse_args()
    profiles, labels, baseline, stations, baseline_daily = load_data()

    print("Building geographic k-NN adjacency (k=8, current paper choice)...")
    A_geo = build_adjacency(stations, k=8)
    print("Building track-topology adjacency (GTFS-based)...")
    A_topo = build_adjacency_topology(stations)

    out_dir = ROOT / "outputs" / "paper" / "nyc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("geo", "both"):
        print("\n=== Training with GEOGRAPHIC k-NN adjacency ===")
        log_geo, best_geo = run_training(A_geo, "geographic_knn", profiles, labels, baseline, stations, baseline_daily)
        log_geo.to_csv(out_dir / "adjacency_sensitivity_geo.csv", index=False)
        print(f"  BEST geographic val_loss: {best_geo:.5f}")

    if args.which in ("topo", "both"):
        print("\n=== Training with TRACK-TOPOLOGY adjacency ===")
        log_topo, best_topo = run_training(A_topo, "track_topology", profiles, labels, baseline, stations, baseline_daily)
        log_topo.to_csv(out_dir / "adjacency_sensitivity_topo.csv", index=False)
        print(f"  BEST topology val_loss: {best_topo:.5f}")


if __name__ == "__main__":
    main()
