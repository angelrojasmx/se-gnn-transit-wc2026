"""
eval_dcrnn_vs_backbone.py -- computes event-day MAE and overall MAE for the
production DisaggregationGNN backbone (outputs/nyc/manhattan_model.pt, fixed,
single checkpoint) vs. the DiffusionConvGNN (DCRNN-style, B4) baseline for a
given training seed, on the same Manhattan 2024 validation set.

This is the permanent version of a comparison that was previously computed
inline (never saved as a script) for a single seed (42): event-day MAE
0.00829 (backbone) vs. 0.00986 (diffusion conv). This script lets that
comparison be re-run for many seeds of the diffusion-conv model against the
SAME fixed backbone checkpoint (the backbone itself is not re-seeded here --
only the diffusion-conv baseline's training-seed sensitivity is in question
for R3.3/R6.4).

Usage:
    python src/ml/disaggregation/eval_dcrnn_vs_backbone.py --seed 1
    python src/ml/disaggregation/eval_dcrnn_vs_backbone.py --seed 1 --skip-print-per-day
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ml.disaggregation.model import DisaggregationGNN, build_adjacency
from ml.disaggregation.train import DailyProfileDataset
from ml.disaggregation.model_dcrnn import DiffusionConvGNN, build_diffusion_matrices

BACKBONE_CKPT = ROOT / "outputs/nyc/manhattan_model.pt"
DATA_DIR = ROOT / "data/nyc/manhattan"


def load_val_dataset():
    profiles = pd.read_parquet(DATA_DIR / "daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"]).dt.date
    labels = pd.read_csv(DATA_DIR / "event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    baseline = pd.read_parquet(DATA_DIR / "baseline_profile.parquet")
    stations = pd.read_csv(DATA_DIR / "station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)
    bdt_path = DATA_DIR / "baseline_daily_totals.parquet"
    baseline_daily = pd.read_parquet(bdt_path) if bdt_path.exists() else None
    venue_coords = (40.8135, -74.0745)

    val_ds = DailyProfileDataset(profiles, labels, stations, baseline, venue_coords,
                                  years=[2024], baseline_daily_df=baseline_daily)
    return val_ds, stations


def load_backbone():
    ckpt = torch.load(BACKBONE_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = DisaggregationGNN(hidden=cfg["hidden"], dropout=cfg["dropout"],
                               n_gcn_layers=cfg["n_gcn_layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def load_dcrnn(seed):
    ckpt_path = ROOT / f"outputs/nyc/manhattan_model_dcrnn_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint for seed {seed} at {ckpt_path}. "
                                 f"Run train_dcrnn.py --seed {seed} first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DiffusionConvGNN(hidden=64, dropout=0.2, n_layers=2, K=2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main(seed):
    val_ds, stations = load_val_dataset()
    A_hat = build_adjacency(stations, k=8)
    P_f, P_b = build_diffusion_matrices(stations, k=8)

    backbone, backbone_cfg = load_backbone()
    dcrnn, dcrnn_ckpt = load_dcrnn(seed)

    rows = []
    for idx in range(len(val_ds)):
        x, target, is_event = val_ds[idx]
        date = val_ds.dates[idx]
        target_np = target.numpy()

        with torch.no_grad():
            pred_backbone = backbone(x, A_hat).numpy()
            pred_dcrnn = dcrnn(x, P_f, P_b).numpy()

        mae_backbone = np.abs(pred_backbone - target_np).mean()
        mae_dcrnn = np.abs(pred_dcrnn - target_np).mean()

        rows.append({
            "date": str(date),
            "is_event": int(is_event),
            "mae_backbone": mae_backbone,
            "mae_dcrnn": mae_dcrnn,
        })

    df = pd.DataFrame(rows)
    event_df = df[df["is_event"] == 1]
    normal_df = df[df["is_event"] == 0]

    summary = {
        "seed": seed,
        "n_days": len(df),
        "n_event_days": len(event_df),
        "overall_mae_backbone": df["mae_backbone"].mean(),
        "overall_mae_dcrnn": df["mae_dcrnn"].mean(),
        "event_mae_backbone": event_df["mae_backbone"].mean(),
        "event_mae_dcrnn": event_df["mae_dcrnn"].mean(),
        "dcrnn_epoch": dcrnn_ckpt.get("epoch"),
        "dcrnn_best_val_loss": dcrnn_ckpt.get("best_val_loss"),
    }

    print(f"Seed {seed}: overall MAE backbone={summary['overall_mae_backbone']:.5f} "
          f"dcrnn={summary['overall_mae_dcrnn']:.5f} | event-day MAE backbone="
          f"{summary['event_mae_backbone']:.5f} dcrnn={summary['event_mae_dcrnn']:.5f} "
          f"(n_event_days={summary['n_event_days']}, dcrnn trained {summary['dcrnn_epoch']} epochs)")

    out_dir = ROOT / "outputs/paper/nyc"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"dcrnn_vs_backbone_per_day_seed{seed}.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / f"dcrnn_vs_backbone_summary_seed{seed}.csv", index=False)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    main(args.seed)
