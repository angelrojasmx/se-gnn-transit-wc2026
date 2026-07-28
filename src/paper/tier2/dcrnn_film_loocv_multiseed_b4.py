"""
tier2/dcrnn_film_loocv_multiseed_b4.py -- B4/R3.3/R6.4 follow-up.

The response letter already reports (R3.3/R6.4) that DiffusionConvGNN (B4's
DCRNN-style competitive backbone, model_dcrnn.py) does NOT beat the production
GCN backbone on event-day MAE across 9 seeds -- that comparison used NEITHER
model with event conditioning. The remaining gap: reviewers asked for "at
least one DCRNN/STGCN-class model AND one event-conditioned method" -- i.e.
they may want to see FiLM conditioning demonstrated on the competitive
backbone too, not only on the model that wins the backbone comparison.

This script runs the EXACT SAME evaluation protocol used for the paper's
headline FiLM result (llm_augmented_gnn.py::run_loocv -- 3-fold Leave-One-Out
CV on the 3 real Copa America 2024 matches at MetLife, Penn Station corridor
MAE, 40 fine-tuning epochs, lr=1e-3, backbone frozen, only FiLM trained),
but with EventConditionedDiffusionConvGNN (model_dcrnn_v2.py) instead of
EventConditionedGNN, and with the ALREADY-TRAINED DiffusionConvGNN backbone
checkpoints (outputs/nyc/manhattan_model_dcrnn_seed{1..9}.pt, one per
backbone-training-seed) standing in for the single fixed manhattan_model.pt
used in production.

IMPORTANT ASYMMETRY, disclosed honestly: production's multi-seed LOO-CV
(leakage_ablation_loocv_allseeds.csv, 12 seeds, mean +64.4%) varies only the
FiLM fine-tuning seed against a SINGLE fixed backbone checkpoint. Here, each
of the 9 available DiffusionConvGNN checkpoints is itself a DIFFERENT
backbone-training run (no single "production" diffusion-conv checkpoint
exists), so this script uses ONE FiLM fine-tune (seeded identically to the
backbone) per backbone checkpoint. The n=9x3=27 observations therefore vary
across backbone-training-seed x fold, not FiLM-seed x fold -- a related but
not identical notion of "stability" from the production number. Both are
reported, not conflated.

Output: outputs/paper/nyc/dcrnn_film_loocv_multiseed_b4.csv
"""
from __future__ import annotations
import sys, warnings, time, random as pyrandom
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "paper"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.disaggregation.model import build_node_features
from ml.disaggregation.model_dcrnn import build_diffusion_matrices
from ml.disaggregation.model_dcrnn_v2 import EventConditionedDiffusionConvGNN
from ml.disaggregation.train import DailyProfileDataset
from ml.disaggregation.event_encoder import EventEncoder
from data.event_catalog import COPA_AMERICA_2024

DATA_DIR = ROOT / "data" / "nyc" / "manhattan"
OUT_NYC = ROOT / "outputs" / "paper" / "nyc"
OUT_NYC.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_NYC / "dcrnn_film_loocv_multiseed_b4.csv"
EMB_CACHE = OUT_NYC / "event_embeddings_cache.json"

METLIFE_COORDS = (40.8135, -74.0745)
PENN_STATION_SIDS = [164, 318, 607]
HERALD_SQ_SID = [319]
PENN_CORRIDOR = PENN_STATION_SIDS + HERALD_SQ_SID
HOUR_COLS = [f"h{h:02d}" for h in range(24)]

BACKBONE_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 42]  # all available manhattan_model_dcrnn_seed*.pt
N_EPOCHS = 40
LR = 1e-3
DEVICE = "cpu"


def load_dataset():
    profiles = pd.read_parquet(DATA_DIR / "daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"]).dt.strftime("%Y-%m-%d")
    labels = pd.read_csv(DATA_DIR / "event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    baseline = pd.read_parquet(DATA_DIR / "baseline_profile.parquet")
    stations = pd.read_csv(DATA_DIR / "station_lookup.csv")
    stations["sid"] = stations["sid"].astype(int)
    profiles["sid"] = profiles["sid"].astype(int)
    bdt_path = DATA_DIR / "baseline_daily_totals.parquet"
    baseline_daily = pd.read_parquet(bdt_path) if bdt_path.exists() else None

    dataset = DailyProfileDataset(profiles, labels, stations, baseline, METLIFE_COORDS,
                                  baseline_daily_df=baseline_daily)
    return dataset, baseline_daily


def build_samples(dataset, baseline_daily_df, encoder):
    sid_to_idx = {int(sid): i for i, sid in enumerate(dataset.stations["sid"])}
    penn_idxs = [sid_to_idx[s] for s in PENN_CORRIDOR if s in sid_to_idx]
    daily_totals_table = dataset.profiles[["date", "sid", "daily_total"]].copy()

    samples = []
    for ev in COPA_AMERICA_2024:
        date_str = ev["date"]
        if date_str not in dataset.dates:
            continue
        ref_totals = daily_totals_table[daily_totals_table["date"] == date_str]
        if ref_totals.empty:
            continue
        totals_series = ref_totals.set_index("sid")["daily_total"]
        x = build_node_features(
            date=date_str, stations_df=dataset.stations,
            baseline_df=dataset.baseline, daily_totals_series=totals_series,
            venue_coords=METLIFE_COORDS, event_flag=1,
            device="cpu", baseline_daily_df=baseline_daily_df,
        )
        day_profiles = dataset.profiles[dataset.profiles["date"] == date_str]
        penn_profiles_real = []
        for sid in PENN_CORRIDOR:
            if sid in sid_to_idx:
                row = day_profiles[day_profiles["sid"] == sid]
                if not row.empty:
                    penn_profiles_real.append(row[HOUR_COLS].values[0])
        if not penn_profiles_real:
            continue
        target = np.mean(penn_profiles_real, axis=0).astype(np.float32)
        t_sum = target.sum()
        if t_sum > 0:
            target = target / t_sum
        samples.append({
            "date": date_str, "match": ev["match"], "stage": ev.get("stage", ""),
            "x": torch.tensor(x.numpy(), dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "event_emb": torch.tensor(encoder.encode(ev["description"]), dtype=torch.float32),
        })
    return samples, penn_idxs


def run_one_backbone_seed(seed, samples, penn_idxs, P_f, P_b):
    ckpt_path = ROOT / f"outputs/nyc/manhattan_model_dcrnn_seed{seed}.pt"
    loss_fn = nn.MSELoss()
    rows = []
    for k in range(len(samples)):
        test_sample = samples[k]
        train_samples = [s for i, s in enumerate(samples) if i != k]

        torch.manual_seed(seed); np.random.seed(seed); pyrandom.seed(seed)
        fold_model = EventConditionedDiffusionConvGNN.from_pretrained(
            ckpt_path, event_emb_dim=384, film_hidden=64, freeze_backbone=True, device=DEVICE,
        )
        fold_model.train()
        optimizer = torch.optim.Adam(fold_model.film_parameters(), lr=LR)

        for epoch in range(N_EPOCHS):
            for s in train_samples:
                optimizer.zero_grad()
                profiles = fold_model(s["x"], P_f, P_b, event_emb=s["event_emb"])
                pred = profiles[penn_idxs].mean(dim=0)
                loss = loss_fn(pred, s["target"])
                loss.backward()
                optimizer.step()

        fold_model.eval()
        with torch.no_grad():
            ts = test_sample
            prof_base = fold_model(ts["x"], P_f, P_b, event_emb=None)[penn_idxs].mean(dim=0)
            prof_film = fold_model(ts["x"], P_f, P_b, event_emb=ts["event_emb"])[penn_idxs].mean(dim=0)
            tgt = ts["target"]
            mae_base = (prof_base - tgt).abs().mean().item()
            mae_film = (prof_film - tgt).abs().mean().item()

        rows.append({
            "backbone_seed": seed, "fold": k + 1, "held_out_match": test_sample["match"],
            "mae_baseline": mae_base, "mae_film": mae_film,
            "improvement_pct": (mae_base - mae_film) / (mae_base + 1e-8) * 100,
        })
    return rows


def main():
    t_start = time.time()
    print("=" * 70)
    print("B4/R3.3/R6.4 -- FiLM event-conditioning on DiffusionConvGNN backbone")
    print("Copa America 2024 3-fold LOO-CV, same protocol as production headline result")
    print("=" * 70)

    dataset, baseline_daily = load_dataset()
    stations = dataset.stations
    P_f, P_b = build_diffusion_matrices(stations, k=8)
    encoder = EventEncoder(cache_path=EMB_CACHE)
    samples, penn_idxs = build_samples(dataset, baseline_daily, encoder)
    print(f"Samples (Copa America matches found in dataset): {len(samples)}")
    for s in samples:
        print(f"  {s['date']}  {s['match']}")

    if OUT_CSV.exists():
        done_df = pd.read_csv(OUT_CSV)
        done_seeds = set(done_df["backbone_seed"].unique().tolist())
        all_rows = done_df.to_dict("records")
        print(f"Resuming: backbone seeds already done = {sorted(done_seeds)}")
    else:
        done_seeds = set()
        all_rows = []

    for seed in BACKBONE_SEEDS:
        if seed in done_seeds:
            continue
        if time.time() - t_start > 35:
            print(f"Time budget reached -- stopping, will resume. Remaining: "
                  f"{[s for s in BACKBONE_SEEDS if s not in done_seeds and s != seed]}")
            break
        t0 = time.time()
        rows = run_one_backbone_seed(seed, samples, penn_idxs, P_f, P_b)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        mean_imp = np.mean([r["improvement_pct"] for r in rows])
        print(f"  backbone_seed {seed}: mean improvement {mean_imp:+.1f}%  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    n_done = df["backbone_seed"].nunique() if len(df) else 0
    print(f"\n{'='*70}\nBackbone seeds completed: {n_done}/{len(BACKBONE_SEEDS)}\n{'='*70}")
    if n_done > 0:
        print(f"Mean improvement across all (backbone_seed x fold): "
              f"{df['improvement_pct'].mean():+.2f}% (std {df['improvement_pct'].std():.2f}, n={len(df)})")
        print(f"Range: [{df['improvement_pct'].min():+.1f}%, {df['improvement_pct'].max():+.1f}%]")
        neg = (df['improvement_pct'] < 0).sum()
        print(f"Folds where FiLM HURT vs backbone-only: {neg}/{len(df)}")
        if n_done >= 2:
            from scipy import stats
            per_seed_mean = df.groupby("backbone_seed")["improvement_pct"].mean()
            t, p = stats.ttest_1samp(per_seed_mean, 0.0)
            print(f"One-sample t-test (per-backbone-seed mean improvement vs 0): "
                  f"mean={per_seed_mean.mean():.2f}%, t={t:.3f}, p={p:.6f}, n_seeds={len(per_seed_mean)}")
    if n_done < len(BACKBONE_SEEDS):
        print(f"\nRe-run to continue ({len(BACKBONE_SEEDS)-n_done} backbone seeds remaining).")
    else:
        print("\nALL BACKBONE SEEDS DONE.")


if __name__ == "__main__":
    main()
