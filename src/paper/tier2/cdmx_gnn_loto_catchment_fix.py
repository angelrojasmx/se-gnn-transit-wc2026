"""
cdmx_gnn_loto_catchment_fix.py

FORK of cdmx_gnn_loto.py - does NOT modify the original (fork-don't-edit
discipline; production script and its cached outputs/tables stay untouched).

Purpose
-------
Addresses ESWA reviewer R2.4: the `is_venue_line` feature (F[2] in
build_node_features) is a coarse binary that flags EVERY station on the
venue's Metro line, regardless of distance to the venue. This leaks a
correlated-but-spatially-irrelevant signal for far-away same-line stations.
Documented leaky stations (response_to_reviewers_ESWA_R1.tex, R2.4):
Tlaltenco (+120% to +217%), Hospital 20 de Noviembre, Lomas Estrella,
San Andres Tomatlan, Parque de los Venados (+30% to +55%), and off-line
noise at Santa Anita (Linea 4) / Deportivo 18 de Marzo (Linea 6) (+60% to +95%).

This fork reimplements build_node_features() with a swappable F[2]:

  - "baseline"       : original coarse same-line indicator (re-run here,
                        multi-seed, for a fair paired comparison - the
                        production number in the paper is single-seed).
  - "catchment"       : binary, 1 only if station name in VENUE_CATCHMENT[venue].
  - "distance_decay"  : same_line * exp(-dist_km / DECAY_KM), DECAY_KM=3.0.
                        (3.0 km chosen as a rough walking-catchment scale;
                        not tuned - documented as a design choice, not fit.)

Everything else (GNN architecture, two-phase LOTO-CV protocol, venue-aware
MAE, historical-average baseline) is imported unchanged from cdmx_gnn_loto.

Reproducibility / resumability design
--------------------------------------
The original run_loto_cv() seeds RNG ONCE per `run()` call and lets folds
share one continuous RNG stream (order-dependent). To make this experiment
resumable in small chunks (needed for sandboxed/time-boxed execution), each
(variant, seed, fold) job here is independently seeded with `set_seed(seed)`
right before that fold's model init - a deliberate deviation from the
original's exact RNG behavior, but standard/better practice for a controlled
multi-seed comparison, and necessary so jobs can run in any order/batch size
and still be reproducible. Documented here for transparency.

Usage
-----
    # one-time per variant: build + cache node-feature samples (~15s each)
    python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --prepare

    # run LOTO-CV jobs until wall-clock budget exhausted, resumable
    python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --train --budget_s 260

    # after all (variant,seed,fold) jobs done: aggregate + print summary
    python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --summarize

    # WC2026 leaky-station check per variant (uses seed=42 only, single
    # forward pass per variant - this is a diagnostic re-check, not a stat test)
    python -m src.paper.tier2.cdmx_gnn_loto_catchment_fix --wc2026check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.paper.tier2 import cdmx_gnn_loto as base
from src.ml.disaggregation.event_encoder import EventEncoder

# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR       = base.OUT_CDMX / "catchment_fix"
CACHE_DIR     = OUT_DIR / "sample_cache"
RESULTS_PATH  = OUT_DIR / "loto_results_all_variants.csv"
WC2026_PATH   = OUT_DIR / "wc2026_leaky_stations_check.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DECAY_KM   = 3.0
VARIANTS   = ["baseline", "catchment", "distance_decay"]
SEEDS      = [42, 43, 44]
# Extended to 12 seeds total (matching the 12-seed bar used elsewhere in this
# revision, e.g. the leakage ablation / NFL-augmentation tests) for baseline
# and distance_decay only. catchment already conclusively rejected on
# accuracy grounds at n=3 (paired t=-13.9, p=0.005 vs baseline) - not worth
# the extra ~18min compute to extend it too.
EXTENDED_SEEDS       = [45, 46, 47, 48, 49, 50, 51, 52, 53]
EXTENDED_SEED_VARIANTS = ["baseline", "distance_decay"]
N_EPOCHS   = 50   # matches production CLI default (argparse --epochs default=50),
                   # NOT run()'s own function-signature default of 100.
                   # Verified: __main__ invokes run(n_epochs=args.epochs) with
                   # args.epochs defaulting to 50, so the cited production
                   # table (Hist 0.00106 / Backbone 0.00138 / SE-GNN 0.00112,
                   # +20.0%, cosine 0.972) was produced at 50 epochs/phase.

KNOWN_LEAKY_STATIONS = [
    "tlaltenco", "hospital 20 de noviembre", "lomas estrella",
    "san andres tomatlan", "parque de los venados",
    "santa anita", "deportivo 18 de marzo",
]

VENUE_LINES = {"Ciudad Deportiva": "linea 9", "Azteca": "linea 12", "Pumas/CU": "linea 3"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature variant logic
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features_variant(
    date_str: str,
    stations_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    venue_name: str,
    feature_variant: str,
    venue_coords=None,
    proj_total: float | None = None,
) -> torch.Tensor:
    """Same (N,6) feature layout as base.build_node_features; only F[2] (is_venue_line)
    changes based on `feature_variant`. F[0],F[1],F[3],F[4],F[5] identical to original."""
    fecha = pd.Timestamp(date_str)
    dow   = fecha.dayofweek
    venue = venue_coords if venue_coords else base.VENUE_COORDS.get(venue_name, (19.42, -99.14))

    N = len(stations_df)
    X = np.zeros((N, 6), dtype=np.float32)
    base_idx = baseline_df.set_index(['sid', 'dow'])['baseline_riders']
    v_line = VENUE_LINES.get(venue_name, "")
    catchment_names = set(base.VENUE_CATCHMENT.get(venue_name, []))

    baseline_vals = np.zeros(N)
    dist_vals     = np.zeros(N)
    feat_vals     = np.zeros(N)

    for i, row in stations_df.reset_index(drop=True).iterrows():
        sid = row['sid']
        b   = float(base_idx.get((sid, dow), 0.0))
        baseline_vals[i] = b
        d = base.haversine_km(row['latitude'], row['longitude'], venue[0], venue[1])
        dist_vals[i] = d
        same_line = 1.0 if row['line'] == v_line else 0.0

        if feature_variant == "baseline":
            feat_vals[i] = same_line
        elif feature_variant == "catchment":
            feat_vals[i] = 1.0 if row['name'] in catchment_names else 0.0
        elif feature_variant == "distance_decay":
            feat_vals[i] = same_line * float(np.exp(-d / DECAY_KM))
        else:
            raise ValueError(f"Unknown feature_variant: {feature_variant}")

    baseline_total = baseline_vals.sum()
    if proj_total is None:
        today = daily_df[daily_df['date'] == date_str]
        proj_total = float(today['riders'].sum()) if not today.empty else baseline_total

    scale_factor = proj_total / max(baseline_total, 1.0)
    scaled_vals  = baseline_vals * scale_factor

    b_max = baseline_vals.max() + 1e-8
    s_max = scaled_vals.max() + 1e-8

    X[:, 0] = np.log1p(baseline_vals) / np.log1p(b_max)
    X[:, 1] = np.log1p(dist_vals)
    X[:, 2] = feat_vals
    X[:, 3] = np.sin(2 * np.pi * dow / 7)
    X[:, 4] = np.cos(2 * np.pi * dow / 7)
    X[:, 5] = np.log1p(scaled_vals) / np.log1p(s_max)

    return torch.tensor(X, dtype=torch.float32)


def prepare_samples_variant(events, stations_df, daily_df, baseline_df, A_hat, encoder, feature_variant):
    samples = []
    skipped = 0
    for _, ev in events.iterrows():
        date_str = ev['date']
        ev_cat   = ev['event_category']
        catalog_venue = ev.get('venue', None)
        if pd.notna(catalog_venue) and str(catalog_venue) in base.VENUE_CATCHMENT:
            venue_name = str(catalog_venue)
        else:
            venue_name = base.TYPE_TO_VENUE.get(ev_cat, "Ciudad Deportiva")

        target = base.get_observed_spatial_distribution(date_str, stations_df, daily_df, baseline_df)
        if target is None:
            skipped += 1
            continue

        x = build_node_features_variant(
            date_str, stations_df, baseline_df, daily_df, venue_name, feature_variant,
        )
        desc      = base.CDMX_EVENT_DESCRIPTIONS.get(ev_cat, base.CDMX_EVENT_DESCRIPTIONS["concert"])
        event_emb = torch.tensor(encoder.encode(desc), dtype=torch.float32)

        samples.append({
            "date": date_str, "venue": venue_name, "event_cat": ev_cat,
            "event_name": ev.get('event_name', ''), "x": x, "A_hat": A_hat,
            "target": target, "event_emb": event_emb,
        })
    if skipped:
        print(f"  [{feature_variant}] skipped {skipped} events (insufficient data)")
    return samples


def get_or_build_samples(variant, stations_df, daily_df, baseline_df, catalog_df, A_hat, encoder):
    cache_path = CACHE_DIR / f"samples_{variant}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    print(f"Building sample cache for variant='{variant}'...")
    samples = prepare_samples_variant(catalog_df, stations_df, daily_df, baseline_df, A_hat, encoder, variant)
    torch.save(samples, cache_path)
    print(f"  cached {len(samples)} samples -> {cache_path}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Manifest / resumability
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_jobs() -> set[tuple[str, int, int]]:
    if not RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(RESULTS_PATH)
    return set(df[["variant", "seed", "fold"]].drop_duplicates().itertuples(index=False, name=None))


def append_results(rows: list[dict]):
    df = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# One fold, one seed, one variant
# ─────────────────────────────────────────────────────────────────────────────

def run_one_fold(variant, seed, fold_idx, test_type, all_samples, stations_df, daily_df, baseline_df,
                  n_epochs=N_EPOCHS, lr=1e-3, hidden=64, n_gcn_layers=2, device="cpu"):
    base.set_seed(seed)

    train_samples = [s for s in all_samples if s["event_cat"] != test_type]
    test_samples  = [s for s in all_samples if s["event_cat"] == test_type]
    if not train_samples or not test_samples:
        return []

    fold_model = base.SpatialDisaggregationGNN(
        hidden=hidden, dropout=0.2, n_gcn_layers=n_gcn_layers, event_emb_dim=384, film_hidden=64,
    ).to(device)

    optimizer_backbone = torch.optim.Adam(fold_model.backbone_parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    fold_model.train()
    for epoch in range(n_epochs):
        for s in train_samples:
            optimizer_backbone.zero_grad()
            pred = fold_model(s["x"].to(device), s["A_hat"].to(device), event_emb=None)
            loss = loss_fn(pred, s["target"].to(device))
            loss.backward()
            optimizer_backbone.step()

    base_model = __import__("copy").deepcopy(fold_model)
    base_model.eval()

    for p in fold_model.backbone_parameters():
        p.requires_grad_(False)
    fold_model = base.train_film(fold_model, train_samples, n_epochs, lr, device)
    fold_model.eval()

    test_rows = []
    with torch.no_grad():
        for s in test_samples:
            target = s["target"]
            venue  = s["venue"]

            hist_avg = base.historical_avg_baseline(s["date"], stations_df, daily_df, baseline_df)
            mae_hist = base.venue_aware_mae(hist_avg, target, venue, stations_df)

            pred_base = base_model(s["x"], s["A_hat"], event_emb=None)
            mae_base  = base.venue_aware_mae(pred_base, target, venue, stations_df)
            cos_base  = base.cosine_similarity(pred_base, target)

            pred_llm  = fold_model(s["x"], s["A_hat"], s["event_emb"])
            mae_llm   = base.venue_aware_mae(pred_llm, target, venue, stations_df)
            cos_llm   = base.cosine_similarity(pred_llm, target)

            test_rows.append({
                "variant": variant, "seed": seed, "fold": fold_idx + 1, "test_type": test_type,
                "date": s["date"], "event_name": s["event_name"], "venue": venue,
                "mae_hist_avg": mae_hist, "mae_baseline": mae_base, "mae_llm_oos": mae_llm,
                "cosine_baseline": cos_base, "cosine_llm": cos_llm,
                "improvement_pct": (mae_base - mae_llm) / (mae_base + 1e-8) * 100,
            })
    return test_rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI phases
# ─────────────────────────────────────────────────────────────────────────────

def phase_prepare():
    stations_df, daily_df, baseline_df, catalog_df, A_hat = base.load_data()
    encoder = EventEncoder(cache_path=base.EMB_CACHE)
    for variant in VARIANTS:
        get_or_build_samples(variant, stations_df, daily_df, baseline_df, catalog_df, A_hat, encoder)
    print("All variant sample caches ready.")


def phase_train(budget_s: float):
    t_start = time.time()
    stations_df, daily_df, baseline_df, catalog_df, A_hat = base.load_data()
    encoder = EventEncoder(cache_path=base.EMB_CACHE)

    event_types = catalog_df['event_category'].unique().tolist()
    completed = load_completed_jobs()
    print(f"Already completed: {len(completed)} (variant,seed,fold) jobs")

    all_jobs = [
        (variant, seed, fold_idx, test_type)
        for variant in VARIANTS
        for seed in SEEDS
        for fold_idx, test_type in enumerate(event_types)
    ] + [
        (variant, seed, fold_idx, test_type)
        for variant in EXTENDED_SEED_VARIANTS
        for seed in EXTENDED_SEEDS
        for fold_idx, test_type in enumerate(event_types)
    ]
    remaining = [j for j in all_jobs if (j[0], j[1], j[2] + 1) not in completed]
    print(f"Remaining jobs: {len(remaining)} / {len(all_jobs)}")

    samples_cache = {}
    n_done_this_call = 0
    for variant, seed, fold_idx, test_type in remaining:
        if time.time() - t_start > budget_s:
            print(f"Budget ({budget_s}s) reached, stopping. Did {n_done_this_call} jobs this call.")
            break
        if variant not in samples_cache:
            samples_cache[variant] = get_or_build_samples(
                variant, stations_df, daily_df, baseline_df, catalog_df, A_hat, encoder
            )
        all_samples = samples_cache[variant]

        t0 = time.time()
        rows = run_one_fold(variant, seed, fold_idx, test_type, all_samples, stations_df, daily_df, baseline_df)
        dt = time.time() - t0
        if rows:
            append_results(rows)
        n_done_this_call += 1
        print(f"  done: variant={variant} seed={seed} fold={fold_idx+1} ({test_type}) "
              f"n_test={len(rows)} took={dt:.1f}s elapsed={time.time()-t_start:.1f}s")

    n_left = len(remaining) - n_done_this_call
    print(f"\nThis call: {n_done_this_call} jobs done. {n_left} jobs still remaining overall.")
    if n_left == 0:
        print("ALL JOBS COMPLETE.")


def phase_summarize():
    if not RESULTS_PATH.exists():
        print("No results yet.")
        return
    df = pd.read_csv(RESULTS_PATH)
    print(f"Total rows: {len(df)}  |  jobs: {df[['variant','seed','fold']].drop_duplicates().shape[0]}")

    per_seed = df.groupby(["variant", "seed"]).agg(
        mae_base=("mae_baseline", "mean"),
        mae_llm=("mae_llm_oos", "mean"),
        cos_llm=("cosine_llm", "mean"),
        improvement=("improvement_pct", "mean"),
    ).reset_index()
    print("\n=== Per (variant, seed) means ===")
    print(per_seed.to_string(index=False))

    summary = per_seed.groupby("variant").agg(
        n_seeds=("seed", "count"),
        mae_base_mean=("mae_base", "mean"), mae_base_std=("mae_base", "std"),
        mae_llm_mean=("mae_llm", "mean"), mae_llm_std=("mae_llm", "std"),
        cos_llm_mean=("cos_llm", "mean"),
        improvement_mean=("improvement", "mean"), improvement_std=("improvement", "std"),
    ).reset_index()
    print("\n=== Variant summary (mean +/- std across seeds) ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)

    from scipy import stats
    base_by_seed = per_seed[per_seed.variant == "baseline"].set_index("seed")["mae_llm"]
    for v in ["catchment", "distance_decay"]:
        v_by_seed = per_seed[per_seed.variant == v].set_index("seed")["mae_llm"]
        common_seeds = sorted(set(base_by_seed.index) & set(v_by_seed.index))
        if len(common_seeds) > 1:
            a = base_by_seed.loc[common_seeds].values
            b = v_by_seed.loc[common_seeds].values
            t, p = stats.ttest_rel(a, b)
            print(f"\nPaired t-test mae_llm_oos: baseline vs {v}: t={t:.3f}, p={p:.4f} (n={len(common_seeds)} seeds: {common_seeds})")


def phase_wc2026check(only_variant: str | None = None):
    """Diagnostic: for each variant, run WC2026 inference (single seed=42,
    full-catalog training) and report uplift_ratio_llm for the known leaky
    off-catchment/same-line stations, to check whether the fix reduces
    their inflated uplift relative to the true Azteca catchment stations.
    Chunked per-variant (via only_variant) + append-to-CSV so it fits a
    time-boxed sandbox call; re-invoke once per variant."""
    from src.paper.config import WC2026_CDMX

    base.set_seed(42)
    stations_df, daily_df, baseline_df, catalog_df, A_hat = base.load_data()
    encoder = EventEncoder(cache_path=base.EMB_CACHE)

    variants_to_run = [only_variant] if only_variant else VARIANTS
    all_rows = []
    for variant in variants_to_run:
        print(f"\n=== WC2026 check: variant={variant} ===")
        base.set_seed(42)
        all_samples = get_or_build_samples(variant, stations_df, daily_df, baseline_df, catalog_df, A_hat, encoder)

        model = base.SpatialDisaggregationGNN(hidden=64, dropout=0.2, n_gcn_layers=2, event_emb_dim=384).to("cpu")
        loss_fn = nn.MSELoss()
        optimizer_backbone = torch.optim.Adam(model.backbone_parameters(), lr=1e-3)
        model.train()
        for epoch in range(N_EPOCHS):
            for s in all_samples:
                optimizer_backbone.zero_grad()
                pred = model(s["x"], s["A_hat"], event_emb=None)
                loss = loss_fn(pred, s["target"])
                loss.backward()
                optimizer_backbone.step()
        for p in model.backbone_parameters():
            p.requires_grad_(False)
        model = base.train_film(model, all_samples, N_EPOCHS, 1e-3, "cpu")
        model.eval()

        wc_desc = base.CDMX_EVENT_DESCRIPTIONS["wc2026_azteca"]
        wc_emb  = torch.tensor(encoder.encode(wc_desc), dtype=torch.float32)

        base_idx_all = baseline_df.set_index(['sid', 'dow'])['baseline_riders']
        AZTECA_BASELINE_SUNDAY = 120_000
        WC_MULTIPLIER = 3.0
        WC_EXTRA = AZTECA_BASELINE_SUNDAY * (WC_MULTIPLIER - 1.0)

        with torch.no_grad():
            for (fecha_str, kickoff, match, fase) in WC2026_CDMX:
                fecha = pd.Timestamp(fecha_str)
                dow = fecha.dayofweek
                baseline_total = float(sum(base_idx_all.get((sid, dow), 0.0) for sid in stations_df['sid'].values))
                proj_total = baseline_total + WC_EXTRA

                x = build_node_features_variant(
                    fecha_str, stations_df, baseline_df, daily_df, "Azteca", variant, proj_total=proj_total,
                )
                weights_llm = model(x, A_hat, wc_emb)
                total_riders_llm = weights_llm.numpy() * proj_total

                for i, row in stations_df.reset_index(drop=True).iterrows():
                    sid = row['sid']
                    base_i = float(base_idx_all.get((sid, dow), 0.0))
                    total_l = float(total_riders_llm[i])
                    name_lower = str(row['name']).lower()
                    if name_lower in KNOWN_LEAKY_STATIONS or name_lower in [n.lower() for n in base.VENUE_CATCHMENT["Azteca"]]:
                        all_rows.append({
                            "variant": variant, "fecha": fecha_str, "match": match,
                            "station": row['name'], "line": row['line'],
                            "is_true_catchment": name_lower in [n.lower() for n in base.VENUE_CATCHMENT["Azteca"]],
                            "baseline_riders": base_i, "total_riders_llm": total_l,
                            "uplift_ratio_llm": total_l / max(base_i, 1.0),
                        })

    wdf = pd.DataFrame(all_rows)
    if WC2026_PATH.exists():
        existing = pd.read_csv(WC2026_PATH)
        existing = existing[~existing["variant"].isin(variants_to_run)]
        wdf = pd.concat([existing, wdf], ignore_index=True)
    wdf.to_csv(WC2026_PATH, index=False)
    print(f"\nSaved -> {WC2026_PATH}")
    summary = wdf.groupby(["variant", "station", "is_true_catchment"])["uplift_ratio_llm"].mean().reset_index()
    print(summary.sort_values(["is_true_catchment", "station", "variant"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--budget_s", type=float, default=260.0)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--wc2026check", action="store_true")
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS)
    args = parser.parse_args()

    if args.prepare:
        phase_prepare()
    if args.train:
        phase_train(args.budget_s)
    if args.summarize:
        phase_summarize()
    if args.wc2026check:
        phase_wc2026check(only_variant=args.variant)
