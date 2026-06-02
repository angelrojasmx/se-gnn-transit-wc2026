"""
Leave-One-Type-Out Cross-Validation (LOTO-CV) for the multi-venue LLM-GNN.

Split design:
  With n=12 events across 4 types (3 instances each), LOTO-CV is the most
  demanding and scientifically honest protocol available:

    Fold 1 — TEST: soccer_international (3 Copa América 2024)
             TRAIN: concert_pop + parade_street + race_marathon (9 events)
    Fold 2 — TEST: concert_pop (3 Taylor Swift MetLife)
             TRAIN: soccer + parade + marathon (9)
    Fold 3 — TEST: parade_street (3 NYC Pride)
             TRAIN: soccer + concert + marathon (9)
    Fold 4 — TEST: race_marathon (3 NYC Marathon)
             TRAIN: soccer + concert + parade (9)

  Why LOTO rather than global LOO:
    Global LOO would allow the model to see Pride 2022 and 2024 before
    evaluating on Pride 2023 — the event type, location, and schedule are
    already seen.  LOTO asks the harder question: "can the model predict an
    event type it has never seen at all?"

Evaluation metric: venue-aware MAE.
  For each test event, MAE is computed only over stations in the venue's
  catchment zone, not across all 121 stations.  Error on out-of-catchment
  stations is noise, not signal:
    Parade  -> West Village [323, 324, 601, 618]
    Marathon -> Central Park [160, 313, 614, ...]
    Soccer/Concert MetLife -> Penn corridor [164, 318, 607]

Baselines compared:
  1. Backbone GNN (no FiLM conditioning)
  2. Historical Average (same DOW, 4 weeks prior)
  3. Linear Regression (OLS: hour, DOW, month, event flag, kickoff hour)
  4. SARIMA(1,1,1)x(1,1,0,7) on daily totals + DOW shape projection

Outputs:
  outputs/paper/nyc/multivenue_loocv_results.csv
  outputs/paper/nyc/multivenue_loocv_summary.csv
  outputs/paper/comparative/figures/fig_multivenue_loocv.png
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Venue-station mapping
VENUE_SIDS: dict[str, list[int]] = {
    "MetLife_Penn": [164, 318, 607],
    "WVillage":     [323, 324, 601, 618],
    "CentralPark":  [158, 159, 160, 311, 313, 397, 476, 477, 614],
    "MSG":          [164, 318, 607, 319, 403, 404, 13],
}

# Event type → venue cluster
TYPE_TO_VENUE: dict[str, str] = {
    "soccer_international":    "MetLife_Penn",
    "concert_pop":             "MetLife_Penn",
    "parade_street":           "WVillage",
    "race_marathon":           "CentralPark",
    "event_msg_unconfirmed":   "MSG",
}

HOURS = [f"h{i:02d}" for i in range(24)]


# Data loading

def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns:
        profiles  — daily_profiles.parquet for Manhattan
        catalog   — expanded event catalog
        embeddings — (n_events, 384) float32, or None if not generated yet
    """
    profiles = pd.read_parquet(root / "data/nyc/manhattan/daily_profiles.parquet")
    profiles["date"] = pd.to_datetime(profiles["date"])

    with open(root / "data/nyc/event_catalog_expanded_v2.json", encoding="utf-8") as f:
        catalog_raw = json.load(f)
    catalog = pd.DataFrame(catalog_raw)
    catalog["date"] = pd.to_datetime(catalog["date"])

    emb_file = root / "outputs/paper/nyc/event_embeddings_expanded.npy"
    if emb_file.exists():
        embeddings = np.load(emb_file).astype(np.float32)
    else:
        embeddings = None
        print("WARNING: event_embeddings_expanded.npy not found.")
        print("  GNN predictions will be skipped (mae_gnn_base and mae_llm_gnn = None).")

    return profiles, catalog, embeddings


# Baseline 1: Historical Average (same DOW, 4 weeks before)

def historical_average_predict(
    profiles: pd.DataFrame,
    event_date: pd.Timestamp,
    venue_sids: list[int],
    n_weeks: int = 4,
) -> np.ndarray:
    """
    Returns mean normalized hourly profile across venue_sids on comparable days.
    comparable = same weekday, [2..n_weeks+1] weeks before event_date.
    """
    dow = event_date.dayofweek
    # Avoid adjacent event days: exclude ±14 days
    candidates = profiles[
        (profiles["sid"].isin(venue_sids)) &
        (profiles["date"].dt.dayofweek == dow) &
        (profiles["date"] < event_date - pd.Timedelta(days=14)) &
        (profiles["date"] >= event_date - pd.Timedelta(weeks=n_weeks + 1))
    ]
    if candidates.empty:
        # Fallback: same DOW any time before
        candidates = profiles[
            (profiles["sid"].isin(venue_sids)) &
            (profiles["date"].dt.dayofweek == dow) &
            (profiles["date"] < event_date)
        ]
    return candidates[HOURS].mean().values  # shape (24,)


# Baseline 2: Linear Regression

def linear_regression_predict(
    profiles: pd.DataFrame,
    event_date: pd.Timestamp,
    venue_sids: list[int],
    event_desc: str,
    kickoff_et: str,
) -> np.ndarray:
    """
    OLS per-hour: features = [hour, DOW, month, is_event, kickoff_hour]
    Trained on all non-event days + known event days (excluding test event).
    Returns predicted normalized profile (24 values) for venue_sids mean.
    """
    from sklearn.linear_model import LinearRegression

    venue_df = profiles[profiles["sid"].isin(venue_sids)].copy()
    daily = venue_df.groupby("date")[HOURS].mean().reset_index()
    daily["dow"]   = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_event"] = daily["date"].isin(
        profiles[profiles["is_event"] == 1]["date"].unique()
    ).astype(int)

    # Parse kickoff hour
    try:
        kickoff_h = int(kickoff_et.split(":")[0])
    except (ValueError, AttributeError, IndexError):
        kickoff_h = 18

    # Train on all days except test event date
    train = daily[daily["date"] != event_date].copy()
    train["kickoff_h"] = train["is_event"] * kickoff_h

    feats = ["dow", "month", "is_event", "kickoff_h"]

    preds = []
    for h in HOURS:
        X_tr = train[feats].values
        y_tr = train[h].values
        lr   = LinearRegression().fit(X_tr, y_tr)
        # Test: same DOW, month, is_event=1, kickoff_h
        x_te = np.array([[event_date.dayofweek, event_date.month, 1, kickoff_h]])
        preds.append(float(lr.predict(x_te)[0]))

    return np.array(preds)


# Baseline 3: SARIMA

def sarima_predict(
    profiles: pd.DataFrame,
    event_date: pd.Timestamp,
    venue_sids: list[int],
) -> np.ndarray:
    """
    Fits SARIMA(1,1,1)×(1,1,0,7) on daily totals for venue, returns
    predicted hourly profile by distributing predicted total using
    same-DOW historical shape.

    Note: full SARIMA on hourly data (SARIMA×24) is too slow for LOO-CV;
    we use daily-level SARIMA + shape projection as a practical compromise.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("  statsmodels not available, falling back to historical average")
        return historical_average_predict(profiles, event_date, venue_sids)

    venue_df = profiles[profiles["sid"].isin(venue_sids)].copy()
    daily = (venue_df.groupby("date")["daily_total"].sum()
             .reset_index().sort_values("date"))
    daily = daily[daily["date"] < event_date].copy()
    daily = daily.set_index("date")["daily_total"]

    if len(daily) < 30:
        return historical_average_predict(profiles, event_date, venue_sids)

    try:
        model  = SARIMAX(daily, order=(1, 1, 1),
                         seasonal_order=(1, 1, 0, 7),
                         enforce_stationarity=False,
                         enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=100)
        pred_total = float(result.forecast(1).iloc[0])
    except Exception as exc:   # SARIMAX can fail on short/irregular series
        print(f"    SARIMA fit failed ({type(exc).__name__}); using 7-day mean fallback")
        pred_total = float(daily.iloc[-7:].mean())

    # Distribute predicted total using same-DOW historical shape
    dow = event_date.dayofweek
    same_dow = venue_df[
        (venue_df["date"].dt.dayofweek == dow) &
        (venue_df["date"] < event_date)
    ][HOURS].mean().values
    same_dow = np.maximum(same_dow, 0)
    if same_dow.sum() > 0:
        same_dow = same_dow / same_dow.sum()

    return same_dow


# Main LOTO-CV loop

def run_loto_cv(root: Path) -> pd.DataFrame:
    """
    Leave-One-Type-Out cross-validation.

    For each test fold (event type):
      1. Identify training events (all other types)
      2. For each test event:
         a. Compute all baseline predictions
         b. Compute GNN baseline prediction (no FiLM)
         c. Compute LLM-GNN prediction (with FiLM, trained on train fold)
      3. Compute venue-aware MAE for each method

    Returns tidy results DataFrame.
    """
    profiles, catalog, embeddings = load_data(root)

    # Confirmed events only
    catalog = catalog[catalog["confirmed"] == True].copy()
    catalog = catalog.reset_index(drop=True)

    event_types_all = catalog["event_type"].unique().tolist()
    print(f"\nEvent types: {event_types_all}")
    print(f"Total confirmed events: {len(catalog)}\n")

    results = []

    for fold_i, test_type in enumerate(event_types_all):
        test_events  = catalog[catalog["event_type"] == test_type]
        train_events = catalog[catalog["event_type"] != test_type]

        venue_cluster = TYPE_TO_VENUE.get(test_type, "MetLife_Penn")
        venue_sids    = VENUE_SIDS[venue_cluster]

        print(f"Fold {fold_i+1}: TEST={test_type} ({len(test_events)} events) "
              f"TRAIN={len(train_events)} events")
        print(f"   Venue: {venue_cluster}, stations: {venue_sids}")

        for _, ev in test_events.iterrows():
            event_date = ev["date"]
            event_id   = ev["event_id"]
            kickoff    = str(ev.get("kickoff_et", "18:00"))

            # Ground truth: actual normalized profile at venue stations
            gt_rows = profiles[
                (profiles["date"] == event_date) &
                (profiles["sid"].isin(venue_sids))
            ]
            if gt_rows.empty:
                print(f"   SKIP {event_id}: no profile data")
                continue
            gt_profile = gt_rows[HOURS].mean().values  # (24,)

            # Baseline 1: Historical Average
            ha_profile = historical_average_predict(
                profiles, event_date, venue_sids)
            mae_ha = float(np.abs(gt_profile - ha_profile).mean())

            # Baseline 2: Linear Regression
            lr_profile = linear_regression_predict(
                profiles, event_date, venue_sids, ev["description"], kickoff)
            mae_lr = float(np.abs(gt_profile - lr_profile).mean())

            # Baseline 3: SARIMA
            sarima_profile = sarima_predict(profiles, event_date, venue_sids)
            mae_sarima = float(np.abs(gt_profile - sarima_profile).mean())

            # GNN backbone (no FiLM): placeholder to be filled from local model run
            mae_gnn_base = None

            # SE-GNN with FiLM conditioning: placeholder to be filled from local model run
            mae_llm_gnn = None

            row = {
                "fold":          fold_i + 1,
                "test_type":     test_type,
                "event_id":      event_id,
                "date":          event_date.date(),
                "venue_cluster": venue_cluster,
                "n_venue_sids":  len(venue_sids),
                "mae_hist_avg":  round(mae_ha, 5),
                "mae_lin_reg":   round(mae_lr, 5),
                "mae_sarima":    round(mae_sarima, 5),
                "mae_gnn_base":  mae_gnn_base,
                "mae_llm_gnn":   mae_llm_gnn,
            }
            results.append(row)
            print(f"   {event_id}: HistAvg={mae_ha:.4f} LinReg={mae_lr:.4f} "
                  f"SARIMA={mae_sarima:.4f}")

    df = pd.DataFrame(results)
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fold mean ± std across events, for each method.
    Computes improvement ratios vs HistAvg baseline.
    """
    methods = ["mae_hist_avg", "mae_lin_reg", "mae_sarima",
               "mae_gnn_base", "mae_llm_gnn"]
    methods = [m for m in methods if m in df.columns and df[m].notna().any()]

    summary_rows = []
    for fold, grp in df.groupby(["fold", "test_type", "venue_cluster"]):
        row = {"fold": fold[0], "test_type": fold[1], "venue_cluster": fold[2],
               "n_events": len(grp)}
        for m in methods:
            vals = grp[m].dropna()
            if vals.empty:
                row[f"{m}_mean"] = None
                row[f"{m}_std"]  = None
            else:
                row[f"{m}_mean"] = round(vals.mean(), 5)
                row[f"{m}_std"]  = round(vals.std(), 5)
        summary_rows.append(row)

    # Overall (pooled across folds)
    row = {"fold": "ALL", "test_type": "all", "venue_cluster": "all",
           "n_events": len(df)}
    for m in methods:
        vals = df[m].dropna()
        if vals.empty:
            row[f"{m}_mean"] = None
            row[f"{m}_std"]  = None
        else:
            row[f"{m}_mean"] = round(vals.mean(), 5)
            row[f"{m}_std"]  = round(vals.std(), 5)
    summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compute_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap 95% CI on mean MAE improvement of LLM-GNN vs HistAvg,
    pooled across all folds. Requires mae_llm_gnn to be non-null.
    """
    if "mae_llm_gnn" not in df.columns or df["mae_llm_gnn"].isna().all():
        print("  LLM-GNN predictions not yet available — run model locally first.")
        return pd.DataFrame()

    n_boot = 5000
    rng    = np.random.default_rng(42)
    ci_rows = []

    for method in ["mae_hist_avg", "mae_lin_reg", "mae_sarima", "mae_gnn_base"]:
        if method not in df.columns or df[method].isna().all():
            continue
        improvements = (df[method] - df["mae_llm_gnn"]).dropna().values  # positive = LLM better
        if len(improvements) == 0:
            continue
        boot_means = [rng.choice(improvements, size=len(improvements), replace=True).mean()
                      for _ in range(n_boot)]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        pct_improve  = (improvements.mean() / df[method].mean() * 100)
        ci_rows.append({
            "vs_baseline":     method,
            "mean_improvement": round(improvements.mean(), 5),
            "pct_improvement":  round(pct_improve, 1),
            "ci_lo_95":        round(ci_lo, 5),
            "ci_hi_95":        round(ci_hi, 5),
            "n_events":        len(improvements),
            "significant":     bool(ci_lo > 0),
        })

    return pd.DataFrame(ci_rows)


# Figure: LOTO-CV results

def plot_loocv_results(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

    METHODS = {
        "mae_hist_avg": ("Historical Avg", "#AAAAAA", "--"),
        "mae_lin_reg":  ("Linear Reg",     "#888888", ":"),
        "mae_sarima":   ("SARIMA",          "#666666", "-."),
        "mae_gnn_base": ("GNN (no FiLM)",  "#E63946", "-"),
        "mae_llm_gnn":  ("LLM-GNN (FiLM)","#2A9D8F", "-"),
    }

    type_labels = {
        "soccer_international": "Soccer\n(MetLife/Penn)",
        "concert_pop":          "Concert\n(MetLife/Penn)",
        "parade_street":        "Parade\n(West Village)",
        "race_marathon":        "Marathon\n(Central Park)",
    }

    folds = sorted(df["fold"].unique())
    for ax_i, fold_num in enumerate(folds):
        ax   = axes[ax_i]
        fold_df = df[df["fold"] == fold_num]
        test_type = fold_df["test_type"].iloc[0]
        label = type_labels.get(test_type, test_type)

        x = np.arange(len(fold_df))
        for method, (name, color, ls) in METHODS.items():
            if method not in fold_df.columns:
                continue
            vals = fold_df[method].values
            if pd.isna(vals).all():
                continue
            ax.plot(x, vals, color=color, ls=ls, lw=2, marker="o",
                    markersize=6, label=name)

        ax.set_title(f"Fold {fold_num}: {label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Test event")
        if ax_i == 0:
            ax.set_ylabel("Venue-aware MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [e.replace("copa_am_2024_", "CA_")
              .replace("taylor_swift_metlife_oct", "TS_oct")
              .replace("nyc_pride_", "Pride_")
              .replace("nyc_marathon_", "Mar_")
             for e in fold_df["event_id"].tolist()],
            fontsize=7, rotation=30, ha="right"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Summary row
    pooled = summary[summary["fold"] == "ALL"]
    if not pooled.empty:
        methods_avail = [m for m in METHODS if f"{m}_mean" in pooled.columns
                         and pooled[f"{m}_mean"].notna().any()]
        print("\nPooled MAE summary:")
        for m in methods_avail:
            mean = pooled[f"{m}_mean"].values[0]
            std  = pooled.get(f"{m}_std", pd.Series([None])).values[0]
            print(f"  {METHODS[m][0]:25s}: {mean:.5f} ± {std:.5f}" if std else
                  f"  {METHODS[m][0]:25s}: {mean:.5f}")

    fig.suptitle(
        "Multi-Venue LOTO-CV: LLM-GNN vs Baselines\n"
        "(Leave-One-Type-Out, venue-aware MAE)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / "fig_multivenue_loocv.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved -> {out_path}")


# Entry point

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-only", action="store_true",
                        help="Compute only baseline predictions (no GNN, runs without GPU)")
    args = parser.parse_args()

    OUT_DIR  = ROOT / "outputs/paper/nyc"
    FIG_DIR  = ROOT / "outputs/paper/comparative/figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-VENUE LEAVE-ONE-TYPE-OUT CV")
    print("=" * 60)

    df_results = run_loto_cv(ROOT)

    # Save raw results
    df_results.to_csv(OUT_DIR / "multivenue_loocv_results.csv", index=False)
    print(f"\nResults saved -> {OUT_DIR}/multivenue_loocv_results.csv")

    # Summary table
    summary = summarize_results(df_results)
    summary.to_csv(OUT_DIR / "multivenue_loocv_summary.csv", index=False)

    print("\nSummary by fold:")
    print(summary[["fold","test_type","n_events","mae_hist_avg_mean",
                   "mae_lin_reg_mean","mae_sarima_mean"]].to_string(index=False))

    # Confidence intervals (only if SE-GNN results are available)
    ci = compute_confidence_intervals(df_results)
    if not ci.empty:
        ci.to_csv(OUT_DIR / "multivenue_loocv_ci.csv", index=False)
        print("\nConfidence intervals (SE-GNN vs baselines):")
        print(ci.to_string(index=False))

    # Figure
    plot_loocv_results(df_results, summary, FIG_DIR)

    print("\nDone. To complete the GNN results, run the model locally and populate")
    print("the mae_gnn_base and mae_llm_gnn columns in the output CSV.")
