"""
Tier 1 CDMX: transit demand shocks and WC2026 projections.

Method:
  1. Load daily post-COVID multimodal ridership (SEMOVI CDMX, 2022-2025).
  2. Compute MAD-based robust z-score by mode × day-of-week; identify top-N shocks.
  3. Build uplift distribution (P25/P50/P75) over those shock days.
  4. Project WC2026 using a June–July seasonal baseline (lower due to school vacation).
  5. Produce citywide analysis and a focused analysis of Azteca-adjacent Metro/Metrobús lines.

Outputs (outputs/paper/cdmx/):
  shock_days_by_mode.csv
  baseline_jun_jul_by_mode.csv
  uplift_distribution_by_mode.csv
  wc2026_projections_cdmx.csv
  azteca_corridor_shocks.csv
  summary_cdmx.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (
    CDMX_AFLUENCIA, POST_COVID_START, TOP_N_SHOCK_DAYS,
    UPLIFT_PERCENTILES, WC2026_CDMX,
    AZTECA_METRO_LINES, AZTECA_METROBUS_LINES, OUT_CDMX,
)
from utils import uplift_distribution, project_wc2026, date_to_weekday_es


def load_afluencia() -> pd.DataFrame:
    """
    Load and normalise the SEMOVI CDMX multimodal ridership dataset.

    Strips whitespace from mode and line name columns, which contain
    inconsistent formatting in the source CSV.
    """
    print("Loading CDMX ridership...")
    df = pd.read_csv(CDMX_AFLUENCIA, low_memory=False, parse_dates=["fecha"])
    df = df[df["fecha"] >= POST_COVID_START].copy()
    df["mode"]  = df["mode"].str.strip()
    df["linea"] = df["linea"].str.strip()
    print(f"  {len(df):,} rows  {df['fecha'].min().date()} -> {df['fecha'].max().date()}")
    return df


def build_daily_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly ridership to daily totals per mode (citywide sum)."""
    daily = (
        df.groupby(["fecha", "mode", "weekday"])["afluencia"]
          .sum()
          .reset_index()
    )
    return daily.sort_values(["mode", "fecha"]).reset_index(drop=True)


def compute_shocks(
    daily: pd.DataFrame,
    top_n: int = TOP_N_SHOCK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute MAD-based robust z-scores and extract the top-N shock days per mode.

    Z-scores are computed within each mode × weekday group so that
    weekend-sized ridership does not suppress weekday shocks.
    """
    def _mad_z(grp: pd.DataFrame) -> pd.DataFrame:
        med = grp["afluencia"].median()
        mad = (grp["afluencia"] - med).abs().median()
        grp = grp.copy()
        if mad < 1:
            grp["z_robust"] = 0.0
        else:
            grp["z_robust"] = (grp["afluencia"] - med) / (1.4826 * mad)   # 1.4826 = MAD consistency factor
        return grp

    daily = daily.groupby(["mode", "weekday"], group_keys=False).apply(_mad_z)

    top_shocks = (
        daily.groupby("mode", group_keys=False)
             .apply(lambda g: g.nlargest(top_n, "z_robust"))
             .reset_index(drop=True)
    )
    return daily, top_shocks


def compute_baseline_june_july(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the expected ridership baseline for each mode × weekday in June–July.

    June and July have depressed ridership due to school holidays in Mexico City.
    Using a full-year median would overestimate the WC2026 denominator and
    understate the projected uplift.  Outliers (|z| > 2) are excluded to
    prevent prior event days from inflating the baseline.

    Returns:
        DataFrame with columns: mode, weekday, baseline_jun_jul.
    """
    jj = daily[daily["fecha"].dt.month.isin([6, 7])].copy()

    def _clean_median(grp: pd.Series) -> float:
        med   = grp.median()
        mad   = (grp - med).abs().median()
        z     = (grp - med) / (1.4826 * mad + 1e-8)   # 1.4826 = MAD consistency factor
        clean = grp[z.abs() <= 2]
        return clean.median() if len(clean) > 3 else grp.median()

    baseline = (
        jj.groupby(["mode", "weekday"])["afluencia"]
          .apply(_clean_median)
          .reset_index()
          .rename(columns={"afluencia": "baseline_jun_jul"})
    )
    return baseline


def compute_uplift_per_mode(
    daily:       pd.DataFrame,
    baseline_df: pd.DataFrame,
    top_n:       int = TOP_N_SHOCK_DAYS,
) -> pd.DataFrame:
    """
    Compute P25/P50/P75 uplift multipliers for each mode × weekday combination.

    Returns a DataFrame with one row per (mode, weekday) containing the
    low/mid/high scenario uplift multipliers.
    """
    results = []
    for mode in daily["mode"].unique():
        md = daily[daily["mode"] == mode].copy()
        for wday in md["weekday"].unique():
            sub    = md[md["weekday"] == wday].copy()
            bl_row = baseline_df[
                (baseline_df["mode"] == mode) & (baseline_df["weekday"] == wday)
            ]
            if bl_row.empty:
                continue
            bl = bl_row["baseline_jun_jul"].values[0]
            if bl < 1:
                continue
            dist = uplift_distribution(
                daily_series    = sub.set_index("fecha")["afluencia"],
                baseline_series = pd.Series(bl, index=sub["fecha"]),
                top_n           = top_n,
                percentiles     = UPLIFT_PERCENTILES,
            )
            results.append({
                "mode":            mode,
                "weekday":         wday,
                "baseline_jun_jul": bl,
                "uplift_low":      dist["uplift_low"],
                "uplift_mid":      dist["uplift_mid"],
                "uplift_high":     dist["uplift_high"],
                "n_top_days":      len(dist["top_days"]),
            })
    return pd.DataFrame(results)


def project_wc2026_cdmx(
    uplift_df:   pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-mode uplift distributions to project WC2026 CDMX demand."""
    rows = []
    for date_str, kickoff, matchup, stage in WC2026_CDMX:
        wday = date_to_weekday_es(date_str)
        for mode in uplift_df["mode"].unique():
            up_row = uplift_df[
                (uplift_df["mode"] == mode) & (uplift_df["weekday"] == wday)
            ]
            bl_row = baseline_df[
                (baseline_df["mode"] == mode) & (baseline_df["weekday"] == wday)
            ]
            if up_row.empty or bl_row.empty:
                continue
            up   = up_row.iloc[0]
            bl   = bl_row["baseline_jun_jul"].values[0]
            proj = project_wc2026(bl, up["uplift_low"], up["uplift_mid"], up["uplift_high"])
            rows.append({
                "date":            date_str,
                "weekday":         wday,
                "kickoff":         kickoff,
                "matchup":         matchup,
                "stage":           stage,
                "mode":            mode,
                "baseline":        bl,
                "proj_low":        proj["proj_low"],
                "proj_mid":        proj["proj_mid"],
                "proj_high":       proj["proj_high"],
                "extra_low":       proj["extra_low"],
                "extra_mid":       proj["extra_mid"],
                "extra_high":      proj["extra_high"],
                "uplift_low":      up["uplift_low"],
                "uplift_mid":      up["uplift_mid"],
                "uplift_high":     up["uplift_high"],
            })
    return pd.DataFrame(rows)


def azteca_corridor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse ridership shocks on Metro and Metrobús lines directly serving
    Estadio Azteca (Line 2 Metrobús, Line 12 Metro, etc.).

    This provides a line-level view of demand concentration, complementing
    the citywide mode-level analysis.  Returns the top-10 shock days per
    line, ranked by robust z-score.
    """
    df = df.copy()
    df["linea_norm"] = (
        df["linea"].str.strip()
                   .str.replace("Línea", "Linea", regex=False)
    )

    az_metro = df[
        (df["mode"] == "STC_Metro") &
        df["linea_norm"].isin([l.replace("Línea", "Linea") for l in AZTECA_METRO_LINES])
    ]
    az_mb = df[
        (df["mode"] == "Metrobus") &
        df["linea_norm"].isin([
            l.replace("Línea", "Linea").replace("linea ", "Linea ")
            for l in AZTECA_METROBUS_LINES
        ])
    ]
    az_all = pd.concat([az_metro, az_mb])

    if az_all.empty:
        print("  WARNING: No data found for Azteca-adjacent lines.")
        return pd.DataFrame()

    daily_az = (
        az_all.groupby(["fecha", "mode", "linea_norm", "weekday"])["afluencia"]
              .sum()
              .reset_index()
    )

    def _mad_z(grp: pd.DataFrame) -> pd.DataFrame:
        med = grp["afluencia"].median()
        mad = (grp["afluencia"] - med).abs().median()
        grp = grp.copy()
        grp["z_robust"] = (grp["afluencia"] - med) / (1.4826 * mad + 1e-8)   # 1.4826 = MAD consistency factor
        return grp

    daily_az = daily_az.groupby(
        ["mode", "linea_norm", "weekday"], group_keys=False
    ).apply(_mad_z)

    top_az = (
        daily_az.groupby(["mode", "linea_norm"], group_keys=False)
                .apply(lambda g: g.nlargest(10, "z_robust"))
                .reset_index(drop=True)
    )
    return top_az


def run():
    OUT_CDMX.mkdir(parents=True, exist_ok=True)

    df = load_afluencia()

    print("Aggregating by mode...")
    daily = build_daily_by_mode(df)

    print("Computing shocks...")
    daily, top_shocks = compute_shocks(daily)
    top_shocks.to_csv(OUT_CDMX / "shock_days_by_mode.csv", index=False)
    print(f"  {len(top_shocks)} rows in shock_days_by_mode.csv")

    print("Computing June-July baseline...")
    baseline_df = compute_baseline_june_july(daily)
    baseline_df.to_csv(OUT_CDMX / "baseline_jun_jul_by_mode.csv", index=False)

    print("Computing uplift distribution...")
    uplift_df = compute_uplift_per_mode(daily, baseline_df)
    uplift_df.to_csv(OUT_CDMX / "uplift_distribution_by_mode.csv", index=False)
    print(f"  Mean mid uplift: {uplift_df['uplift_mid'].mean():.3f}x")

    print("Projecting WC2026 CDMX...")
    wc_proj = project_wc2026_cdmx(uplift_df, baseline_df)
    wc_proj.to_csv(OUT_CDMX / "wc2026_projections_cdmx.csv", index=False)
    print(f"  {len(wc_proj)} projection rows")

    print("Analysing Azteca corridor...")
    az_df = azteca_corridor_analysis(df)
    if not az_df.empty:
        az_df.to_csv(OUT_CDMX / "azteca_corridor_shocks.csv", index=False)

    summary = (
        wc_proj.groupby("mode")
               .agg(
                   baseline_median  = ("baseline",   "median"),
                   extra_mid_median = ("extra_mid",  "median"),
                   uplift_mid_median= ("uplift_mid", "median"),
                   games            = ("date",       "nunique"),
               )
               .reset_index()
    )
    summary["pct_uplift_mid"] = (summary["uplift_mid_median"] - 1) * 100
    summary.to_csv(OUT_CDMX / "summary_cdmx.csv", index=False)

    print("\n=== CDMX Tier 1 complete ===")
    print(summary[["mode", "baseline_median", "extra_mid_median", "pct_uplift_mid"]].to_string(index=False))
    return wc_proj, uplift_df


if __name__ == "__main__":
    run()
