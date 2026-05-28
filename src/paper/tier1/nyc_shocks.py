"""
Tier 1 NYC: Penn Station corridor demand shocks and WC2026 projections.

Method:
  1. Load MTA hourly ridership 2022-2024; aggregate to daily per station.
  2. Focus on Penn Station hub (sids 164, 318, 607): these feed MetLife via NJ Transit.
  3. Compute MAD-based robust z-score by station × day-of-week; identify top-N shocks.
  4. Cross-reference with Copa América 2024 at MetLife — the closest historical analog.
  5. Compute uplift distribution (P25/P50/P75) and project for each WC2026 match.

Outputs (outputs/paper/nyc/):
  daily_penn_corridor_by_station.csv
  shock_days_penn.csv
  copa_america_2024_impact.csv
  uplift_distribution_penn.csv
  wc2026_projections_nyc.csv
  summary_nyc.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (
    NYC_HOURLY, POST_COVID_START, TOP_N_SHOCK_DAYS,
    UPLIFT_PERCENTILES, WC2026_NYC, PENN_CORRIDOR, OUT_NYC,
    COPA_AMERICA_METLIFE_2024,
)
from utils import uplift_distribution, project_wc2026


def load_nyc_daily(sids: list = None) -> pd.DataFrame:
    """
    Load MTA hourly ridership and aggregate to daily totals.

    Args:
        sids: If provided, keep only these station IDs.

    Returns:
        DataFrame with columns: date, sid, ridership, weekday.
    """
    print("Loading NYC MTA hourly ridership...")
    df = pd.read_parquet(NYC_HOURLY)
    df["date"] = df["ts"].dt.date.astype(str)
    df["hour"] = df["ts"].dt.hour

    if sids:
        df = df[df["sid"].isin(sids)]

    print(f"  {len(df):,} hourly rows for {df['sid'].nunique()} stations")

    daily = (
        df.groupby(["date", "sid"])["ridership"]
          .sum()
          .reset_index()
    )
    daily = daily[daily["date"] >= POST_COVID_START]
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name()
    return daily


def compute_corridor_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-station daily ridership to a single corridor total."""
    corridor = (
        daily.groupby(["date", "weekday"])["ridership"]
             .sum()
             .reset_index()
             .rename(columns={"ridership": "ridership_corridor"})
    )
    return corridor


def compute_shocks(
    corridor: pd.DataFrame,
    top_n: int = TOP_N_SHOCK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute MAD-based robust z-scores within each day-of-week group.

    A within-weekday normalisation ensures that Saturday's naturally higher
    ridership does not mask weekday shocks or produce inflated z-scores.

    Returns:
        (corridor_with_z, top_n_shock_days)
    """
    def _mad_z(grp: pd.DataFrame) -> pd.DataFrame:
        med  = grp["ridership_corridor"].median()
        mad  = (grp["ridership_corridor"] - med).abs().median()
        grp  = grp.copy()
        grp["z_robust"] = (grp["ridership_corridor"] - med) / (1.4826 * mad + 1e-8)   # 1.4826 = MAD consistency factor
        return grp

    corridor = corridor.groupby("weekday", group_keys=False).apply(_mad_z)
    top      = corridor.nlargest(top_n, "z_robust").copy()
    return corridor, top


def copa_america_impact(corridor: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Penn Station corridor ridership for Copa América 2024 match days.

    These three games at MetLife Stadium (same venue as WC2026) are the
    highest-quality historical analog available.  Observed uplifts here
    directly inform the WC2026 projection scenarios.
    """
    ca_dates = [date for date, _, _, _ in COPA_AMERICA_METLIFE_2024]
    ca_df    = corridor[corridor["date"].isin(ca_dates)].copy()

    def weekday_median(row):
        return corridor[
            corridor["weekday"] == row["weekday"]
        ]["ridership_corridor"].median()

    ca_df["baseline_wday"]   = ca_df.apply(weekday_median, axis=1)
    ca_df["uplift_observed"] = (
        ca_df["ridership_corridor"] / ca_df["baseline_wday"].clip(lower=1)
    )

    match_info = {
        date: (matchup, stage)
        for date, _, matchup, stage in COPA_AMERICA_METLIFE_2024
    }
    ca_df["matchup"] = ca_df["date"].map(lambda d: match_info.get(d, ("", ""))[0])
    ca_df["stage"]   = ca_df["date"].map(lambda d: match_info.get(d, ("", ""))[1])

    return ca_df[[
        "date", "weekday", "matchup", "stage",
        "ridership_corridor", "baseline_wday", "uplift_observed", "z_robust",
    ]]


def compute_baseline_june_july(corridor: pd.DataFrame) -> dict:
    """
    Compute a weekday-specific ridership baseline for June–July.

    June and July have depressed commuter ridership due to school vacations,
    so a full-year baseline would overestimate the WC2026 denominator.
    Outliers (|z| > 2) are excluded to avoid prior event days contaminating
    the baseline estimate.

    Returns:
        Dict mapping weekday name -> median ridership (excluding outliers).
    """
    jj       = corridor[pd.to_datetime(corridor["date"]).dt.month.isin([6, 7])].copy()
    baseline = {}
    for wday, grp in jj.groupby("weekday"):
        med   = grp["ridership_corridor"].median()
        mad   = (grp["ridership_corridor"] - med).abs().median()
        z     = (grp["ridership_corridor"] - med) / (1.4826 * mad + 1e-8)   # 1.4826 = MAD consistency factor
        clean = grp[z.abs() <= 2]["ridership_corridor"]
        baseline[wday] = clean.median() if len(clean) > 3 else med
    return baseline


def compute_uplift_distribution(
    corridor:         pd.DataFrame,
    baseline_by_wday: dict,
    top_n:            int = TOP_N_SHOCK_DAYS,
) -> pd.DataFrame:
    """
    Compute P25/P50/P75 uplift for each weekday group.

    Returns a DataFrame with one row per weekday containing the low/mid/high
    uplift multipliers used for scenario projections.
    """
    results = []
    for wday, grp in corridor.groupby("weekday"):
        bl = baseline_by_wday.get(wday)
        if bl is None or bl < 1:
            continue
        dist = uplift_distribution(
            daily_series    = grp.set_index("date")["ridership_corridor"],
            baseline_series = pd.Series(bl, index=grp["date"]),
            top_n           = top_n,
            percentiles     = UPLIFT_PERCENTILES,
        )
        results.append({
            "weekday":          wday,
            "baseline_jun_jul": bl,
            "uplift_low":       dist["uplift_low"],
            "uplift_mid":       dist["uplift_mid"],
            "uplift_high":      dist["uplift_high"],
            "n_top_days":       len(dist["top_days"]),
        })
    return pd.DataFrame(results)


def project_wc2026_nyc(
    uplift_df:        pd.DataFrame,
    baseline_by_wday: dict,
) -> pd.DataFrame:
    """Apply uplift distribution to baseline to produce WC2026 projections."""
    rows = []
    for date_str, kickoff, matchup, stage in WC2026_NYC:
        wday    = pd.Timestamp(date_str).day_name()
        bl      = baseline_by_wday.get(wday)
        up_row  = uplift_df[uplift_df["weekday"] == wday]
        if up_row.empty or bl is None:
            continue
        up   = up_row.iloc[0]
        proj = project_wc2026(bl, up["uplift_low"], up["uplift_mid"], up["uplift_high"])
        rows.append({
            "date":               date_str,
            "weekday":            wday,
            "kickoff_et":         kickoff,
            "matchup":            matchup,
            "stage":              stage,
            "baseline_corridor":  bl,
            "proj_low":           proj["proj_low"],
            "proj_mid":           proj["proj_mid"],
            "proj_high":          proj["proj_high"],
            "extra_low":          proj["extra_low"],
            "extra_mid":          proj["extra_mid"],
            "extra_high":         proj["extra_high"],
            "uplift_low":         up["uplift_low"],
            "uplift_mid":         up["uplift_mid"],
            "uplift_high":        up["uplift_high"],
        })
    return pd.DataFrame(rows)


def run():
    OUT_NYC.mkdir(parents=True, exist_ok=True)

    print("Loading Penn Station corridor data...")
    daily = load_nyc_daily(sids=PENN_CORRIDOR)
    daily.to_csv(OUT_NYC / "daily_penn_corridor_by_station.csv", index=False)

    corridor = compute_corridor_daily(daily)

    print("Computing shocks...")
    corridor, top_shocks = compute_shocks(corridor)
    top_shocks.to_csv(OUT_NYC / "shock_days_penn.csv", index=False)

    print("Extracting Copa América 2024 impact...")
    ca_impact = copa_america_impact(corridor)
    ca_impact.to_csv(OUT_NYC / "copa_america_2024_impact.csv", index=False)
    print("  Observed uplifts:")
    print(ca_impact[["date", "matchup", "uplift_observed", "z_robust"]].to_string(index=False))

    print("Computing June-July baseline...")
    baseline_by_wday = compute_baseline_june_july(corridor)

    print("Computing uplift distribution...")
    uplift_df = compute_uplift_distribution(corridor, baseline_by_wday)
    uplift_df.to_csv(OUT_NYC / "uplift_distribution_penn.csv", index=False)
    print(f"  Mean mid uplift: {uplift_df['uplift_mid'].mean():.3f}x")

    print("Projecting WC2026 NYC...")
    wc_proj = project_wc2026_nyc(uplift_df, baseline_by_wday)
    wc_proj.to_csv(OUT_NYC / "wc2026_projections_nyc.csv", index=False)

    summary = pd.DataFrame([{
        "corridor":           "Penn Station (34 St A/C/E + 1/2/3 + Herald Sq)",
        "stations":           len(PENN_CORRIDOR),
        "baseline_sat_med":   baseline_by_wday.get("Saturday", 0),
        "baseline_sun_med":   baseline_by_wday.get("Sunday", 0),
        "uplift_mid_avg":     uplift_df["uplift_mid"].mean(),
        "copa_am_uplift_avg": ca_impact["uplift_observed"].mean() if len(ca_impact) > 0 else None,
        "wc2026_games":       len(wc_proj),
        "total_extra_mid":    wc_proj["extra_mid"].sum(),
    }])
    summary.to_csv(OUT_NYC / "summary_nyc.csv", index=False)

    print("\n=== NYC Tier 1 complete ===")
    print(wc_proj[["date", "matchup", "stage", "uplift_mid", "extra_mid"]].to_string(index=False))
    return wc_proj, uplift_df, ca_impact


if __name__ == "__main__":
    run()
