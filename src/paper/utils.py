"""
Shared utility functions for the WC2026 transit paper pipeline.
"""

import numpy as np
import pandas as pd


def robust_zscore_by_group(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    MAD-based robust z-score, computed within each group.

    Using the median and median absolute deviation prevents extreme event days
    from inflating the baseline standard deviation and suppressing true shocks.
    Normalising by group (e.g. day-of-week) removes weekday seasonality so
    Saturday spikes are not masked by Saturday's naturally higher ridership.

    Args:
        series: Numeric series to normalise.
        group:  Group labels of the same length (e.g. day-of-week integers).

    Returns:
        Series of the same index with robust z-scores.
    """
    def _mad_zscore(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median()
        if mad < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - med) / (1.4826 * mad)   # 1.4826 = 1/Φ⁻¹(0.75): consistency factor, MAD → σ

    return series.groupby(group).transform(_mad_zscore)


def uplift_distribution(
    daily_series:    pd.Series,
    baseline_series: pd.Series,
    top_n:           int   = 30,
    percentiles:     tuple = (25, 50, 75),
) -> dict:
    """
    Compute the uplift distribution for the top-N highest-demand days.

    Uplift is defined as observed / expected, where expected comes from a
    day-of-week baseline.  The distribution of uplifts for the top-N days
    provides scenario bounds (low / mid / high) for WC2026 projections.

    Args:
        daily_series:    Observed daily ridership, indexed by date.
        baseline_series: Expected ridership for each date (same index).
        top_n:           Number of shock days to include.
        percentiles:     Percentile thresholds for scenario bounds.

    Returns:
        Dict with keys:
          uplift_low, uplift_mid, uplift_high  (float multipliers)
          top_days   (DataFrame: date, uplift)
          percentile_values  (dict mapping percentile -> value)
    """
    uplift      = daily_series / baseline_series.clip(lower=1)
    top_uplifts = uplift.nlargest(top_n).dropna()

    p_low, p_mid, p_high = np.percentile(top_uplifts.values, list(percentiles))

    return {
        "uplift_low":  float(p_low),
        "uplift_mid":  float(p_mid),
        "uplift_high": float(p_high),
        "top_days": pd.DataFrame({
            "date":   top_uplifts.index,
            "uplift": top_uplifts.values,
        }),
        "percentile_values": {
            p: v for p, v in zip(percentiles, [p_low, p_mid, p_high])
        },
    }


def project_wc2026(
    baseline:    float,
    uplift_low:  float,
    uplift_mid:  float,
    uplift_high: float,
) -> dict:
    """
    Apply uplift multipliers to a baseline to produce three WC2026 scenarios.

    Returns:
        Dict with baseline, projected ridership (low/mid/high), and
        additional riders above baseline for each scenario.
    """
    return {
        "baseline":   baseline,
        "proj_low":   baseline * uplift_low,
        "proj_mid":   baseline * uplift_mid,
        "proj_high":  baseline * uplift_high,
        "extra_low":  baseline * (uplift_low  - 1),
        "extra_mid":  baseline * (uplift_mid  - 1),
        "extra_high": baseline * (uplift_high - 1),
    }


def weekday_es_to_int(weekday_es: str) -> int:
    """
    Convert a Spanish weekday name to a Python day-of-week integer (0=Monday).

    Used when parsing CDMX SEMOVI ridership files, which label days in Spanish.
    Returns -1 if the name is unrecognised.
    """
    mapping = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3,
        "viernes": 4, "sábado": 5, "domingo": 6,
    }
    return mapping.get(weekday_es.lower(), -1)


def date_to_weekday_es(date_str: str) -> str:
    """
    Return the Spanish weekday name for an ISO date string.

    Used when constructing CDMX output labels that match the source data format.
    """
    names = [
        "lunes", "martes", "miércoles", "jueves",
        "viernes", "sábado", "domingo",
    ]
    return names[pd.Timestamp(date_str).dayofweek]


def normalize_linea(linea: str) -> str:
    """
    Normalise CDMX Metro/Metrobús line name variants.

    The SEMOVI dataset uses inconsistent capitalisation and accent marks
    (e.g. 'Línea 2', 'linea_2', 'linea 2').  This function maps all forms
    to a canonical 'Linea N' format for reliable groupby/join operations.
    """
    return (
        linea.strip()
             .replace("Línea", "Linea")
             .replace("linea_", "Linea")
             .replace("linea ", "Linea ")
             .replace("  ", " ")
    )
