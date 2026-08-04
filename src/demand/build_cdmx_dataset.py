"""
build_cdmx_dataset.py - Build data/cdmx/metro/daily_totals.parquet from the raw
SEMOVI STC Metro ridership CSV.

Run this ONCE after downloading the raw SEMOVI CSV (see data/cdmx/README.md).

Usage:
    python src/demand/build_cdmx_dataset.py --input data/cdmx/afluencia/afluenciastc_simple_12_2025.csv

The raw CSV can be downloaded from SEMOVI's open-data portal:
    https://datos.cdmx.gob.mx/dataset/afluencia-diaria-del-metro-cdmx

Outputs:
    data/cdmx/metro/daily_totals.parquet
        columns: date, sid, riders, fecha, is_event, event_category, year, month, dow

Station-name matching (important, do not "clean up" without re-verifying):
    The raw CSV's `estacion` column contains full, correctly-accented Spanish
    station names (once its mojibake encoding is fixed - see `fix_mojibake`
    below). `data/cdmx/metro/station_lookup.csv`, however, was built by some
    earlier, undocumented process that DROPPED every non-ASCII character
    entirely instead of replacing it with its unaccented base letter (e.g.
    "Cuauhtémoc" -> "cuauhtmoc", not "cuauhtemoc"; "Gómez Farías" -> two
    different rows, "gmez faras" and "gmez farias", because the raw source
    itself changed how it spelled that station partway through 2023). This
    script reproduces that exact (buggy) normalization in `ascii_drop` so
    that `sid` assignments match `station_lookup.csv` and, transitively,
    every other output already computed against those `sid` values (Table 13,
    the WC2026 real-ridership validation, etc.). Do NOT "fix" the
    station-name normalization to be linguistically correct; doing so will
    silently break the sid mapping used everywhere else.

Verification status (checked 2026-08-04, see data/cdmx/README.md for detail):
    Every one of the 545,438 (date, sid) rows in the previously committed
    daily_totals.parquet has an EXACT match here (riders, is_event, and
    event_category all identical) - zero discrepancies. This script's raw
    CSV export (December 2025) is more complete than whatever export
    produced the committed file, though: it additionally covers 2020 and
    2021 (this script excludes both by default, see EXCLUDE_YEARS below,
    since the committed file clearly omits the pandemic-disrupted period
    entirely) plus roughly 25,000 isolated per-station-day rows scattered
    elsewhere in the committed file's history whose absence there is NOT
    explained by any window or filter tried so far. This means: (a) every
    number this script CAN verify against matches exactly, but (b) a fresh
    run will not be byte-identical to the committed parquet, it will have
    a small number of extra rows. Do not treat this script's raw output as
    an automatic drop-in replacement for the committed file without
    re-running the downstream Table 13 / WC2026 validation and confirming
    the extra rows do not change the results.

Known gap, not solved by this script:
    data/cdmx/metro/baseline_daily.parquet (median expected ridership by
    sid x day-of-week) could NOT be reproduced from this raw CSV. Its values
    do not match any median/mean computed over any tested date window
    (full history, post-COVID, pre-COVID, 2020/2021-excluded, event-excluded,
    zero-excluded, trailing N-week windows), and at least one value (sid=1,
    dow=0: 12297) is not even a real observed daily total in this raw file
    for that station/weekday - meaning baseline_daily.parquet was likely
    built from a different (possibly older) vintage of the SEMOVI export
    that is no longer available. It remains committed to the repository
    as-is, since there is currently no way to regenerate it.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LOOKUP_PATH = ROOT / "data/cdmx/metro/station_lookup.csv"
CATALOG_PATH = ROOT / "data/cdmx/event_catalog_2016_2025.csv"
OUT_PATH = ROOT / "data/cdmx/metro/daily_totals.parquet"

DATE_START = "2016-01-01"
DATE_END = "2025-12-31"
# The committed daily_totals.parquet excludes 2020-2021 entirely (pandemic
# ridership collapse). Match that by default; pass --include-covid-years to
# keep them (e.g. for a from-scratch, non-reproduction-focused pipeline).
EXCLUDE_YEARS = [2020, 2021]

COL_DATE = "fecha"
COL_LINE = "linea"
COL_STATION = "estacion"
COL_RIDERS = "afluencia"


def fix_mojibake(x):
    """Raw CSV is UTF-8 text that was previously mis-decoded as latin1
    somewhere upstream (e.g. "CatÃ³lica" instead of "Católica"). Undo it."""
    if not isinstance(x, str):
        return x
    try:
        return x.encode("latin1").decode("utf-8")
    except Exception:
        return x


def ascii_drop(s):
    """Reproduces station_lookup.csv's legacy normalization: DROP any
    non-ASCII character entirely (do not replace with a base letter)."""
    if not isinstance(s, str):
        return s
    return "".join(c for c in s if ord(c) < 128)


def strip_accents_replace(s):
    """Correct accent handling (á -> a, etc.), used only for the `linea`
    column, which station_lookup.csv normalized correctly (unlike `estacion`)."""
    import unicodedata
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def main(input_csv: Path):
    print(f"Loading raw SEMOVI CSV: {input_csv}")
    raw = pd.read_csv(input_csv, encoding="utf-8", low_memory=False)

    raw["estacion_norm"] = (
        raw[COL_STATION].map(fix_mojibake).map(ascii_drop).str.lower().str.strip()
    )
    raw["linea_norm"] = (
        raw[COL_LINE].map(fix_mojibake).map(strip_accents_replace).str.lower().str.strip()
    )
    raw["fecha"] = pd.to_datetime(raw[COL_DATE], errors="coerce")
    raw[COL_RIDERS] = pd.to_numeric(raw[COL_RIDERS], errors="coerce")

    lookup = pd.read_csv(LOOKUP_PATH)
    merged = raw.merge(
        lookup[["sid", "name", "line"]],
        left_on=["linea_norm", "estacion_norm"],
        right_on=["line", "name"],
        how="inner",
    )
    n_unmatched = len(raw) - len(merged)
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} raw rows did not match any station_lookup.csv entry")

    merged["date"] = merged["fecha"].dt.strftime("%Y-%m-%d")
    daily = (
        merged.groupby(["date", "sid"])[COL_RIDERS]
        .sum()
        .reset_index()
        .rename(columns={COL_RIDERS: "riders"})
    )
    daily = daily[(daily["date"] >= DATE_START) & (daily["date"] <= DATE_END)].copy()
    daily["fecha"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["fecha"].dt.year.astype("int32")
    daily["month"] = daily["fecha"].dt.month.astype("int32")
    daily["dow"] = daily["fecha"].dt.dayofweek.astype("int32")

    if EXCLUDE_YEARS:
        n_before = len(daily)
        daily = daily[~daily["year"].isin(EXCLUDE_YEARS)].copy()
        print(f"  Excluded {n_before - len(daily):,} rows from years {EXCLUDE_YEARS} (pandemic-disrupted ridership)")

    # is_event / event_category: broadcast the curated catalog's per-day
    # event flag to every station on that date (citywide flag, not
    # catchment-specific - matches the previously committed file).
    catalog = pd.read_csv(CATALOG_PATH)
    catalog["date"] = pd.to_datetime(catalog["fecha"]).dt.strftime("%Y-%m-%d")
    cat_map = catalog.drop_duplicates("date").set_index("date")["event_category"]

    daily["is_event"] = daily["date"].isin(cat_map.index).astype("int64")
    daily["event_category"] = daily["date"].map(cat_map).fillna("none")
    daily["sid"] = daily["sid"].astype("int64")
    daily["riders"] = daily["riders"].astype("int64")

    daily = daily[["date", "sid", "riders", "fecha", "is_event", "event_category", "year", "month", "dow"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(daily):,} rows -> {OUT_PATH}")
    print(f"  Date range: {daily['date'].min()} .. {daily['date'].max()}")
    print(f"  Stations: {daily['sid'].nunique()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                    help="Path to the raw SEMOVI afluenciastc_*.csv file")
    args = p.parse_args()
    main(args.input)
