"""
Fetch BLS CPI-U (All Urban Consumers, US city average, all items, not seasonally
adjusted) monthly, used to compute inflation-adjusted (real) wages and real wage changes.

Series ID CUUR0000SA0 is a standard, well-known BLS series (unlike OEWS/JOLTS,
this one is a genuine continuous monthly time series going back decades), so
no per-year archive downloads are needed here. The unregistered API returns at
most 10 years per query, so 2015 to the present takes two cached queries.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_PROCESSED, JOLTS_RAW

from fetch_jolts import START_YEAR, END_YEAR, fetch_windows  # same cached API client (cache dir is shared, not JOLTS-specific)

CPI_SERIES_ID = "CUUR0000SA0"


def build_cpi_series() -> pd.DataFrame:
    points = fetch_windows([CPI_SERIES_ID], START_YEAR, END_YEAR)[CPI_SERIES_ID]
    rows = [
        {"year": int(p["year"]), "month": int(p["period"][1:]), "cpi": float(p["value"])}
        for p in points
        if p["period"] != "M13" and p["value"] not in ("-", "")
    ]
    df = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)

    # Rebase to 2020 = 100 for readability (2020 annual average as base).
    base = df[df["year"] == 2020]["cpi"].mean()
    df["cpi_index_2020base"] = df["cpi"] / base * 100
    return df


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    JOLTS_RAW.mkdir(parents=True, exist_ok=True)
    cpi = build_cpi_series()
    out_path = DATA_PROCESSED / "cpi_monthly.csv"
    cpi.to_csv(out_path, index=False)
    print(f"Saved {len(cpi):,} rows to {out_path}")
    print(cpi.groupby("year")["cpi_index_2020base"].agg(["count", "mean"]))
