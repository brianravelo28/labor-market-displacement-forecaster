"""
Fetch BLS CPI-U (All Urban Consumers, US city average, all items) monthly,
used to compute inflation-adjusted wages and real wage compression.

Series ID CUUR0000SA0 is a standard, well-known BLS series (unlike OEWS/JOLTS,
this one is a genuine continuous monthly time series going back decades), so
no per-year archive downloads are needed here -- a single API call covers the
whole 2015-2024 range.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_PROCESSED, JOLTS_RAW

from fetch_jolts import fetch_batch  # reuses the same cached/keyed API client (cache dir is shared, not JOLTS-specific)

CPI_SERIES_ID = "CUUR0000SA0"
START_YEAR = 2015
END_YEAR = 2024


def build_cpi_series() -> pd.DataFrame:
    data = fetch_batch([CPI_SERIES_ID], START_YEAR, END_YEAR)
    series = data["Results"]["series"][0]["data"]
    rows = [
        {"year": int(p["year"]), "month": int(p["period"][1:]), "cpi": float(p["value"])}
        for p in series
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
    print(cpi.groupby("year")["cpi_index_2020base"].mean())
