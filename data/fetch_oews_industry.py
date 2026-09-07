"""
Extract national-level OEWS employment/wage by occupation x industry sector
(2-digit/range NAICS), for Tab 3 (Industry & Sector Analysis).

Reuses the already-downloaded oesm{YY}all.zip archives from fetch_oews.py --
no new downloads or BLS API calls needed. Sector rows are identified by
matching NAICS values directly against OEWS_SECTOR_NAICS_CODES rather than
filtering on the i_group column, because i_group doesn't exist at all in
2015-2018 releases (verified while building fetch_oews.py); the NAICS sector
code strings themselves ('11', '44-45', etc.) are standardized and stable
across years regardless of that column-naming drift.

National-level only: state x sector x detailed-occupation combinations are
too sparse/suppressed in OEWS to build a reliable FL-specific sector panel.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_PROCESSED, OEWS_SECTOR_NAICS_CODES, OEWS_YEARS
from fetch_oews import NUMERIC_COLUMNS, load_year_raw

NATIONAL_AREA_TYPE = 1


def parse_year_industry(year: int) -> pd.DataFrame:
    df = load_year_raw(year)
    df = df[
        (df["o_group"].isin(["detailed", "detail"]))
        & (df["naics"].isin(OEWS_SECTOR_NAICS_CODES))
        & (df["area_type"] == NATIONAL_AREA_TYPE)
    ]
    cols = ["year", "naics", "occ_code", "occ_title"] + NUMERIC_COLUMNS
    if "naics_title" in df.columns:
        cols.insert(2, "naics_title")
    return df[cols]


def build_industry_panel(years=OEWS_YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        print(f"Parsing OEWS industry-sector data {year} ...")
        frames.append(parse_year_industry(year))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={
        "tot_emp": "employment_level",
        "h_mean": "wage_hourly_mean",
        "a_mean": "wage_annual_mean",
        "h_median": "wage_hourly_median",
        "a_median": "wage_annual_median",
        "a_pct10": "wage_annual_pct10",
        "a_pct90": "wage_annual_pct90",
    })
    return panel


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = build_industry_panel()
    out_path = DATA_PROCESSED / "oews_industry_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nSaved {len(panel):,} rows to {out_path}")
    print(panel.groupby("year")["naics"].nunique())
    if "naics_title" in panel.columns:
        print(panel[["naics", "naics_title"]].drop_duplicates().sort_values("naics").to_string(index=False))
