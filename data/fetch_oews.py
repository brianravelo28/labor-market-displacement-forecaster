"""
Download and parse BLS OEWS annual "All Data" archives into a single
occupation x year panel (national + Florida).

OEWS is published annually (reference period May), not as a continuous
multi-year timeseries under stable series IDs -- see README.md for why this
downloads per-year archives instead of using the BLS timeseries API.

Column names and casing vary by release year (verified: 2015 uses lowercase
with spaces e.g. "occ code"; 2019 uses lowercase with underscores; 2024 uses
UPPERCASE_WITH_UNDERSCORES; PRIM_STATE doesn't exist before ~2020). This
script normalizes headers before extracting fields, and identifies Florida
via AREA_TYPE==2 and AREA==12 (FL FIPS code), which is stable across years,
rather than relying on PRIM_STATE.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BLS_USER_AGENT, OEWS_ARCHIVE_URL, OEWS_RAW, OEWS_YEARS, DATA_PROCESSED

NATIONAL_AREA_TYPE = 1
STATE_AREA_TYPE = 2
FLORIDA_AREA_CODE = 12  # FIPS code, stable across years regardless of PRIM_STATE column

# canonical_name -> possible header variants seen across release years
COLUMN_SYNONYMS = {
    "area": ["area"],
    "area_title": ["area_title"],
    "area_type": ["area_type"],
    "naics": ["naics"],
    "naics_title": ["naics_title"],
    "occ_code": ["occ_code", "occ code"],
    "occ_title": ["occ_title", "occ title"],
    "o_group": ["o_group", "group"],
    "tot_emp": ["tot_emp"],
    "h_mean": ["h_mean"],
    "a_mean": ["a_mean"],
    "h_median": ["h_median"],
    "a_median": ["a_median"],
    "a_pct10": ["a_pct10"],
    "a_pct90": ["a_pct90"],
}

NUMERIC_COLUMNS = ["tot_emp", "h_mean", "a_mean", "h_median", "a_median", "a_pct10", "a_pct90"]


def _normalize_header(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[\s]+", "_", col)
    return col


def download_year(year: int) -> "zipfile.ZipFile":
    yy = f"{year % 100:02d}"
    dest = OEWS_RAW / f"oesm{yy}all.zip"
    if not dest.exists():
        url = OEWS_ARCHIVE_URL.format(yy=yy)
        print(f"  downloading {url} ...")
        resp = requests.get(url, headers={"User-Agent": BLS_USER_AGENT}, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return zipfile.ZipFile(dest)


def load_year_raw(year: int) -> pd.DataFrame:
    """Download (if needed) and return the normalized, full raw OEWS table for a year."""
    zf = download_year(year)
    xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
    with zf.open(xlsx_name) as f:
        raw = pd.read_excel(io.BytesIO(f.read()))

    raw.columns = [_normalize_header(c) for c in raw.columns]

    reverse_map = {}
    for canonical, variants in COLUMN_SYNONYMS.items():
        for v in variants:
            if v in raw.columns:
                reverse_map[v] = canonical
                break
    df = raw.rename(columns=reverse_map)

    required = [c for c in COLUMN_SYNONYMS if c != "naics_title"]  # naics_title missing in some years, non-essential
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OEWS {year}: missing expected columns after normalization: {missing}")

    df["naics"] = df["naics"].astype(str)
    df["o_group"] = df["o_group"].astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = year
    return df


def parse_year(year: int) -> pd.DataFrame:
    df = load_year_raw(year)
    # 2015-era files truncate the value to "detail" instead of "detailed"
    df = df[(df["o_group"].isin(["detailed", "detail"])) & (df["naics"].isin(["000000", "0"]))]
    df = df[df["area_type"].isin([NATIONAL_AREA_TYPE, STATE_AREA_TYPE])]
    df = df[~((df["area_type"] == STATE_AREA_TYPE) & (df["area"] != FLORIDA_AREA_CODE))]

    df["region"] = df["area_type"].map({NATIONAL_AREA_TYPE: "US", STATE_AREA_TYPE: "FL"})

    return df[["year", "region", "occ_code", "occ_title"] + NUMERIC_COLUMNS]


def build_panel(years=OEWS_YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        print(f"Parsing OEWS {year} ...")
        frames.append(parse_year(year))
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
    OEWS_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    out_path = DATA_PROCESSED / "oews_occupation_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nSaved {len(panel):,} rows to {out_path}")
    print(panel.groupby("year")["occ_code"].nunique())
