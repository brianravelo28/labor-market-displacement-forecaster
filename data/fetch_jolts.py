"""
Fetch monthly JOLTS (Job Openings and Labor Turnover Survey) rates: the US
by industry sector, and Florida for total nonfarm only (BLS publishes no
state-level industry detail).

JOLTS has no occupation-level breakdown -- series are keyed by NAICS industry
only (verified against https://download.bls.gov/pub/time.series/jt/jt.industry
and jt.dataelement). These industry-level rates get joined onto occupations
via a primary-industry mapping in the merge step, as a regional-tightness
proxy rather than a true occupation-specific vacancy rate.

Series ID format (post-Oct-2020 restructure), 21 chars -- verified by
character-index decomposition of a known-working series ID pulled live from
the API (JTS000000120000000JOR, Florida total-nonfarm job openings rate):
    JTS + industry_code(6) + state_code(2) + area_code(6) + sizeclass(1)
        + dataelement_code(2) + ratelevel_code(1)
National uses state_code "00". Note: a secondary source (GovEx-style tutorial
summary) described the area code field as 5 digits instead of 6 -- that was
wrong; trust the empirical decomposition above.
"""
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (
    BLS_API_KEY,
    BLS_API_URL,
    FL_STATE_FIPS,
    JOLTS_DATA_ELEMENTS,
    JOLTS_INDUSTRY_SECTORS,
    JOLTS_RAW,
    DATA_PROCESSED,
)

START_YEAR = 2015
END_YEAR = date.today().year
BATCH_SIZE = 25  # unregistered BLS API v2 limit: 25 series per query, 25 queries/day
WINDOW_YEARS = 10  # ...and at most 10 years per query, so 2015-2026 takes two windows
REGIONS = {"US": "00", "FL": FL_STATE_FIPS}


def build_series_ids():
    ids = {}
    for industry_code, industry_name in JOLTS_INDUSTRY_SECTORS.items():
        for region, state_code in REGIONS.items():
            # BLS publishes state-level JOLTS for total nonfarm only; industry
            # detail exists nationally. Florida retail, leisure & hospitality,
            # health care, construction and manufacturing were each queried and
            # return "Series does not exist" (checked 2026-10-12).
            if region != "US" and industry_code != "000000":
                continue
            for elem_code, elem_name in JOLTS_DATA_ELEMENTS.items():
                sid = f"JTS{industry_code}{state_code}0000000{elem_code}"
                ids[sid] = {
                    "industry_code": industry_code,
                    "industry_name": industry_name,
                    "region": region,
                    "data_element": elem_name,
                }
    return ids


def _cache_path(series_ids, start_year, end_year):
    key = f"{start_year}-{end_year}:" + ",".join(series_ids)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return JOLTS_RAW / f"batch_{digest}.json"

def fetch_batch(series_ids, start_year=START_YEAR, end_year=END_YEAR):
    cache_file = _cache_path(series_ids, start_year, end_year)
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if BLS_API_KEY:
        payload["registrationkey"] = BLS_API_KEY
    resp = requests.post(BLS_API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] not in ("REQUEST_SUCCEEDED", "REQUEST_SUCCEEDED_MESSAGES"):
        raise RuntimeError(f"BLS API error: {data.get('message')}")

    # Cache before validating, so a bad series ID costs one query, not one per rerun.
    cache_file.write_text(json.dumps(data))
    return data


def check_no_empty_series(data):
    """The API reports unknown series as "success" with an empty data list, so
    check for that explicitly instead of trusting the status field."""
    empty = [s["seriesID"] for s in data["Results"]["series"] if not s["data"]]
    if empty:
        raise RuntimeError(f"BLS returned no data for {len(empty)} series (first: {empty[0]}); messages: {data.get('message')}")


def fetch_windows(series_ids, start_year=START_YEAR, end_year=END_YEAR):
    """fetch_batch over consecutive <=10-year windows; returns {series_id: [data points]}."""
    points = {sid: [] for sid in series_ids}
    for w_start in range(start_year, end_year + 1, WINDOW_YEARS):
        data = fetch_batch(series_ids, w_start, min(w_start + WINDOW_YEARS - 1, end_year))
        check_no_empty_series(data)
        for series in data["Results"]["series"]:
            points[series["seriesID"]].extend(series["data"])
    return points


def build_panel() -> pd.DataFrame:
    meta = build_series_ids()
    all_ids = list(meta.keys())
    rows = []

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]
        print(f"Fetching JOLTS series {i + 1}-{i + len(batch)} of {len(all_ids)} ...")
        fetched = fetch_windows(batch)
        for sid, series_data in fetched.items():
            info = meta[sid]
            for point in series_data:
                if point["period"] == "M13":  # annual average, skip
                    continue
                value = point["value"]
                if value in ("-", ""):
                    continue
                rows.append(
                    {
                        "year": int(point["year"]),
                        "month": int(point["period"][1:]),
                        "region": info["region"],
                        "industry_code": info["industry_code"],
                        "industry_name": info["industry_name"],
                        "data_element": info["data_element"],
                        "rate": float(value),
                    }
                )
        if i + BATCH_SIZE < len(all_ids):
            time.sleep(1)

    df = pd.DataFrame(rows)
    return df.sort_values(["region", "industry_code", "data_element", "year", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    JOLTS_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    out_path = DATA_PROCESSED / "jolts_industry_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nSaved {len(panel):,} rows to {out_path}")
    print(panel.groupby(["region", "data_element"])["rate"].count())
