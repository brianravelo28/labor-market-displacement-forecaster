"""Invariants on the committed data/processed files."""
import re

import pandas as pd
import pytest

from config import OEWS_SECTOR_NAICS_CODES, OEWS_YEARS

from conftest import PROCESSED, ROOT


def test_occupation_panel_has_one_row_per_key_and_every_year(occupation_features):
    f = occupation_features
    assert not f.duplicated(["year", "region", "occ_code"]).any()
    assert sorted(f["year"].unique()) == OEWS_YEARS
    assert set(f["region"]) == {"US", "FL"}
    assert (f["employment_level"].dropna() > 0).all()


def test_features_contain_the_panel_unchanged(occupation_features):
    keys = ["year", "region", "occ_code"]
    panel = pd.read_csv(PROCESSED / "oews_occupation_panel.csv").sort_values(keys).reset_index(drop=True)
    feats = occupation_features.sort_values(keys).reset_index(drop=True)
    assert feats[keys].equals(panel[keys])
    for col in panel.columns:
        assert feats[col].equals(panel[col]), col


def test_industry_panel_keys_and_sectors(industry_features):
    f = industry_features
    assert not f.duplicated(["year", "naics", "occ_code"]).any()
    assert set(f["naics"]) == set(OEWS_SECTOR_NAICS_CODES)
    assert sorted(f["year"].unique()) == OEWS_YEARS


def test_jolts_panel_shape_and_ranges():
    j = pd.read_csv(PROCESSED / "jolts_industry_panel.csv", dtype={"industry_code": str})
    # A rate is a percent of employment. The genuine April 2020 spikes (layoffs and separations
    # in leisure & hospitality reached 30%+) are the largest values, so 40 is a sanity ceiling, not a trim.
    assert j["rate"].between(0, 40).all()
    assert set(j["region"]) == {"US", "FL"}
    assert set(j[j["region"] == "FL"]["industry_code"]) == {"000000"}      # BLS publishes no Florida industry detail
    assert j[j["region"] == "US"]["industry_code"].nunique() == 16
    assert not j.duplicated(["region", "industry_code", "data_element", "year", "month"]).any()


def test_every_oews_year_has_a_may_cpi_value():
    cpi = pd.read_csv(PROCESSED / "cpi_monthly.csv")
    may = cpi[cpi["month"] == 5].set_index("year")["cpi"]
    assert set(OEWS_YEARS) <= set(may.index)
    assert may.loc[2025] > may.loc[2015]
    assert not cpi.duplicated(["year", "month"]).any()


def test_real_wage_equals_nominal_in_the_base_year(occupation_features):
    f = occupation_features
    base = f[f["year"] == f["year"].max()].dropna(subset=["wage_annual_mean"])
    assert (base["wage_annual_mean_real"] - base["wage_annual_mean"]).abs().max() < 1e-6


def test_public_docs_never_mention_the_private_quirks_file():
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), *(ROOT / "app").glob("*.py"), *(ROOT / "src").glob("*.py")]:
        if path.name == "WALKTHROUGH.md":      # local-only file, never committed
            continue
        assert not re.search(r"DATA_QUIRKS|data[- ]quirks", path.read_text(encoding="utf-8"), re.I), path
