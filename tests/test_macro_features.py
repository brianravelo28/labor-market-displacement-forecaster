"""CPI real wages and JOLTS exposure (src/build_macro_features.py)."""
import numpy as np
import pandas as pd
import pytest

import build_macro_features as bmf
from config import JOLTS_INDUSTRY_SECTORS, OEWS_SECTOR_NAICS_CODES


@pytest.fixture
def flat_two_percent_cpi(monkeypatch):
    monkeypatch.setattr(bmf, "load_cpi_may", lambda: pd.Series({2015: 100.0, 2016: 102.0, 2017: 104.04}))


def wages(rows):
    return pd.DataFrame(rows, columns=["year", "wage_annual_mean", "wage_growth_yoy_pct"])


def test_real_wage_is_expressed_in_base_year_dollars(flat_two_percent_cpi):
    out = bmf.add_real_wage_features(wages([(2015, 1000.0, np.nan), (2017, 1000.0, 0.0)]))
    assert out["wage_annual_mean_real"].tolist() == pytest.approx([1040.4, 1000.0])  # base year = latest year


def test_real_wage_growth_subtracts_inflation(flat_two_percent_cpi):
    out = bmf.add_real_wage_features(wages([(2016, 1, 2.0), (2016, 1, 1.0), (2016, 1, -1.0)]))
    assert out["real_wage_growth_yoy_pct"].tolist() == pytest.approx([0.0, 100 * (1.01 / 1.02 - 1), 100 * (0.99 / 1.02 - 1)])


def test_purchasing_power_loss_needs_a_nominal_raise_that_lags_inflation(flat_two_percent_cpi):
    out = bmf.add_real_wage_features(wages([(2016, 1, 3.0), (2016, 1, 1.0), (2016, 1, -1.0), (2016, 1, np.nan)]))
    flag = out["purchasing_power_loss_flag"]
    assert flag.iloc[0] == 0      # raise beats inflation
    assert flag.iloc[1] == 1      # raise, but below inflation
    assert flag.iloc[2] == 0      # nominal cut is not a "raise that lagged"
    assert pd.isna(flag.iloc[3])  # unknown growth stays unknown


def test_real_wage_is_null_where_the_wage_is_null(flat_two_percent_cpi):
    out = bmf.add_real_wage_features(wages([(2016, np.nan, np.nan)]))
    assert out["wage_annual_mean_real"].isna().all()


def test_every_oews_sector_has_a_jolts_mapping_or_is_agriculture():
    unmapped = set(OEWS_SECTOR_NAICS_CODES) - set(bmf.SECTOR_TO_JOLTS)
    assert unmapped == {"11"}


def test_mapped_jolts_industries_are_ones_the_fetcher_pulls():
    assert set(bmf.SECTOR_TO_JOLTS.values()) <= set(JOLTS_INDUSTRY_SECTORS)


def jolts_stub(region, year=2020):
    """Annual JOLTS rates as annual_jolts() returns them: retail 4, food service 8, total 5 (+1 in Florida);
    quits and layoffs are half the openings rate."""
    rows = []
    for code, rate in (("440000", 4.0), ("720000", 8.0), ("000000", 5.0 + (1.0 if region == "FL" else 0.0))):
        for element, short in bmf.JOLTS_ELEMENTS.items():
            rows.append({"year": year, "industry_code": code, "data_element": element,
                         "rate": rate * (1 if short == "openings" else 0.5), "suffix": short})
    return pd.DataFrame(rows)


@pytest.fixture
def stub_jolts(monkeypatch):
    monkeypatch.setattr(bmf, "annual_jolts", lambda region="US": jolts_stub(region))


def test_occupation_exposure_is_the_employment_weighted_industry_average(stub_jolts):
    industry = pd.DataFrame([
        (2020, "44-45", "X", 300.0), (2020, "72", "X", 100.0),       # 75% retail, 25% food service
        (2020, "11", "Y", 500.0),                                     # agriculture: no JOLTS
        (2020, "44-45", "Z", 50.0), (2020, "72", "Z", np.nan),       # missing employment is ignored
    ], columns=["year", "naics", "occ_code", "employment_level"])
    occ = pd.DataFrame({"year": 2020, "region": ["US", "FL", "US", "US"], "occ_code": ["X", "X", "Y", "Z"]})
    out = bmf.add_occupation_jolts(occ, industry)
    assert len(out) == len(occ)
    assert out.loc[0, "jolts_exposure_openings_rate"] == pytest.approx(0.75 * 4 + 0.25 * 8)
    assert out.loc[0, "jolts_exposure_quits_rate"] == pytest.approx((0.75 * 4 + 0.25 * 8) / 2)
    assert out.loc[1, "jolts_exposure_openings_rate"] == pytest.approx(out.loc[0, "jolts_exposure_openings_rate"])  # FL uses national industry rates
    assert np.isnan(out.loc[2, "jolts_exposure_openings_rate"])
    assert out.loc[3, "jolts_exposure_openings_rate"] == pytest.approx(4.0)


def test_region_total_uses_the_regions_own_series(stub_jolts):
    industry = pd.DataFrame([(2020, "44-45", "X", 10.0)], columns=["year", "naics", "occ_code", "employment_level"])
    occ = pd.DataFrame({"year": 2020, "region": ["US", "FL"], "occ_code": "X"})
    out = bmf.add_occupation_jolts(occ, industry).set_index("region")
    assert out.loc["US", "jolts_region_openings_rate"] == pytest.approx(5.0)
    assert out.loc["FL", "jolts_region_openings_rate"] == pytest.approx(6.0)


def test_sector_rows_get_their_sectors_rates(stub_jolts):
    industry = pd.DataFrame({"year": 2020, "naics": ["44-45", "72", "11"], "occ_code": "X", "employment_level": 1.0})
    out = bmf.add_sector_jolts(industry)
    assert out["jolts_openings_rate"].iloc[:2].tolist() == [4.0, 8.0]
    assert np.isnan(out["jolts_openings_rate"].iloc[2])
