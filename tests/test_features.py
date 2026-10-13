"""Year-aware lags and growth features (src/build_features.py)."""
import numpy as np
import pandas as pd
import pytest

from build_features import KEYS, add_growth_features, lagged_values


def panel(rows):
    return pd.DataFrame(rows, columns=["region", "occ_code", "year", "employment_level", "wage_annual_mean"])


def test_lag_skips_missing_year_instead_of_spanning_it():
    # 2017 is absent, so 2018 has no prior-year value. A positional shift would
    # have returned 110 (the 2016 row) and called the 2016 -> 2018 change "YoY".
    df = panel([("US", "A", 2015, 100, 10), ("US", "A", 2016, 110, 11), ("US", "A", 2018, 150, 15)])
    lag = lagged_values(df, KEYS, ["employment_level"], 1)
    assert lag["employment_level"].tolist()[:2] == [pytest.approx(np.nan, nan_ok=True), 100]
    assert np.isnan(lag["employment_level"].iloc[2])


def test_lag_does_not_leak_across_occupations_or_regions():
    df = panel([("US", "A", 2015, 100, 10), ("US", "A", 2016, 110, 11),
                ("US", "B", 2016, 5, 1), ("FL", "A", 2016, 7, 2)])
    lag = lagged_values(df, KEYS, ["employment_level"], 1)
    assert lag["employment_level"].tolist()[1] == 100
    assert np.isnan(lag["employment_level"].iloc[2]) and np.isnan(lag["employment_level"].iloc[3])


def test_lag_keeps_the_callers_index_even_when_unsorted():
    df = panel([("US", "A", 2016, 110, 11), ("US", "A", 2015, 100, 10)])
    df.index = [7, 3]
    lag = lagged_values(df, KEYS, ["employment_level"], 1)
    assert list(lag.index) == [7, 3]
    assert lag.loc[7, "employment_level"] == 100 and np.isnan(lag.loc[3, "employment_level"])


def test_lag_rejects_duplicate_keys():
    df = panel([("US", "A", 2015, 100, 10), ("US", "A", 2015, 101, 10)])
    with pytest.raises(AssertionError):
        lagged_values(df, KEYS, ["employment_level"], 1)


def test_five_year_lag_matches_exactly_five_years_back():
    df = panel([("US", "A", y, 100 + y, 10) for y in (2015, 2016, 2020, 2021)])
    lag = lagged_values(df, KEYS, ["employment_level"], 5)
    by_year = dict(zip(df["year"], lag["employment_level"]))
    assert by_year[2020] == 2115 and by_year[2021] == 2116
    assert np.isnan(by_year[2016])


def test_growth_features_values_and_gaps():
    df = panel([("US", "A", 2015, 100, 50_000), ("US", "A", 2016, 95, 51_000),
                ("US", "A", 2017, 94, 52_000), ("US", "A", 2019, 120, 60_000)])
    out = add_growth_features(df).set_index("year")
    assert out.loc[2016, "employment_change_yoy_pct"] == pytest.approx(-5.0)
    assert out.loc[2016, "wage_growth_yoy_pct"] == pytest.approx(2.0)
    assert out.loc[2016, "contraction_flag_12mo"] == 1          # below -2%
    assert out.loc[2017, "contraction_flag_12mo"] == 0          # about -1.05%
    assert pd.isna(out.loc[2015, "contraction_flag_12mo"])       # no prior year
    assert pd.isna(out.loc[2019, "employment_change_yoy_pct"])   # 2018 is missing
    assert pd.isna(out.loc[2019, "contraction_flag_12mo"])


def test_five_year_cagr():
    df = panel([("US", "A", 2015, 100, 10), ("US", "A", 2020, 200, 20)])
    out = add_growth_features(df).set_index("year")
    assert out.loc[2020, "employment_change_5yr_cagr"] == pytest.approx(2 ** 0.2 - 1)


def test_committed_yoy_matches_an_independent_recomputation(occupation_features):
    f = occupation_features
    prev = f[["region", "occ_code", "year", "employment_level"]].copy()
    prev["year"] += 1
    merged = f.merge(prev, on=["region", "occ_code", "year"], how="left", suffixes=("", "_prev"))
    expected = (merged["employment_level"] / merged["employment_level_prev"] - 1) * 100
    pd.testing.assert_series_equal(merged["employment_change_yoy_pct"], expected, check_names=False, rtol=1e-9)
