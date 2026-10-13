"""Supervised-pair construction, scoring and walk-forward discipline (src/train_model.py)."""
import numpy as np
import pandas as pd
import pytest

import train_model as tm


def features(rows):
    cols = ["region", "occ_code", "year", "employment_level", "wage_annual_mean", "wage_annual_median",
            "employment_change_yoy_pct", "wage_growth_yoy_pct", "contraction_flag_12mo"]
    return pd.DataFrame([(r, o, y, e, 1.0, 1.0, 0.0, 0.0, 0.0) for r, o, y, e in rows], columns=cols)


def test_pairs_are_exactly_horizon_years_apart():
    f = features([("US", "11-1", y, 100 * (y - 2014)) for y in (2015, 2016, 2017, 2019)])
    h1 = tm.build_supervised(f)
    assert sorted(h1["year"]) == [2015, 2016]               # 2017 -> 2018 and 2018 -> 2019 don't exist
    h2 = tm.build_supervised(f, horizon=2)
    # 2015 -> 2017 is kept. 2016 -> 2018 has no 2018, and 2017 -> 2019 spans the missing 2018, so a pair is only
    # built when every year in between exists (conservative: it never invents a multi-year "next").
    assert sorted(h2["year"]) == [2015]


def test_target_is_the_log_growth_ratio_and_the_future_level():
    f = features([("US", "11-1", 2015, 100), ("US", "11-1", 2016, 150)])
    row = tm.build_supervised(f).iloc[0]
    assert row[tm.TARGET_COL] == 150
    assert row["target_log_growth"] == pytest.approx(np.log(1.5))


def test_pairs_do_not_cross_regions_or_occupations():
    f = features([("US", "11-1", 2015, 100), ("FL", "11-1", 2016, 100), ("US", "13-1", 2016, 100)])
    assert tm.build_supervised(f).empty


def test_rows_missing_a_model_input_are_dropped():
    f = features([("US", "11-1", 2015, 100), ("US", "11-1", 2016, 100)])
    f.loc[0, "wage_annual_median"] = np.nan
    assert tm.build_supervised(f).empty


def test_default_horizon_is_the_one_year_forecast(occupation_features):
    assert len(tm.build_supervised(occupation_features)) == len(tm.build_supervised(occupation_features, horizon=1))


def test_score_of_a_perfect_forecast_and_of_persistence():
    data = pd.DataFrame({tm.TARGET_COL: [110.0, 90.0, 200.0], "employment_level": [100.0, 100.0, 100.0]})
    perfect = tm.score(data, data[tm.TARGET_COL].values)
    assert perfect["rmse"] == 0 and perfect["median_ape"] == 0
    assert perfect["closer_than_persistence"] == 1.0
    persistence = tm.score(data, data["employment_level"].values)
    assert persistence["closer_than_persistence"] == 0.0
    assert persistence["rmse"] == pytest.approx(np.sqrt((100 + 100 + 10000) / 3))


def test_walk_forward_trains_only_on_targets_already_known(monkeypatch):
    seen = []

    class Stub:
        def predict(self, X):
            return np.zeros(len(X))

    def spy(train, params):
        seen.append(int(train["year"].max()))
        return Stub()

    monkeypatch.setattr(tm, "fit_model", spy)
    rows = {c: 1.0 for c in tm.FEATURE_COLS}
    data = pd.DataFrame([{**rows, "year": y, "employment_level": 100.0, tm.TARGET_COL: 105.0, "target_log_growth": 0.05}
                         for y in range(2016, 2025) for _ in range(3)])
    tm.walk_forward(data, [2020, 2021, 2022, 2023], params={})
    # forecasting from year T may use feature years <= T - 1 (their targets, year <= T, are known)
    assert seen == [2019, 2020, 2021, 2022]


def test_feature_sets_agree_everywhere(dashboard):
    # the dashboard rebuilds the model's input row by hand, so its list must match training exactly
    assert dashboard.MODEL_FEATURE_COLS == tm.FEATURE_COLS
