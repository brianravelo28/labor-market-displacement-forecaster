"""Dashboard callbacks and helpers (app/app.py), run against the committed data."""
import numpy as np
import pandas as pd
import pytest

RETAIL = "41-2031"


def text(component):
    return component.children[1].children      # kpi_card: [label, value, sub]


def test_layout_has_the_four_tabs(dashboard):
    assert len(dashboard.app.layout.children[2].children) == 4


def test_header_years_come_from_the_data(dashboard, occupation_features):
    assert dashboard.FIRST_YEAR == occupation_features["year"].min()
    assert dashboard.LATEST_YEAR == occupation_features["year"].max()


# ---- Tab 1 -----------------------------------------------------------------

@pytest.mark.parametrize("region", ["US", "FL"])
def test_occupation_tab_renders_kpis_and_both_wage_series(dashboard, region):
    kpis, emp_fig, wage_fig = dashboard.update_dashboard(RETAIL, region)
    assert len(kpis) == 7
    assert [t.name for t in wage_fig.data] == ["Nominal", f"Real ({dashboard.LATEST_YEAR} dollars)"]
    assert len(emp_fig.data[0].x) >= 10


def test_occupation_tab_handles_a_missing_selection(dashboard):
    kpis, emp_fig, wage_fig = dashboard.update_dashboard("00-0000", "US")
    assert text(kpis[0]) == "--"


def test_real_wage_kpi_is_red_when_the_real_wage_fell(dashboard, occupation_features):
    f = occupation_features
    row = f[(f["year"] == f["year"].max()) & (f["region"] == "US") & (f["real_wage_growth_yoy_pct"] < 0)].iloc[0]
    kpis, *_ = dashboard.update_dashboard(row["occ_code"], "US")
    real_kpi = next(k for k in kpis if k.children[0].children == "Real wage YoY")
    assert real_kpi.children[1].style["color"] == "#c0392b"


# ---- Tab 2 -----------------------------------------------------------------

def test_regional_tab_returns_six_figures_with_two_bars_each(dashboard):
    figs = dashboard.update_regional_tab("tab-2")
    assert len(figs) == 6
    assert all(len(f.data[0].y) == 2 for f in figs)


def test_share_negative_ignores_missing_values(dashboard):
    assert dashboard.share_negative(pd.Series([-1.0, 1.0, np.nan, np.nan])) == 50.0
    assert dashboard.share_negative(pd.Series([np.nan])) != dashboard.share_negative(pd.Series([np.nan]))  # NaN, not 0


# ---- Tab 3 -----------------------------------------------------------------

def test_sector_ranking_floor_applies_to_both_years(dashboard, industry_features):
    for naics in ["72", "62", "44-45"]:
        _, _, contracting, growth = dashboard.update_sector_tab(naics)
        for table in (contracting, growth):
            for r in table.data:
                prior = r["employment_level"] / (1 + r["employment_change_yoy_pct"] / 100)
                assert r["employment_level"] >= dashboard.SECTOR_RANKING_MIN_EMPLOYMENT
                assert round(prior) >= dashboard.SECTOR_RANKING_MIN_EMPLOYMENT


def test_sector_tab_explains_the_missing_jolts_series_for_agriculture(dashboard):
    cards, *_ = dashboard.update_sector_tab("11")
    assert "nonfarm" in cards[0].children


def test_sector_tab_shows_three_jolts_rates(dashboard):
    cards, *_ = dashboard.update_sector_tab("62")
    assert [c.children[0].children for c in cards] == ["Job openings rate", "Quits rate", "Layoffs & discharges rate"]


# ---- Tab 4 -----------------------------------------------------------------

def test_baseline_forecast_for_a_large_occupation_is_plausible(dashboard):
    b = dashboard.predict_baseline(RETAIL, "US")
    assert -30 < b["baseline_growth_pct"] < 30
    assert b["baseline_employment_next"] == pytest.approx(b["current_employment"] * (1 + b["baseline_growth_pct"] / 100))


def test_no_forecast_when_an_input_is_missing(dashboard):
    df = dashboard.df
    latest = df[df["year"] == df["year"].max()]
    inputs = ["employment_level", "wage_annual_mean", "wage_annual_median", "employment_change_yoy_pct",
              "wage_growth_yoy_pct", "contraction_flag_12mo"]
    incomplete = latest[latest[inputs].isna().any(axis=1)].iloc[0]
    assert dashboard.predict_baseline(incomplete["occ_code"], incomplete["region"]) is None


def test_the_scenario_tab_shows_growth_numbers_not_risk_labels(dashboard):
    kpis, fig, summary = dashboard.update_scenario_tab(RETAIL, "US", 1)
    labels = [k.children[0].children for k in kpis]
    assert labels == ["Baseline forecast growth", "Scenario forecast growth", "2020 response"]
    assert not hasattr(dashboard, "risk_level")
    assert "risk" not in summary.lower() and "forecast growth for" in summary and "moves from" in summary


def test_growth_numbers_are_red_when_negative_and_green_otherwise(dashboard):
    assert dashboard.growth_color(-0.1) == "#c0392b" and dashboard.growth_color(0.0) == "#27ae60"


def test_scenario_is_baseline_plus_severity_times_the_2020_response(dashboard):
    response, source = dashboard.shock_for(RETAIL, "US")
    base_kpis, *_ = dashboard.update_scenario_tab(RETAIL, "US", 0)
    kpis, fig, summary = dashboard.update_scenario_tab(RETAIL, "US", 0.5)
    base = float(text(base_kpis[0]).rstrip("%"))
    assert float(text(base_kpis[1]).rstrip("%")) == pytest.approx(base)          # severity 0 changes nothing
    assert float(text(kpis[1]).rstrip("%")) == pytest.approx(base + 0.5 * response, abs=0.11)
    assert len(fig.data[0].y) == 3


def test_full_severity_lowers_a_hard_hit_occupation_but_barely_moves_a_resilient_one(dashboard):
    waiters = dashboard.update_scenario_tab("35-3031", "US", 1)[0]
    nurses = dashboard.update_scenario_tab("29-1141", "US", 1)[0]
    assert float(text(waiters[1]).rstrip("%")) < float(text(waiters[0]).rstrip("%")) - 15
    assert abs(float(text(nurses[1]).rstrip("%")) - float(text(nurses[0]).rstrip("%"))) < 5


def test_scenario_shows_where_the_response_came_from(dashboard):
    big = dashboard.update_scenario_tab(RETAIL, "US", 1)[0]
    assert "own 2020 change" in big[2].children[2].children
    small_fl = dashboard.shock_response[(dashboard.shock_response["source"] == "group")].index[0]
    kpis = dashboard.update_scenario_tab(small_fl[1], small_fl[0], 1)[0]
    assert "occupation group" in kpis[2].children[2].children


def test_scenario_employment_never_goes_below_zero(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "shock_for", lambda occ, region: (-500.0, "occupation"))
    kpis, fig, summary = dashboard.update_scenario_tab(RETAIL, "US", 1)
    assert fig.data[0].y[2] >= 0


def test_scenario_without_a_2020_response_still_shows_the_baseline(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "shock_for", lambda occ, region: None)
    kpis, fig, summary = dashboard.update_scenario_tab(RETAIL, "US", 1)
    assert len(kpis) == 1 and len(fig.data[0].y) == 2 and "No 2020 response" in summary


def test_scenario_with_no_forecast_says_why(dashboard):
    kpis, fig, summary = dashboard.update_scenario_tab("00-0000", "US", 0)
    assert kpis == [] and summary.startswith("No forecast")


def test_tab_has_one_slider_and_no_invented_adjustments(dashboard):
    ids = {getattr(c, "id", None) for c in dashboard.app.layout._traverse()}
    assert "severity-slider" in ids
    assert not {"gdp-slider", "automation-slider", "immigration-slider"} & ids


def test_accuracy_note_quotes_the_training_metrics(dashboard):
    import json

    h = json.loads((dashboard.PROCESSED_DIR / "model_metrics.json").read_text())["holdout"]
    assert f"{h['model']['median_ape']:.1f}%" in dashboard.BASELINE_ACCURACY_NOTE


# ---- chart hints -----------------------------------------------------------

def test_legend_sentence_only_when_the_chart_has_a_legend(dashboard):
    assert "legend" not in dashboard.chart_hint().children
    assert "legend" in dashboard.chart_hint(has_legend=True).children


# ---- Tab 2: occupation vs worker weighting -----------------------------------

def test_weighted_quantile_with_equal_weights_is_the_ordinary_median(dashboard):
    s = pd.Series([5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
    ones = pd.Series(np.ones(len(s)))
    assert dashboard.weighted_quantile(s, ones, 0.5) == pytest.approx(s.median())
    assert dashboard.weighted_quantile(s.iloc[:5], ones.iloc[:5], 0.5) == pytest.approx(s.iloc[:5].median())


def test_weighted_quantile_follows_the_weight(dashboard):
    values = pd.Series([1.0, 2.0, 3.0])
    assert dashboard.weighted_quantile(values, pd.Series([1.0, 1.0, 98.0]), 0.5) == pytest.approx(3.0, abs=0.05)  # interpolated, so near 3
    assert dashboard.weighted_quantile(values, pd.Series([98.0, 1.0, 1.0]), 0.5) == pytest.approx(1.0, abs=0.05)


def test_weighted_quantile_ignores_missing_values_and_weights(dashboard):
    values = pd.Series([1.0, np.nan, 3.0, 100.0])
    weights = pd.Series([1.0, 5.0, 1.0, np.nan])
    assert dashboard.weighted_quantile(values, weights, 0.5) == pytest.approx(2.0)
    assert np.isnan(dashboard.weighted_quantile(pd.Series([np.nan]), pd.Series([1.0]), 0.5))


def test_weighted_share_counts_workers_not_occupations(dashboard):
    values = pd.Series([-1.0, 1.0])
    assert dashboard.share_negative(values) == 50.0                              # one of two occupations
    assert dashboard.share_negative(values, pd.Series([3.0, 1.0])) == 75.0       # three of four workers
    assert dashboard.share_where(pd.Series([1, 0, np.nan]), lambda s: s == 1) == 50.0


def regional_numbers(dashboard, weighting):
    figs = dashboard.update_regional_tab("tab-2", weighting)
    return [tuple(f.data[0].y) for f in figs], [f.layout.title.text for f in figs]


def test_default_view_is_by_occupation_and_matches_plain_pandas(dashboard, occupation_features):
    f = occupation_features
    latest = f[f["year"] == f["year"].max()]
    both = set(latest[latest["region"] == "US"]["occ_code"]) & set(latest[latest["region"] == "FL"]["occ_code"])
    us, fl = (latest[(latest["region"] == r) & latest["occ_code"].isin(both)] for r in ("US", "FL"))
    ys, _ = regional_numbers(dashboard, "occupation")
    assert ys[0] == pytest.approx((us["employment_change_yoy_pct"].median(), fl["employment_change_yoy_pct"].median()))
    assert ys[1] == pytest.approx((us["wage_annual_mean"].median(), fl["wage_annual_mean"].median()))
    assert dashboard.update_regional_tab("tab-2")[1].data[0].y == tuple(ys[1])      # the default argument is "occupation"


def test_worker_view_weights_each_region_by_its_own_employment(dashboard, occupation_features):
    f = occupation_features
    latest = f[f["year"] == f["year"].max()]
    both = set(latest[latest["region"] == "US"]["occ_code"]) & set(latest[latest["region"] == "FL"]["occ_code"])
    fl = latest[(latest["region"] == "FL") & latest["occ_code"].isin(both)].dropna(subset=["wage_annual_mean", "employment_level"])
    fl = fl.sort_values("wage_annual_mean")
    cumulative = fl["employment_level"].cumsum() / fl["employment_level"].sum()
    median_worker_wage = fl["wage_annual_mean"][cumulative >= 0.5].iloc[0]       # the occupation holding the 50th-percentile worker
    ys, _ = regional_numbers(dashboard, "worker")
    assert ys[1][1] == pytest.approx(median_worker_wage, rel=0.02)
    assert ys[1][1] < regional_numbers(dashboard, "occupation")[0][1][1]          # workers cluster in lower-paid occupations


def test_worker_view_relabels_charts_and_leaves_job_openings_alone(dashboard):
    ys_o, titles_o = regional_numbers(dashboard, "occupation")
    ys_w, titles_w = regional_numbers(dashboard, "worker")
    assert all("worker" in t.lower() for t in titles_w[:5])
    assert not any("worker" in t.lower() for t in titles_o)
    assert ys_o[5] == ys_w[5]
    assert all(len(t) <= 60 for t in titles_w)                                    # long titles clipped in this grid before


# ---- Tab 3: real vs nominal wage axis ------------------------------------------

def scatter_x(dashboard, naics, basis):
    _, fig, *_ = dashboard.update_sector_tab(naics, basis)
    return pd.Series(fig.data[0].x), fig


def test_sector_scatter_defaults_to_real_wage_growth(dashboard):
    _, fig_default, *_ = dashboard.update_sector_tab("62")
    assert "real wage growth" in fig_default.layout.title.text.lower()
    assert fig_default.layout.xaxis.title.text.startswith("Real")


def test_real_and_nominal_views_plot_the_same_occupations_shifted_by_one_years_inflation(dashboard):
    real, fig_real = scatter_x(dashboard, "62", "real")
    nominal, fig_nominal = scatter_x(dashboard, "62", "nominal")
    assert len(real) == len(nominal) > 50
    assert list(fig_real.data[0].y) == list(fig_nominal.data[0].y)               # same employment changes, same order
    # (1 + nominal) / (1 + real) is the same CPI ratio for every occupation in a given year
    ratio = ((1 + nominal / 100) / (1 + real / 100)).round(6)
    assert ratio.nunique() == 1 and ratio.iloc[0] > 1
    assert fig_nominal.layout.xaxis.title.text.startswith("Nominal")


def test_sector_ranking_tables_do_not_depend_on_the_wage_basis(dashboard):
    a = dashboard.update_sector_tab("72", "real")
    b = dashboard.update_sector_tab("72", "nominal")
    assert a[2].data == b[2].data and a[3].data == b[3].data
