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


# ---- Tab 4: 2020 shock and recovery ----------------------------------------------

def card(kpis, label):
    return next(k for k in kpis if k.children[0].children.startswith(label))


def card_value(kpis, label):
    return card(kpis, label).children[1].children


def test_the_tab_is_descriptive_with_no_forecast_controls(dashboard):
    ids = {getattr(c, "id", None) for c in dashboard.app.layout._traverse()}
    assert {"shock-occ-dropdown", "shock-region-toggle", "shock-chart"} <= ids
    assert not {"severity-slider", "gdp-slider", "automation-slider", "immigration-slider"} & ids
    for name in ("predict_baseline", "forecast_model", "risk_level", "BASELINE_ACCURACY_NOTE"):
        assert not hasattr(dashboard, name)


def test_growth_numbers_are_red_when_negative_and_green_otherwise(dashboard):
    assert dashboard.growth_color(-0.1) == "#c0392b" and dashboard.growth_color(0.0) == "#27ae60"


def test_cards_match_the_published_employment_figures(dashboard, occupation_features):
    f = occupation_features
    us = f[(f["region"] == "US") & (f["occ_code"] == RETAIL)].set_index("year")["employment_level"]
    kpis, fig, *_ = dashboard.update_shock_tab(RETAIL, "US")
    assert card_value(kpis, "Employment, 2019").replace(",", "") == f"{us[2019]:.0f}"
    assert float(card_value(kpis, "Change in the 2020").rstrip("%")) == pytest.approx((us[2020] / us[2019] - 1) * 100, abs=0.06)
    assert float(card_value(kpis, "2025 vs 2019").rstrip("%")) == pytest.approx((us[2025] / us[2019] - 1) * 100, abs=0.06)


def test_chart_is_indexed_to_2019_and_compares_with_the_balanced_total(dashboard):
    _, fig, *_ = dashboard.update_shock_tab(RETAIL, "US")
    own, total = fig.data
    assert dict(zip(own.x, own.y))[2019] == pytest.approx(100.0)
    assert dict(zip(total.x, total.y))[2019] == pytest.approx(100.0)
    assert [t.name for t in fig.data][0] == "This occupation"
    # the comparison set is fixed, so it cannot move because occupations were added or dropped
    totals, n = dashboard.BALANCED_TOTALS["US"]
    assert n > 600 and totals.index.min() == dashboard.FIRST_YEAR and totals.index.max() == dashboard.LATEST_YEAR


def test_a_hard_hit_occupation_fell_more_than_a_resilient_one(dashboard):
    waiters = dashboard.update_shock_tab("35-3031", "US")[0]
    nurses = dashboard.update_shock_tab("29-1141", "US")[0]
    assert float(card_value(waiters, "Change in the 2020").rstrip("%")) < float(card_value(nurses, "Change in the 2020").rstrip("%")) - 10
    assert float(card_value(waiters, "2020 response").split()[0]) < -10


def test_the_card_says_where_the_response_came_from(dashboard):
    own = dashboard.update_shock_tab(RETAIL, "US")[0]
    assert "own 2020 change" in card(own, "2020 response").children[2].children
    group_rows = dashboard.shock_response[dashboard.shock_response["source"] == "group"]
    region, code = next((r, c) for r, c in group_rows.index if not dashboard.df[(dashboard.df["occ_code"] == c) & (dashboard.df["region"] == r) & (dashboard.df["year"] == 2019)].empty
                        and not dashboard.df[(dashboard.df["occ_code"] == c) & (dashboard.df["region"] == r) & (dashboard.df["year"] == 2020)].empty)
    borrowed = dashboard.update_shock_tab(code, region)[0]
    assert "occupation group" in card(borrowed, "2020 response").children[2].children


def test_an_occupation_without_2019_data_says_so(dashboard):
    kpis, fig, hardest, resilient = dashboard.update_shock_tab("00-0000", "US")
    assert kpis[0].children[1].children == "--" and fig.layout.title.text.startswith("No 2019/2020 data")
    assert hardest is not None and resilient is not None            # the two tables still render


def test_tables_list_only_reliable_own_responses_in_the_selected_region(dashboard):
    for region in ("US", "FL"):
        *_, hardest, resilient = dashboard.update_shock_tab(RETAIL, region)
        own = dashboard.shock_response.xs(region, level="region")
        own_codes = set(own[own["source"] == "occupation"].index)
        titles = dashboard.df[(dashboard.df["region"] == region) & dashboard.df["occ_code"].isin(own_codes)]["occ_title"].unique()
        for table in (hardest, resilient):
            assert len(table.data) == 10
            assert {r["occ_title"] for r in table.data} <= set(titles)
            assert not any("All Other" in r["occ_title"] for r in table.data)
        assert [r["response_pct"] for r in hardest.data] == sorted(r["response_pct"] for r in hardest.data)
        assert hardest.data[0]["response_pct"] < 0 < resilient.data[0]["response_pct"] + 100
        assert max(r["response_pct"] for r in hardest.data) <= min(r["response_pct"] for r in resilient.data)


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
