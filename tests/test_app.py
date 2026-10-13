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
        _, _, risk, growth = dashboard.update_sector_tab(naics)
        for table in (risk, growth):
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


@pytest.mark.parametrize("pct,level", [(5, "LOW"), (-2, "LOW"), (-2.01, "MODERATE"), (-10, "MODERATE"), (-10.01, "HIGH")])
def test_risk_thresholds(dashboard, pct, level):
    assert dashboard.risk_level(pct)[0] == level


def test_scenario_adds_the_documented_adjustments(dashboard):
    base_kpis, *_ = dashboard.update_scenario_tab(RETAIL, "US", 0, 0, 0)
    kpis, fig, summary = dashboard.update_scenario_tab(RETAIL, "US", -10, 5, 2)
    base = float(text(base_kpis[0]).rstrip("%"))
    scenario = float(text(kpis[1]).rstrip("%"))
    expected = base + (-10) * dashboard.GDP_PASSTHROUGH - 5 + 2
    assert scenario == pytest.approx(expected, abs=0.11)    # both numbers are rounded to 0.1
    assert len(fig.data[0].y) == 3


def test_scenario_with_no_forecast_says_why(dashboard):
    kpis, fig, summary = dashboard.update_scenario_tab("00-0000", "US", 0, 0, 0)
    assert kpis == [] and summary.startswith("No forecast")


def test_accuracy_note_quotes_the_training_metrics(dashboard):
    import json

    h = json.loads((dashboard.PROCESSED_DIR / "model_metrics.json").read_text())["holdout"]
    assert f"{h['model']['median_ape']:.1f}%" in dashboard.BASELINE_ACCURACY_NOTE


# ---- chart hints -----------------------------------------------------------

def test_legend_sentence_only_when_the_chart_has_a_legend(dashboard):
    assert "legend" not in dashboard.chart_hint().children
    assert "legend" in dashboard.chart_hint(has_legend=True).children
