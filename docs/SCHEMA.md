# Data Schema

All processed files live in `data/processed/` and are **committed to the repo** so the deployed dashboard can run without re-downloading from BLS (the raw `data/raw/` archives are the git-ignored part). Regenerate them via the scripts in `data/` and `src/` — pipeline order is in the [README](../README.md#rebuild-everything-from-raw-bls-data-optional).

Also tracked: `jolts_industry_panel.csv` and `cpi_monthly.csv` (BLS API pulls, below), `feature_set_experiment.json` (the test of CPI/JOLTS model inputs; see [METHODOLOGY.md](METHODOLOGY.md#cpi-and-jolts)), `overlap_test.json` (the 1/2/3-year-horizon test; see [METHODOLOGY.md](METHODOLOGY.md#model)) and `shock_2020_response.csv` (each occupation's observed 2020 employment response, used by the scenario tab; see [METHODOLOGY.md](METHODOLOGY.md#scenario-tool-a-replay-of-2020)).

## `oews_occupation_panel.csv`

One row per occupation x region x year. Annual cadence (OEWS reference period is May of each year) — see [METHODOLOGY.md](METHODOLOGY.md) for why this isn't monthly.

| Column | Type | Description |
|---|---|---|
| `year` | int | Release year, 2015-2025 |
| `region` | str | `US` (national) or `FL` (Florida as a whole; no metro or county rows) |
| `occ_code` | str | 6-digit SOC occupation code, e.g. `29-1141` |
| `occ_title` | str | Occupation name |
| `employment_level` | float | Estimated employment count |
| `wage_hourly_mean` | float | Mean hourly wage ($) |
| `wage_annual_mean` | float | Mean annual wage ($) |
| `wage_hourly_median` | float | Median hourly wage ($) |
| `wage_annual_median` | float | Median annual wage ($) |
| `wage_annual_pct10` | float | 10th percentile annual wage ($) |
| `wage_annual_pct90` | float | 90th percentile annual wage ($) |

Source: `data/fetch_oews.py`, parsing BLS's annual `oesm{YY}all.zip` archives. Rows are filtered to `O_GROUP == "detailed"` (not aggregated major/minor/broad groups) and `NAICS == "000000"` (cross-industry total, not a specific sector). 17,587 rows; 789–831 distinct occupation codes per year.

**Nulls:** BLS marks suppressed or out-of-range estimates with `*` / `#` strings, which the parser coerces to null. Share of nulls per column (all rows): `employment_level` 1.6%, `wage_annual_mean` 1.0%, `wage_annual_median` 2.1%, `wage_hourly_mean` 7.7%, `wage_hourly_median` 8.7%, `wage_annual_pct90` 4.8% (`wage_annual_pct10` 1.0%). The missingness is not random: hourly columns are null mainly for occupations BLS reports as annual salary only (postsecondary teachers, legislators, school administrators — rows with a null `wage_hourly_mean` have a median annual mean wage of $78,120 vs. $50,970 for the rest), and `wage_annual_pct90` is null for the very top earners (median annual mean wage $172,320 vs. $51,560; e.g. Cardiologists, Pediatric Surgeons). Analyses on those columns silently drop higher-paid occupations. Growth features are also null wherever the occupation code is absent from the comparison year (SOC code revisions mean not every occupation has a continuous 11-year history).

## `oews_occupation_features.csv`

`oews_occupation_panel.csv` plus engineered growth features (built by `src/build_features.py`). "YoY" and "5yr" are between consecutive/5-years-apart *annual releases*, not rolling calendar windows.

| Column | Type | Description |
|---|---|---|
| `employment_change_yoy_pct` | float | % change in `employment_level` vs. prior release |
| `wage_growth_yoy_pct` | float | % change in `wage_annual_mean` vs. prior release |
| `employment_change_5yr_cagr` | float | 5-year compound annual growth rate, employment |
| `wage_growth_5yr_cagr` | float | 5-year compound annual growth rate, wage |
| `contraction_flag_12mo` | Int64 (nullable) | 1 if `employment_change_yoy_pct < -2%`, else 0; null if no prior-year data |

The lags are matched on the actual prior year (year − 1, year − 5), not the previous row, so an occupation missing a year gets a blank rather than a change that silently spans several years.

Added by `src/build_macro_features.py` (CPI and JOLTS; see [METHODOLOGY.md](METHODOLOGY.md#cpi-and-jolts)):

| Column | Type | Description |
|---|---|---|
| `wage_annual_mean_real` | float | `wage_annual_mean` in latest-release-year (2025) dollars, deflated with May CPI-U |
| `real_wage_growth_yoy_pct` | float | YoY change in the real wage (%); null wherever `wage_growth_yoy_pct` is null |
| `purchasing_power_loss_flag` | Int64 (nullable) | 1 if the nominal wage rose but the real wage fell, else 0 |
| `jolts_exposure_openings_rate` | float | Job openings rate (%), annual average, of the occupation's industries weighted by its national employment in each sector; Florida rows use the same national industry rates |
| `jolts_exposure_quits_rate` | float | Same, quits rate |
| `jolts_exposure_layoffs_rate` | float | Same, layoffs and discharges rate |
| `jolts_region_openings_rate` | float | Total-nonfarm job openings rate for the row's own region (US, or Florida) |

JOLTS columns are null for the 20 rows whose occupation has no employment in any JOLTS-covered sector.

## `jolts_industry_panel.csv`

Monthly JOLTS rates (11,860 rows), from `data/fetch_jolts.py`: `year`, `month`, `region` (`US` or `FL`), `industry_code` (JOLTS industry code; string), `industry_name`, `data_element` (job openings, hires, total separations, quits, or layoffs and discharges rate) and `rate` (%). US has 16 industries, January 2015 to August 2026; Florida has total nonfarm only (`000000`), January 2015 to December 2025. Seasonally adjusted.

## `cpi_monthly.csv`

Monthly CPI-U, all items, US city average, not seasonally adjusted (series `CUUR0000SA0`): `year`, `month`, `cpi`, and `cpi_index_2020base` (2020 annual average = 100). 139 rows, January 2015 to August 2026, with no October 2025 value.

## `oews_industry_panel.csv` / `oews_industry_features.csv`

Same structure as above, but national-level only, broken out by industry sector instead of collapsed to cross-industry. Adds:

| Column | Type | Description |
|---|---|---|
| `naics` | str | 2-digit/range NAICS sector code, e.g. `62`, `44-45` |
| `naics_title` | str | Sector name, e.g. "Health Care and Social Assistance" |

The features file also has `wage_annual_mean_real`, `real_wage_growth_yoy_pct` and `purchasing_power_loss_flag` (same definitions as above) and `jolts_openings_rate`, `jolts_quits_rate` and `jolts_layoffs_rate`: the national annual-average JOLTS rates for the sector's closest JOLTS industry (null for sector `11`, agriculture, which JOLTS doesn't cover).

92,104 rows: 20 sectors × 11 years, with 143–641 detailed occupations present per sector in 2025 (about 426 on average) — most occupations don't appear in most sectors, so this is far smaller than 20 × ~800 × 10. About 8,200–8,600 rows per year (8,527 in 2025). Sector titles aren't stable across releases: NAICS `99` (government) carries four different `naics_title` strings across the 11 years, so group on `naics`, not `naics_title` (the dashboard uses the most recent year's title). State-level industry breakdowns are excluded — OEWS suppresses too many state x sector x detailed-occupation cells to build a reliable Florida-specific version.

## `employment_forecast_model.txt`

A saved LightGBM booster (text format), 75 trees. **Predicts log employment growth ratio, not absolute employment level** — see [METHODOLOGY.md](METHODOLOGY.md#model) for why. Reconstruct the level forecast as `employment_level * exp(model.predict(features))`. Refit on every available pair (feature years 2016–2024), so it is applied to 2025 features to forecast 2026.

Feature columns (order matters for the saved categorical encoding):
`employment_level, wage_annual_mean, wage_annual_median, employment_change_yoy_pct, wage_growth_yoy_pct, contraction_flag_12mo, region, occ_major_group` (the last two are categorical; `occ_major_group` is `occ_code[:2]`; `contraction_flag_12mo` is passed as a float). The 5-year CAGR columns exist in the feature table but are not model inputs.

## `model_metrics.json`

Written by `src/train_model.py`: the selected hyperparameters, the top of the hyperparameter grid, per-fold walk-forward results for the model and both baselines (persistence and median growth), the 2024 → 2025 holdout results with bootstrap intervals, and each feature's share of total gain. The dashboard's accuracy note on the Predictive Scenarios tab is generated from this file.
