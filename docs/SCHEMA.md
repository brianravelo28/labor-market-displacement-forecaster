# Data Schema

All processed files live in `data/processed/` (git-ignored; regenerate via the scripts in `data/` and `src/`, or see [METHODOLOGY.md](METHODOLOGY.md) for the pipeline order).

## `oews_occupation_panel.csv`

One row per occupation x region x year. Annual cadence (OEWS reference period is May of each year) — see [METHODOLOGY.md](METHODOLOGY.md) for why this isn't monthly.

| Column | Type | Description |
|---|---|---|
| `year` | int | Release year, 2015-2024 |
| `region` | str | `US` (national) or `FL` (Florida state-level) |
| `occ_code` | str | 6-digit SOC occupation code, e.g. `29-1141` |
| `occ_title` | str | Occupation name |
| `employment_level` | float | Estimated employment count |
| `wage_hourly_mean` | float | Mean hourly wage ($) |
| `wage_annual_mean` | float | Mean annual wage ($) |
| `wage_hourly_median` | float | Median hourly wage ($) |
| `wage_annual_median` | float | Median annual wage ($) |
| `wage_annual_pct10` | float | 10th percentile annual wage ($) |
| `wage_annual_pct90` | float | 90th percentile annual wage ($) |

Source: `data/fetch_oews.py`, parsing BLS's annual `oesm{YY}all.zip` archives. Rows are filtered to `O_GROUP == "detailed"` (not aggregated major/minor/broad groups) and `NAICS == "000000"` (cross-industry total, not a specific sector).

## `oews_occupation_features.csv`

`oews_occupation_panel.csv` plus engineered growth features (built by `src/build_features.py`). "YoY" and "5yr" are between consecutive/5-years-apart *annual releases*, not rolling calendar windows.

| Column | Type | Description |
|---|---|---|
| `employment_change_yoy_pct` | float | % change in `employment_level` vs. prior release |
| `wage_growth_yoy_pct` | float | % change in `wage_annual_mean` vs. prior release |
| `employment_change_5yr_cagr` | float | 5-year compound annual growth rate, employment |
| `wage_growth_5yr_cagr` | float | 5-year compound annual growth rate, wage |
| `contraction_flag_12mo` | Int64 (nullable) | 1 if `employment_change_yoy_pct < -2%`, else 0; null if no prior-year data |

## `oews_industry_panel.csv` / `oews_industry_features.csv`

Same structure as above, but national-level only, broken out by industry sector instead of collapsed to cross-industry. Adds:

| Column | Type | Description |
|---|---|---|
| `naics` | str | 2-digit/range NAICS sector code, e.g. `62`, `44-45` |
| `naics_title` | str | Sector name, e.g. "Health Care and Social Assistance" |

20 sectors x ~800 detailed occupations x 10 years ≈ 84K rows. State-level industry breakdowns are excluded — OEWS suppresses too many state x sector x detailed-occupation cells to build a reliable Florida-specific version.

## `employment_forecast_model.txt`

A saved LightGBM booster (text format). **Predicts log employment growth ratio, not absolute employment level** — see [METHODOLOGY.md](METHODOLOGY.md#model) for why. Reconstruct the level forecast as `employment_level * exp(model.predict(features))`.

Feature columns (order matters for the saved categorical encoding):
`employment_level, wage_annual_mean, wage_annual_median, employment_change_yoy_pct, wage_growth_yoy_pct, employment_change_5yr_cagr, wage_growth_5yr_cagr, contraction_flag_12mo, region, occ_major_group` (the last two are categorical; `occ_major_group` is `occ_code[:2]`).
