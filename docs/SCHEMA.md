# Data Schema

All processed files live in `data/processed/` and are **committed to the repo** so the deployed dashboard can run without re-downloading from BLS (the raw `data/raw/` archives are the git-ignored part). Regenerate them via the scripts in `data/` and `src/` — pipeline order is in the [README](../README.md#rebuild-everything-from-raw-bls-data-optional).

Not yet produced (scripts written, never completed/merged): `jolts_industry_panel.csv` (`data/fetch_jolts.py`) and `cpi_monthly.csv` (`data/fetch_cpi.py`).

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

Source: `data/fetch_oews.py`, parsing BLS's annual `oesm{YY}all.zip` archives. Rows are filtered to `O_GROUP == "detailed"` (not aggregated major/minor/broad groups) and `NAICS == "000000"` (cross-industry total, not a specific sector). 15,967 rows; 789–831 distinct occupation codes per year.

**Nulls:** BLS marks suppressed or out-of-range estimates with `*` / `#` strings, which the parser coerces to null. Share of nulls per column (all rows): `employment_level` 1.6%, `wage_annual_mean` 1.0%, `wage_annual_median` 2.2%, `wage_hourly_mean` 7.7%, `wage_hourly_median` 8.8%, `wage_annual_pct90` 5.2% (`wage_annual_pct10` 1.0%). The missingness is not random: hourly columns are null mainly for occupations BLS reports as annual salary only (postsecondary teachers, legislators, school administrators — rows with a null `wage_hourly_mean` have a median annual mean wage of $77,890 vs. $49,990 for the rest), and `wage_annual_pct90` is null for the very top earners (median annual mean wage $172,320 vs. $50,430; e.g. Cardiologists, Pediatric Surgeons). Analyses on those columns silently drop higher-paid occupations. Growth features are also null wherever the occupation code is absent from the comparison year (SOC code revisions mean not every occupation has a continuous 10-year history).

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

83,577 rows: 20 sectors × 10 years, with 151–656 detailed occupations present per sector (about 430 on average) — most occupations don't appear in most sectors, so this is far smaller than 20 × ~800 × 10. About 8,200–8,600 rows per year. Sector titles aren't stable across releases: NAICS `99` (government) carries four different `naics_title` strings across the 10 years, so group on `naics`, not `naics_title` (the dashboard uses the most recent year's title). State-level industry breakdowns are excluded — OEWS suppresses too many state x sector x detailed-occupation cells to build a reliable Florida-specific version.

## `employment_forecast_model.txt`

A saved LightGBM booster (text format). **Predicts log employment growth ratio, not absolute employment level** — see [METHODOLOGY.md](METHODOLOGY.md#model) for why. Reconstruct the level forecast as `employment_level * exp(model.predict(features))`.

Feature columns (order matters for the saved categorical encoding):
`employment_level, wage_annual_mean, wage_annual_median, employment_change_yoy_pct, wage_growth_yoy_pct, employment_change_5yr_cagr, wage_growth_5yr_cagr, contraction_flag_12mo, region, occ_major_group` (the last two are categorical; `occ_major_group` is `occ_code[:2]`).
