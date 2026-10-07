# Methodology

## Data architecture: corrected from the original plan

The project was originally scoped assuming BLS OEWS could be pulled as a monthly time series (2015-present) via the BLS public API, and that JOLTS had occupation-level vacancy/separation rates. Both were verified false against the live API before any pipeline code was written:

- **OEWS is annual, not monthly**, and BLS does not retain OEWS history under stable series IDs in its timeseries API — every series has `begin_year == end_year` (the current release only). The real historical panel comes from BLS's separate annual archive files (`oesm{YY}all.zip`, one per year, published at bls.gov/oes/special-requests/), which this project downloads and stacks itself (`data/fetch_oews.py`). Cadence is therefore annual, reference period May of each year.
- **JOLTS has no occupation-level breakdown** — series are keyed by NAICS industry only. `data/fetch_jolts.py` is written to pull monthly job openings/hires/separations rates by industry sector for the US and Florida; the plan is to join them onto occupations via a primary-industry mapping as a *regional-tightness proxy*, not a true occupation-specific rate. **That pull has not been completed or merged yet** (see Known limitations).
- OEWS column names and casing drift across release years (2015 uses `"occ code"` with a space and truncates the group label to `"detail "`; 2019 uses `occ_code`/`o_group`; 2024 uses `OCC_CODE`/`O_GROUP`; the `PRIM_STATE` column doesn't exist before ~2020). The parser normalizes headers and identifies Florida via `AREA_TYPE == 2 & AREA == 12` (the FIPS code), which is stable across every release year regardless of the naming drift.

## Model

**Target:** next-year `employment_level` per occupation x region ("12-months-ahead" from the original brief, adapted to annual cadence since OEWS doesn't publish monthly).

**A real bug, found and fixed during development:** the first version trained LightGBM to predict the absolute employment level directly (even after a `log1p` transform to tame the scale). It looked fine in aggregate (MAPE ~13-20%) but was badly wrong for the handful of very largest occupations — verified it mispredicted Retail Salespersons (the single largest US occupation) by ~60%, *even on a row it was trained on*. Root cause: employment levels span three orders of magnitude (50 to 4.6M), and only 7 of 1,276 training rows exceed 2M employees (median ~14,000) — squared-error loss simply doesn't get enough signal at that extreme to fit it well.

**Fix:** predict `log(employment_next / employment_now)` — the growth ratio — instead of the absolute level, then reconstruct `employment_next = employment_now * exp(prediction)`. A 50-employee occupation's growth rate and a 3.8M-employee occupation's growth rate live on the same numeric scale, so the extreme tail no longer starves the model of comparable training examples. This dropped RMSE for 1M+-employee occupations from ~700K-940K to ~90K-180K, and fixed the Retail Salespersons case to within a few percent.

**Train/val/test split** (annual analog of the original 2015-2021 / 2022 / 2023-2024 design):
- train: feature year 2015-2020, target = employment in year+1
- val: feature year 2021, target = employment in 2022
- test: feature year 2022-2023, target = employment in year+1

**Hyperparameters:** `n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=32, min_data_in_leaf=20, feature_fraction=0.75, bagging_fraction=0.9, bagging_freq=1`, with early stopping on the validation set.

**Test-set results** (current build, OEWS-only features): RMSE ≈ 16,500, MAPE ≈ 12.1%, median absolute % error ≈ 7.4%, RMSE for 1M+-employee occupations ≈ 91,500. The original spec's target of <50,000 RMSE for 1M+ occupations isn't hit yet — that target assumed a richer feature set (JOLTS vacancy rates, job-posting volume, wage compression flags) that isn't wired in yet; see Limitations.

## Displacement risk scoring

The full formula from the original brief is:

```
risk = 0.4*(contraction relative to national avg) + 0.3*(wage compression) 
     + 0.2*(low hiring demand) + 0.1*(high separation rate)
```

70% of that weight (wage compression needs CPI; hiring demand needs job-posting volume, which this build doesn't collect; separation rate needs JOLTS) isn't available yet. The Predictive Scenarios tab uses a simplified three-tier `LOW/MODERATE/HIGH` threshold on forecast growth rate alone as a placeholder, clearly labeled as such in the UI.

## Known limitations

- **Annual, not monthly** — trend charts and forecasts move once a year, not continuously.
- **Small-occupation noise** — occupations with employment in the tens/hundreds show large YoY % swings (OEWS sampling error dominates at that scale). At the industry-sector cross-tab level this got bad enough to look like a data bug: occupations reported at n≈130 within a sector (e.g. "Self-Enrichment Teachers" under NAICS Manufacturing — real BLS data, likely in-house corporate trainers at a diversified manufacturer, but confusing out of context) were showing up in the sector tab's "top movers" tables purely from sampling noise. The Industry & Sector Analysis tab's ranked tables now apply a 500-employee floor (`SECTOR_RANKING_MIN_EMPLOYMENT` in `app/app.py`) before ranking; the scatter plot still shows every occupation (small ones are naturally de-emphasized by bubble size). The floor is a partial fix: it applies to current-year employment, so a jump from a tiny base to just above 500 still ranks (Construction Laborers in Accommodation & Food Services: 960 employees, +700%) — see [DATA_QUIRKS.md](../DATA_QUIRKS.md) entry 8. Notebook 1's national occupation-level "top movers" tables don't have this floor and show the effect (e.g. "Timing Device Assemblers and Adjusters", 230 employees, -42.5%).
- **Estimates carried forward unchanged** — for some smaller Florida occupations BLS publishes the identical employment figure in consecutive releases (13 of 744 comparable occupations in 2024, vs. 2 nationally). That's why Florida's median employment growth is exactly 0.0% in 2024. The Regional Comparison tab therefore shows the median *with 25th–75th percentile whiskers* rather than a median-only bar, which read as a broken chart; the whiskers are scoped to the IQR (not min/max) because a handful of small occupations swing from about -82% to +300% and stretched a box plot's axis until the IQR was unreadable. Full write-up in [DATA_QUIRKS.md](../DATA_QUIRKS.md).
- **JOLTS and CPI not integrated yet** — a multi-day BLS API outage in August 2026 delayed the pulls; the API is reachable again (CPI series `CUUR0000SA0` confirmed live on 2026-10-07), so the remaining work is running `data/fetch_jolts.py` and `data/fetch_cpi.py`, merging into the feature table, and retraining. Until then there are no vacancy/separation-rate features and no inflation-adjusted wages, so the "wage compression" half of the project's premise is still unimplemented — the dashboard shows nominal figures only and says so.
- **JOLTS is industry-level only**, not occupation-specific, everywhere it will be used.
- **No job-postings data** — the original brief's Indeed/LinkedIn scraping scope was deliberately deferred (cost/ToS considerations); skill-diversity and posting-volume features don't exist in this build.
- **No fairness/equity audit yet** — the original brief's approach of inferring gender/age from job-posting text patterns is methodologically weak and there's no job-postings data source in this build anyway; needs a sounder approach before it's built.
- **Regional wage premium finding**: Florida pays *less*, not more, than the national figure — contrary to the original brief's assumed +3-8% premium. Kept as a genuine finding, not corrected; see the next section for exactly what the numbers do and don't say.

## Florida vs. national wages

Two different comparisons produce two different magnitudes, and the README/dashboard should be read with the distinction in mind (all figures: 2024 OEWS, nominal annual mean wage, not cost-of-living adjusted):

- **Median across occupations** (what the Regional Comparison tab's wage chart shows): the median occupation's mean annual wage is $57,935 in Florida vs. $63,945 nationally, a **-9.4%** gap, over the 791 occupations present in both regions.
- **Head-to-head, same occupation** (FL wage ÷ US wage, per occupation, 778 occupations with a wage in both): the median ratio is **-7.2%** (25th–75th percentile -13.5% to -2.5%), **84%** of occupations pay less in Florida, and weighting by Florida employment gives **-6.4%**.

The direction is robust either way; the ~7% head-to-head figure is the better estimate of "how much less does the same job pay in Florida," while the ~9% is the gap between two distribution medians, which is not the same thing as the median of the per-occupation gaps. Neither adjusts for cost of living, so neither says anything about purchasing power — that comparison would need a regional price index that isn't part of this build.
