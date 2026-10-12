# Methodology

## Data architecture: corrected from the original plan

The project was originally scoped assuming BLS OEWS could be pulled as a monthly time series (2015-present) via the BLS public API, and that JOLTS had occupation-level vacancy/separation rates. Both were verified false against the live API before any pipeline code was written:

- **OEWS is annual, not monthly**, and BLS does not retain OEWS history under stable series IDs in its timeseries API — every series has `begin_year == end_year` (the current release only). The real historical panel comes from BLS's separate annual archive files (`oesm{YY}all.zip`, one per year, published at bls.gov/oes/special-requests/), which this project downloads and stacks itself (`data/fetch_oews.py`). Cadence is therefore annual, reference period May of each year. The panel covers the 2015–2025 releases.
- **JOLTS has no occupation-level breakdown** — series are keyed by NAICS industry only, and BLS publishes industry detail for the US but, at the state level, only Florida *total nonfarm* (queried 2026-10-12: Florida retail, leisure & hospitality, health care, construction and manufacturing all return "Series does not exist"). `data/fetch_jolts.py` is written to pull the US industry series and Florida total nonfarm; the plan is to join the US industry rates onto occupations via a primary-industry mapping as a *regional-tightness proxy*, not a true occupation-specific rate. **That pull has not been completed or merged yet** (see Known limitations).
- OEWS column names and casing drift across release years (2015 uses `"occ code"` with a space and truncates the group label to `"detail "`; 2019 uses `occ_code`/`o_group`; 2024 uses `OCC_CODE`/`O_GROUP`; the `PRIM_STATE` column doesn't exist before ~2020). The parser normalizes headers and identifies Florida via `AREA_TYPE == 2 & AREA == 12` (the FIPS code), which is stable across every release year regardless of the naming drift. "Florida" throughout this project means the state as a whole; no metro or county rows are used.

## Model

**Target:** next-year `employment_level` per occupation × region ("12-months-ahead" from the original brief, adapted to annual cadence since OEWS doesn't publish monthly). The model predicts the **log growth ratio** `log(employment_next / employment_now)` and the level is reconstructed as `employment_now × exp(prediction)`. A level regression fails at the top end: employment spans 50 to ~4.6M, and with so few very large occupations a level model mispredicted the largest one (Retail Salespersons) by 40–60% even on rows it had trained on. Growth ratios share one scale across occupation sizes.

**Features** (8): current employment, mean and median annual wage, YoY change in employment and in wage, the contraction flag (employment down >2% YoY), region, and the first two digits of the SOC code. The 5-year CAGR features are in the feature table but deliberately *not* model inputs: they are empty for every occupation before 2020, so requiring them silently discards every training pair before 2020.

**Observations:** each row is a pair — features in year *t*, employment in year *t+1* — for a (region, occupation) with both years present. Only true adjacent-year pairs are used. This gives 12,928 pairs (feature years 2016–2024, targets 2017–2025).

**Baselines.** A forecast is only worth something if it beats doing nothing, so every model figure is reported beside two baselines: *persistence* (next year = this year) and *median growth* (every occupation grows by the training period's median).

**Validation is walk-forward.** To forecast from year *T*, a model may train only on pairs whose target was already known by *T* (features from years ≤ *T*−1).
- **Inner folds** (features 2020, 2021, 2022, 2023) pick the hyperparameters: a 36-point grid over objective (L2, Huber, L1), number of trees, leaves and minimum leaf size, scored by mean median-error over the four folds. The winner is an L1 (median) objective, 75 trees, 31 leaves, `min_data_in_leaf=100`, learning rate 0.05. The grid is flat — the five best configurations differ by 0.06 points of median error (7.67–7.73) — so the exact choice shouldn't be over-read.
- **Holdout** (features 2024 → the 2025 release): evaluated once, after the choices were fixed. The model for it is trained only on pairs through 2023→2024.
- The **shipped** model is then refit on all 12,928 pairs, so it has no holdout of its own; its accuracy is the walk-forward estimate above. The dashboard forecasts 2026 from 2025 features.

**Results** (all errors are on next-year employment *level*; "closer" = share of rows where the model's error is smaller than persistence's):

| Features → target | n | Model: RMSE / median error | Persistence: RMSE / median error | Closer than persistence |
|---|---|---|---|---|
| 2020 → 2021 | 1,398 | 27,037 / 9.5% | 26,558 / 9.9% | 54% |
| 2021 → 2022 | 1,398 | 21,856 / 8.7% | 24,328 / 8.2% | 46% |
| 2022 → 2023 | 1,524 | 11,517 / 6.2% | 15,853 / 6.9% | 53% |
| 2023 → 2024 | 1,511 | 10,768 / 6.3% | 13,769 / 6.9% | 56% |
| **Inner-fold mean** | | **17,794 / 7.7%** | 20,127 / 8.0% | |
| **Holdout 2024 → 2025** | 1,503 | **11,067 / 6.1%** | 13,370 / 6.4% | 54% |

On the holdout, MAPE is 13.3% vs. 13.6%, and for the 33 occupations with 1M+ employees RMSE is 62,216 vs. 77,355. Predicted and actual growth are only weakly related: rank correlation 0.21 across all holdout rows and 0.29 across the 854 occupations with 10,000+ employees.

**How to read this.** The model is slightly better than assuming no change — it wins on RMSE in three of the four inner folds and on median error in three of the four (both in 2022 and 2023, and on the holdout) — but the margin is small and not statistically clear-cut: bootstrapping the holdout rows gives a 95% interval of −0.6 to +0.0 points for the difference in median error and 0.63 to 1.20 for the ratio of RMSEs (a ratio of 1 is a tie). The RMSE gain comes mostly from a few large occupations. It lost on median error in the 2021 fold (features 2021 → target 2022, when median occupation growth was +4.3% after −0.9% the year before) and on RMSE in the 2020 fold. This is a modest signal, not a precise predictor. Share of total gain by feature: SOC major group 25%, employment change YoY 23%, employment level 19%, median wage 10%, wage change YoY 10%, mean wage 8%, region 4%, contraction flag 1% (gain share describes how the trees split, not causal importance).

**Model history.** The model has been through two failed versions, both fixed:
1. *Level regression* (`log1p(employment_next)`). Fine in aggregate but wrong on the largest occupations: only 7 of 1,276 training rows exceeded 2M employees, and Retail Salespersons was mispredicted by ~60%.
2. *Growth-ratio target, fixed 2015–2020 split.* This fixed the large-occupation problem, but the 5-year-CAGR features were empty before 2020, so after rows with missing features were dropped it trained on a **single** year (2020 → 2021, 1,276 rows) and early stopping on 2021 kept a single tree. On the 2022–23 test years it lost to persistence (RMSE 16,521 vs. 11,233) and gave nearly identical forecasts for every occupation. Its published error numbers had never been compared to a baseline.
3. *Current version.* All usable years, walk-forward validation, baselines reported alongside, and an untouched holdout.

## Displacement risk scoring

The full formula from the original brief is:

```
risk = 0.4*(contraction relative to national avg) + 0.3*(wage compression) 
     + 0.2*(low hiring demand) + 0.1*(high separation rate)
```

70% of that weight (wage compression needs CPI; hiring demand needs job-posting volume, which this build doesn't collect; separation rate needs JOLTS) isn't available yet. The Predictive Scenarios tab uses a simplified three-tier `LOW/MODERATE/HIGH` threshold on forecast growth alone (below −2% = MODERATE, below −10% = HIGH), clearly labeled as such in the UI. The thresholds are arbitrary and unvalidated. Applied to the shipped model's forecasts from 2025 features, 1,323 of 1,550 forecastable occupation-region rows come out LOW, 227 MODERATE and none HIGH.

## Known limitations

- **Annual, not monthly** — trend charts and forecasts move once a year, not continuously.
- **OEWS isn't designed for year-over-year comparison.** BLS's FAQ says the survey produces cross-sectional estimates and is "less useful for looking at changes over time", citing classification changes, methodology changes, and its three-year pooled sample design ([BLS](https://www.bls.gov/oes/oes_ques.htm)). Each release combines six semiannual panels collected over three years; the May 2025 estimates use May 2025, Nov 2024, May 2024, Nov 2023, May 2023 and Nov 2022, so consecutive releases share four of their six panels. YoY changes here are therefore changes in a moving average, and a forecast of "next year's employment" is a forecast of the next such estimate. Everything in this project that is a growth rate inherits this.
- **Occupation codes change between releases.** US codes 2018 → 2019: 808 → 789, with 98 dropped and 79 new; 2020 → 2021: 30 dropped, 72 new. Only 700 US occupation codes exist in all 11 years. YoY is blank where the prior year is absent (matched on the actual prior year, not the previous row — an earlier positional `shift` silently spanned gaps for ~0.4% of occupation and ~1.8% of sector values).
- **Small-occupation noise** — occupations with employment in the tens/hundreds show large YoY % swings (OEWS sampling error dominates at that scale; the standard deviation of next-year growth is about 61 points for occupations under 1,000 employees vs. 7 above 50,000). At the industry-sector cross-tab level this got bad enough to look like a data bug: in the 2024 release, occupations reported at n≈130 within a sector (e.g. "Self-Enrichment Teachers" under NAICS Manufacturing — real BLS data, likely in-house corporate trainers at a diversified manufacturer, but confusing out of context) were showing up in the sector tab's "top movers" tables purely from sampling noise. The Industry & Sector Analysis tab's ranked tables now apply a 500-employee floor (`SECTOR_RANKING_MIN_EMPLOYMENT` in `app/app.py`) before ranking; the scatter plot still shows every occupation (small ones are naturally de-emphasized by bubble size). The floor applies to **both** years being compared: a floor on current-year employment alone let a jump from a tiny base straight through (Construction Laborers in Accommodation & Food Services, 2024: 120 → 960 employees, +700%). The prior-year count is recovered exactly from the YoY percentage. Notebook 1's national occupation-level "top movers" tables have no floor and show the effect.
- **Estimates carried forward unchanged** — for some smaller Florida occupations BLS publishes the identical employment figure in consecutive releases (2024 release: 13 of 744 comparable occupations vs. 2 nationally; 2025 release: 11 of 737 vs. 3 of 790). In 2024 that put Florida's median employment growth at exactly 0.0%, and it is −0.25% in 2025. The Regional Comparison tab therefore shows the median *with 25th–75th percentile whiskers* rather than a median-only bar, which read as a broken chart; the whiskers are scoped to the IQR (not min/max) because a handful of small occupations swing from about −82% to +300% and stretched a box plot's axis until the IQR was unreadable.
- **JOLTS and CPI not integrated yet** — a multi-day BLS API outage in August 2026 delayed the pulls; the API is reachable again (CPI series `CUUR0000SA0` confirmed live on 2026-10-07), so the remaining work is running `data/fetch_jolts.py` and `data/fetch_cpi.py`, merging into the feature table, and retraining. Until then there are no vacancy/separation-rate features and no inflation-adjusted wages, so the "wage compression" half of the project's premise is still unimplemented — the dashboard shows nominal figures only and says so.
- **JOLTS is industry-level only**, not occupation-specific, and state-level only for Florida total nonfarm, everywhere it will be used.
- **No job-postings data** — the original brief's Indeed/LinkedIn scraping scope was deliberately deferred (cost/ToS considerations); skill-diversity and posting-volume features don't exist in this build.
- **No fairness/equity audit yet** — the original brief's approach of inferring gender/age from job-posting text patterns is methodologically weak and there's no job-postings data source in this build anyway; needs a sounder approach before it's built.
- **Regional wage premium finding**: Florida pays *less*, not more, than the national figure — contrary to the original brief's assumed +3-8% premium. Kept as a genuine finding, not corrected; see the next section for exactly what the numbers do and don't say.

## Florida vs. national wages

Two different comparisons produce two different magnitudes, and the README/dashboard should be read with the distinction in mind (all figures: 2025 OEWS, nominal annual mean wage, not cost-of-living adjusted):

- **Median across occupations** (what the Regional Comparison tab's wage chart shows): the median occupation's mean annual wage is $58,730 in Florida vs. $66,290 nationally, a **−11.4%** gap, over the 790 occupations present in both regions.
- **Head-to-head, same occupation** (FL wage ÷ US wage, per occupation, 777 occupations with a wage in both): the median ratio is **−7.7%** (25th–75th percentile −13.5% to −2.5%), **84%** of occupations pay less in Florida, and weighting by Florida employment gives **−6.5%**.

The direction is robust either way; the ~8% head-to-head figure is the better estimate of "how much less does the same job pay in Florida," while the ~11% is the gap between two distribution medians, which is not the same thing as the median of the per-occupation gaps. Neither adjusts for cost of living, so neither says anything about purchasing power — that comparison would need a regional price index that isn't part of this build.
