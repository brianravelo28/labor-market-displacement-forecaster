# Methodology

## Data architecture: corrected from the original plan

The project was originally scoped assuming BLS OEWS could be pulled as a monthly time series (2015-present) via the BLS public API, and that JOLTS had occupation-level vacancy/separation rates. Both were verified false against the live API before any pipeline code was written:

- **OEWS is annual, not monthly**, and BLS does not retain OEWS history under stable series IDs in its timeseries API — every series has `begin_year == end_year` (the current release only). The real historical panel comes from BLS's separate annual archive files (`oesm{YY}all.zip`, one per year, published at bls.gov/oes/special-requests/), which this project downloads and stacks itself (`data/fetch_oews.py`). Cadence is therefore annual, reference period May of each year. The panel covers the 2015–2025 releases.
- **JOLTS has no occupation-level breakdown** — series are keyed by NAICS industry only, and BLS publishes industry detail for the US but, at the state level, only Florida *total nonfarm* (queried 2026-10-12: Florida retail, leisure & hospitality, health care, construction and manufacturing all return "Series does not exist"). `data/fetch_jolts.py` pulls the US industry series and Florida total nonfarm, and the industry rates are joined onto occupations by employment-weighting (see [CPI and JOLTS](#cpi-and-jolts)) as a *tightness proxy*, not a true occupation-specific rate.
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

**Is the edge just overlapping survey data?** (`src/overlap_test.py`) Consecutive OEWS releases share four of their six survey panels, so next year's estimate is partly this year's data, and that overlap could flatter any forecast of it. If the model's edge came from the overlap it should shrink as the horizon grows: releases two years apart share two panels, three years apart share none. The test forecasts employment 1, 2 and 3 years ahead from the same feature years (2020, 2021, 2022), with the same walk-forward rule (train only on pairs whose target was already known) and the hyperparameters fixed at the one-year selection, so no horizon gets extra tuning. Pooled over the three feature years:

| Horizon (panels shared) | n | Median error: model / no-change | Difference, 95% interval | RMSE: model / no-change | RMSE ratio, 95% interval |
|---|---|---|---|---|---|
| 1 year (4 of 6) | 4,320 | 8.05% / 8.24% | −0.41 to +0.05 pts | 20,927 / 22,549 | 0.84 to 1.04 |
| 2 years (2 of 6) | 4,296 | 11.14% / 11.47% | −0.67 to +0.04 pts | 37,597 / 35,379 | 0.95 to 1.21 |
| 3 years (0 of 6) | 4,274 | 13.40% / 13.62% | −0.70 to +0.27 pts | 46,515 / 44,043 | 0.95 to 1.18 |

The result is **inconclusive**, not reassuring. On median error the model's small edge (0.2 to 0.3 points) is about the same size at every horizon, which doesn't look like an overlap effect, but the interval includes zero at every horizon, so there is no horizon at which the edge is clearly there. On RMSE the model is ahead at one year (ratio 0.93 in the pooled point estimate) and behind at two and three years (about 1.06 at both), which is what an overlap contribution would look like, but those intervals include 1 too. The three-year models also train on fewer pairs (2,612 to 5,219 versus 5,594 to 8,390), which handicaps them. So: the data can't show that the edge survives without overlapping panels, and can't show that it doesn't. The shipped one-year forecast should be read as a roughly-no-change forecast with a small, uncertain tilt.

**Model history.** The model has been through two failed versions, both fixed:
1. *Level regression* (`log1p(employment_next)`). Fine in aggregate but wrong on the largest occupations: only 7 of 1,276 training rows exceeded 2M employees, and Retail Salespersons was mispredicted by ~60%.
2. *Growth-ratio target, fixed 2015–2020 split.* This fixed the large-occupation problem, but the 5-year-CAGR features were empty before 2020, so after rows with missing features were dropped it trained on a **single** year (2020 → 2021, 1,276 rows) and early stopping on 2021 kept a single tree. On the 2022–23 test years it lost to persistence (RMSE 16,521 vs. 11,233) and gave nearly identical forecasts for every occupation. Its published error numbers had never been compared to a baseline.
3. *Current version.* All usable years, walk-forward validation, baselines reported alongside, and an untouched holdout.

## CPI and JOLTS

**Real wages (CPI-U).** OEWS wages describe a May reference period, so each release year is deflated with that year's May CPI-U (all items, US city average, not seasonally adjusted) and expressed in dollars of the latest release year (currently 2025). `wage_annual_mean_real` is the result, `real_wage_growth_yoy_pct` is `(1 + nominal growth) × CPI(May of prior year) ÷ CPI(May of this year) − 1`, and `purchasing_power_loss_flag` is 1 when the nominal wage rose but the real wage fell. Adding a newer release changes the base year, so rerun the feature builders afterwards. Two caveats: the deflator is one national price index, so Florida wages are deflated with the national rate (no state price index), and OEWS wages are themselves updated to the reference period by BLS, so this is an approximate deflation, not an exact one.

What it shows: the share of US occupations whose real wage fell was 19% (2016), 40%, 63%, 34%, 7% (2020), **81% (2021), 85% (2022)**, 32%, 40%, 39% (2025). Across 2015–2025 the median US occupation's real wage rose 0.8% in total, and 46% of the 713 occupations present in both years ended lower.

**JOLTS exposure.** JOLTS has no occupation dimension, so each occupation gets an *exposure* rate: its industries' annual-average rates weighted by where its employment sits in the national occupation × sector OEWS panel (`src/build_macro_features.py`). OEWS sectors map to the closest JOLTS industry (wholesale trade takes the broader trade, transportation and utilities rate; agriculture has none because JOLTS is nonfarm). Three rates are used (openings, quits, layoffs and discharges), averaged over the calendar year of the OEWS release, which is known before the next May release. Florida occupations use the same national industry rates — BLS publishes no Florida industry detail — and the Florida total-nonfarm openings rate enters separately as a regional feature (`jolts_region_openings_rate`). Each sector row in the industry table also carries its own sector's three rates. This measures the tightness of the industries an occupation works in, not of the occupation itself.

**Do they help the forecaster? No.** `src/feature_set_experiment.py` re-ran the model selection described above with five candidate input sets, on identical rows and only on the inner walk-forward folds (the 2024 → 2025 holdout was not used for this decision, so the holdout number reported for the shipped model is unaffected):

| Inputs | Mean median error | Mean RMSE |
|---|---|---|
| OEWS-only (shipped) | **7.64%** | **17,816** |
| + real wage growth | 7.67% | 17,831 |
| + JOLTS levels | 7.83% | 18,048 |
| + JOLTS relative to that year's mean | 7.70% | 18,399 |
| + real wage growth + JOLTS relative | 7.66% | 18,525 |
| *no-change baseline* | *7.96%* | *20,130* |

None improved on the OEWS-only inputs. A likely reason (my interpretation, not tested separately) is that JOLTS and CPI mostly vary by *year*, and with only nine feature years a tree can't learn what a year-wide level means; the "relative" variants strip that out and still don't help. The shipped model therefore stays OEWS-only, and the dashboard says so. The difference between the candidates is small compared with the fold-to-fold noise, so this reads as "no evidence they help", not "they hurt".

## Displacement risk scoring

The full formula from the original brief is:

```
risk = 0.4*(contraction relative to national avg) + 0.3*(wage compression) 
     + 0.2*(low hiring demand) + 0.1*(high separation rate)
```

The wage-compression and separation-rate inputs now exist as real-wage and JOLTS data, but the formula was not implemented: it is unvalidated, the hiring-demand input needs job-posting volume (which this build doesn't collect, 20% of the weight), and the experiment above found no evidence that these signals predict employment changes. The Predictive Scenarios tab uses a simplified three-tier `LOW/MODERATE/HIGH` threshold on forecast growth alone (below −2% = MODERATE, below −10% = HIGH), clearly labeled as such in the UI. The thresholds are arbitrary and unvalidated. Applied to the shipped model's forecasts from 2025 features, 1,323 of 1,550 forecastable occupation-region rows come out LOW, 227 MODERATE and none HIGH.

## Known limitations

- **Annual, not monthly** — trend charts and forecasts move once a year, not continuously.
- **OEWS isn't designed for year-over-year comparison.** BLS's FAQ says the survey produces cross-sectional estimates and is "less useful for looking at changes over time", citing classification changes, methodology changes, and its three-year pooled sample design ([BLS](https://www.bls.gov/oes/oes_ques.htm)). Each release combines six semiannual panels collected over three years; the May 2025 estimates use May 2025, Nov 2024, May 2024, Nov 2023, May 2023 and Nov 2022, so consecutive releases share four of their six panels. YoY changes here are therefore changes in a moving average, and a forecast of "next year's employment" is a forecast of the next such estimate. Everything in this project that is a growth rate inherits this. A horizon test of how much of the forecaster's edge could come from that overlap was inconclusive (see the Model section).
- **Occupation codes change between releases.** US codes 2018 → 2019: 808 → 789, with 98 dropped and 79 new; 2020 → 2021: 30 dropped, 72 new. Only 700 US occupation codes exist in all 11 years. YoY is blank where the prior year is absent (matched on the actual prior year, not the previous row — an earlier positional `shift` silently spanned gaps for ~0.4% of occupation and ~1.8% of sector values).
- **Small-occupation noise** — occupations with employment in the tens/hundreds show large YoY % swings (OEWS sampling error dominates at that scale; the standard deviation of next-year growth is about 61 points for occupations under 1,000 employees vs. 7 above 50,000). At the industry-sector cross-tab level this got bad enough to look like a data bug: in the 2024 release, occupations reported at n≈130 within a sector (e.g. "Self-Enrichment Teachers" under NAICS Manufacturing — real BLS data, likely in-house corporate trainers at a diversified manufacturer, but confusing out of context) were showing up in the sector tab's "top movers" tables purely from sampling noise. The Industry & Sector Analysis tab's ranked tables now apply a 500-employee floor (`SECTOR_RANKING_MIN_EMPLOYMENT` in `app/app.py`) before ranking; the scatter plot still shows every occupation (small ones are naturally de-emphasized by bubble size). The floor applies to **both** years being compared: a floor on current-year employment alone let a jump from a tiny base straight through (Construction Laborers in Accommodation & Food Services, 2024: 120 → 960 employees, +700%). The prior-year count is recovered exactly from the YoY percentage. Notebook 1's national occupation-level "top movers" tables have no floor and show the effect.
- **Estimates carried forward unchanged** — for some smaller Florida occupations BLS publishes the identical employment figure in consecutive releases (2024 release: 13 of 744 comparable occupations vs. 2 nationally; 2025 release: 11 of 737 vs. 3 of 790). In 2024 that put Florida's median employment growth at exactly 0.0%, and it is −0.25% in 2025. The Regional Comparison tab therefore shows the median *with 25th–75th percentile whiskers* rather than a median-only bar, which read as a broken chart; the whiskers are scoped to the IQR (not min/max) because a handful of small occupations swing from about −82% to +300% and stretched a box plot's axis until the IQR was unreadable.
- **"Wage compression" is measured as real wage decline, not spread.** The dashboard reports whether wages kept up with inflation. It does not measure narrowing of pay differentials across the wage distribution (the 90th-percentile wage is null for the highest-paid occupations, so a spread measure would silently drop them).
- **Real wages use one national price index.** Florida wages are deflated with the national CPI-U; a Florida-specific price index isn't part of this build, so regional real-wage comparisons reflect nominal differences plus the same inflation rate.
- **JOLTS is industry-level only**, not occupation-specific, and state-level only for Florida total nonfarm. Occupation JOLTS figures are employment-weighted industry averages and Florida occupations use national industry rates. The CPI series has no October 2025 value, which doesn't affect this build because it uses May values.
- **No job-postings data** — the original brief's Indeed/LinkedIn scraping scope was deliberately deferred (cost/ToS considerations); skill-diversity and posting-volume features don't exist in this build.
- **No fairness/equity audit yet** — the original brief's approach of inferring gender/age from job-posting text patterns is methodologically weak and there's no job-postings data source in this build anyway; needs a sounder approach before it's built.
- **Regional wage premium finding**: Florida pays *less*, not more, than the national figure — contrary to the original brief's assumed +3-8% premium. Kept as a genuine finding, not corrected; see the next section for exactly what the numbers do and don't say.

## Florida vs. national wages

Two different comparisons produce two different magnitudes, and the README/dashboard should be read with the distinction in mind (all figures: 2025 OEWS, nominal annual mean wage, not cost-of-living adjusted):

- **Median across occupations** (what the Regional Comparison tab's wage chart shows): the median occupation's mean annual wage is $58,730 in Florida vs. $66,290 nationally, a **−11.4%** gap, over the 790 occupations present in both regions.
- **Head-to-head, same occupation** (FL wage ÷ US wage, per occupation, 777 occupations with a wage in both): the median ratio is **−7.7%** (25th–75th percentile −13.5% to −2.5%), **84%** of occupations pay less in Florida, and weighting by Florida employment gives **−6.5%**.

The direction is robust either way; the ~8% head-to-head figure is the better estimate of "how much less does the same job pay in Florida," while the ~11% is the gap between two distribution medians, which is not the same thing as the median of the per-occupation gaps. Neither adjusts for cost of living, so neither says anything about purchasing power — that comparison would need a regional price index that isn't part of this build.
