# Occupation Displacement & Wage Compression Forecaster

![Occupation Displacement Forecaster dashboard](docs/screenshot.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/dashboard-Plotly%20Dash-blue.svg)](https://dash.plotly.com/)

A labor-market analytics pipeline for occupation-level employment and wages, built on real BLS data (not synthetic), comparing the US with Florida as a whole. **Status: in progress** — 4 of 5 dashboard tabs are built and deployed; see [Status & Limitations](#status--limitations) for exactly what's done vs. pending.

**Live dashboard: [labor-market-displacement-forecaster.onrender.com](https://labor-market-displacement-forecaster.onrender.com/)** — hosted on Render's free tier, which spins the app down when idle, so the first load after a quiet period can take up to a minute.

## Overview

Which jobs are shrinking, and which are quietly losing purchasing power even while nominal pay rises? This project pulls BLS Occupational Employment and Wage Statistics (OEWS) into an 11-year annual panel (~800 occupations, national + statewide Florida), adds CPI-U inflation adjustment and JOLTS job-market tightness, trains a LightGBM model to forecast next-year employment, and ships an interactive dashboard for exploring the results — including a transparent, documented what-if scenario tool.

**Results:**
- 📊 **11-year OEWS panel**: 2015–2025, 789–831 occupations per year (17,587 occupation × region × year rows), national + statewide Florida
- 🤖 **Employment forecaster, validated against "assume no change"**: LightGBM, walk-forward validated. On the 2025 OEWS release, which it never saw, the median error is 6.1% vs. 6.4% for simply assuming next year equals this year, and it is closer on 54% of 1,503 occupation-region rows. The edge is small and statistically unclear (95% bootstrap interval for the difference in median error: −0.6 to +0.0 points; in the four earlier walk-forward folds it was ahead on RMSE in three and on median error in three). Every model number in this repo is reported next to that baseline. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#model).
- 💵 **Nominal raises mostly didn't keep up with inflation in 2021–22**: deflating each year's wages with May CPI-U, more than 80% of US occupations had a real wage decline in both 2021 and 2022 (85% in 2022), against 7% in 2020 and 32% in 2023. Across the whole 2015–2025 span the median US occupation's real wage rose just 0.8%, and 46% of occupations ended lower. (Retail Salespersons' real wage is essentially flat for a decade.) In 2025, 39% of US occupations and 45% of Florida occupations had a real wage decline.
- 📈 **JOLTS tightness merged in, but it didn't make the forecast better**: JOLTS publishes job openings, quits and layoffs by industry only, so each occupation gets the rates of its industries weighted by where its employment sits. I tested real-wage and JOLTS inputs in the forecaster on the same walk-forward folds; none beat the OEWS-only inputs (median error 7.64% without them, 7.66–7.83% with them), so the shipped model doesn't use them. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#cpi-and-jolts).
- 🐛 **Two earlier model versions failed and were fixed**: a level regression mispredicted the largest US occupation by ~60%, and its replacement trained on a single year (a feature that is empty before 2020 silently dropped every earlier row) and lost to "no change". Details in the methodology doc.
- 🗺️ **Unexpected finding**: Florida pays less than the national figure for the same job, not more. Compared head-to-head on 777 occupations, Florida's mean annual wage is a median 7.7% lower, and 84% of those occupations pay less in Florida (6.5% lower when weighted by Florida employment). The Regional Comparison tab's −11.4% is a different quantity: the gap between the two *medians across occupations*. All nominal — not cost-of-living adjusted — contrary to the +3–8% regional premium the original project brief assumed. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#florida-vs-national-wages).
- 🏭 **Industry-sector breakdown**: 20 NAICS sectors (143–641 occupations each in 2025, ~426 on average), 92,104 occupation × sector × year rows, national level

## Quick Start

```bash
git clone https://github.com/brianravelo28/labor-market-displacement-forecaster.git
cd labor-market-displacement-forecaster
pip install -r requirements.txt
python app/app.py
# -> http://localhost:8051
```

The processed datasets and trained model are committed under `data/processed/`, so the dashboard runs immediately — no pipeline run required.

### Rebuild everything from raw BLS data (optional)

```bash
# Download BLS OEWS annual archives, 2015-2025 (~900MB total, ~30-40 min the first time)
python data/fetch_oews.py
python data/fetch_oews_industry.py

# BLS API pulls: JOLTS rates (85 series) and CPI-U (10 queries; the unregistered
# limit is 25 queries/day and every response is cached under data/raw/jolts/)
python data/fetch_jolts.py
python data/fetch_cpi.py

# Engineer growth/trend features, then merge in real wages and JOLTS exposure
python src/build_features.py
python src/build_industry_features.py

# Walk-forward validation, baseline comparison, and refit of the forecaster
# (writes the model and data/processed/model_metrics.json)
python src/train_model.py

# Optional: re-run the test of whether CPI/JOLTS inputs help the forecaster
python src/feature_set_experiment.py
```

The scripts self-identify to BLS with a `User-Agent` (BLS serves an "Access Denied" page to generic browser user agents). Before running the downloads, create a `.env` file in the repo root containing `BLS_CONTACT_EMAIL=you@example.com` (read by `src/config.py`; `.env` is git-ignored). `BLS_API_KEY` is optional; with a registered key the API allows far more queries and longer windows.

The CPI and JOLTS pulls go through BLS's timeseries API, which returns at most 10 years per query without a key, so each series is fetched in two year-windows (2015–2024, 2025–now) and merged.

### Explore the notebooks

```bash
pip install -r requirements-dev.txt   # adds jupyter + matplotlib on top of requirements.txt
jupyter notebook notebooks/
```
- `01_eda_occupation_panel.ipynb` — employment/wage distributions, regional comparison, growth trends
- `02_model_evaluation.ipynb` — walk-forward results against the no-change baseline, the 2025 holdout, feature importance, example occupations

## Dashboard

4 tabs, each verified against real data (locally and in production):

| Tab | What it shows |
|---|---|
| **Occupation Search & Risk** | Search any occupation, see employment history, nominal vs. inflation-adjusted wage history, YoY trends (employment, nominal wage, real wage), a contraction flag (employment down >2% YoY), and the JOLTS job-openings/quits/layoffs rates of the industries it works in, US or FL. A full displacement *risk score* is not implemented — see [Status & Limitations](#status--limitations). |
| **Regional Comparison** | US vs. Florida: employment growth (median with 25th–75th percentile whiskers, since a median-only bar hid how much occupations actually vary), median wage level, the share of occupations with nominal and with real (after-CPI) wage decline, the share contracting, and the JOLTS job-openings rate |
| **Industry & Sector Analysis** | Pick a NAICS sector, see its JOLTS openings, quits and layoffs rates, a wage-growth-vs-employment-change scatter and top movers. Occupations under 500 employees within a sector — in either year compared — are excluded from the ranked tables (sampling noise), though still shown in the scatter |
| **Predictive Scenarios** | The model's baseline forecast (with a note on how well it validated, and that it doesn't use CPI or JOLTS) plus GDP contraction / automation / immigration sliders. The sliders are documented linear assumptions, not fitted elasticities — treat as directional |

A 5th tab (Fairness & Equity audit) is **not built** — the original plan's approach of inferring gender/age from job-posting text is methodologically weak, and this build has no job-postings data source at all. Needs a sounder methodology before implementation.

## Deployment (Render)

The repo is Render-ready as-is:

1. On [render.com](https://render.com), **New > Blueprint**, point it at this repo — `render.yaml` configures the build (`pip install -r requirements.txt`) and start command.
2. Or create a **New > Web Service** manually (Python 3.11+, free plan is fine) with build command `pip install -r requirements.txt` and start command:

   ```bash
   gunicorn app.app:server --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 4 --timeout 120
   ```

**Why a threaded worker:** Dash fires every tab's callbacks as simultaneous requests on page load. With gunicorn's default single sync worker those requests queue one at a time, and on Render's free tier the LightGBM-loading Predictive Scenarios callback could wait long enough to hit the 30-second default timeout — surfacing as intermittent `502`s and a blank tab. Threads share the already-loaded data/model inside one process (unlike extra worker processes, which would each duplicate it).

**Gotcha:** for a service created manually, the Start Command lives in Render's dashboard settings — editing `Procfile`/`render.yaml` later does *not* change an existing service. Update it under **Settings > Start Command**.

No environment variables or secrets are required — the dashboard reads only the processed CSVs and trained model committed under `data/processed/` (unlike the raw BLS archives, which are git-ignored; see [.gitignore](.gitignore) for why: Render can't reliably regenerate them at build time, since BLS's API has had multi-day outages).

`app/app.py`'s `if __name__ == "__main__"` block is for local dev (`python app/app.py`; reads `PORT`/`DASH_DEBUG` env vars, defaults to port 8051 with debug on). In production, gunicorn serves `app.app:server` per the `Procfile`.

## Project Structure

```
labor-market-displacement-forecaster/
├── README.md              # this file
├── requirements.txt        # runtime deps (what Render installs)
├── requirements-dev.txt    # + jupyter/matplotlib for notebooks
├── Procfile                 # gunicorn start command
├── render.yaml              # Render Blueprint config
├── LICENSE                 # MIT
├── .gitignore
├── data/                   # acquisition scripts (BLS OEWS/JOLTS/CPI fetchers) + processed/ (tracked, see Deployment)
├── src/                    # feature engineering (incl. CPI/JOLTS merge), model training, feature-set experiment
├── notebooks/              # EDA and model evaluation (executed, real outputs)
├── app/                    # Plotly Dash dashboard
└── docs/                   # schema, methodology, data source citations
```

See [docs/SCHEMA.md](docs/SCHEMA.md) for exact column definitions and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full data-architecture and modeling writeup.

## Data Sources

- **BLS OEWS** (annual employment/wage by occupation): downloaded directly as archive files, not via the timeseries API — BLS doesn't retain multi-year OEWS history under stable series IDs. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for how this was verified.
- **BLS JOLTS** (monthly job openings, hires, quits, layoffs and separations rates, industry-level only — not occupation-level; state-level only for Florida total nonfarm): pulled through the timeseries API (US 16 industries and Florida total nonfarm, 2015 to 2026-08 for the US and through 2025-12 for Florida) and merged into the occupation and industry feature tables.
- **BLS CPI-U** (all items, US city average, not seasonally adjusted, series `CUUR0000SA0`): used to express wages in constant 2025 dollars, deflating each OEWS year with May CPI.

Full citations and API documentation links: [docs/CITATIONS.md](docs/CITATIONS.md).

## Status & Limitations

- **Done**: OEWS annual panel (2015–2025) + engineered features, CPI-adjusted real wages and JOLTS tightness merged into the occupation and industry tables and shown on Tabs 1–3, industry-sector breakdown, LightGBM employment forecaster with walk-forward validation against a no-change baseline, dashboard Tabs 1–4 deployed on Render, 2 executed EDA/evaluation notebooks.
- **Tested, not adopted**: real-wage and JOLTS features as forecaster inputs. They did not improve out-of-sample accuracy (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md#cpi-and-jolts)), so the forecast still runs on OEWS-only inputs.
- **Not implemented**: a displacement risk *score*. Today the Predictive Scenarios tab only applies a simple growth-rate threshold (low/moderate/high); the formula in the original brief also needs job-postings data this build doesn't collect. Also not built: "wage compression" in the sense of narrowing pay differentials across the wage distribution — the dashboard reports real wage *decline* (purchasing power), not spread.
- **Not started**: Fairness/equity audit (Tab 5) — needs a methodology decision, not just more data.
- **OEWS is not built for year-over-year comparison.** BLS's own FAQ says the survey is designed for cross-sectional estimates and is "less useful for looking at changes over time", citing classification changes and its three-year pooled sample ([BLS: Can OEWS data be used to look at changes over time?](https://www.bls.gov/oes/oes_ques.htm)). Every growth rate on the dashboard and every forecast inherits that: the numbers are best read as directional, and consecutive releases share most of their underlying survey panels.
- **Known data-quality caveats**: occupations with employment in the tens/hundreds show large YoY % swings from OEWS sampling variance — the sector tab filters these out of its rankings, but the notebook's top-movers tables don't. Some BLS estimates are carried forward unchanged between releases (in the 2024 release it pushed Florida's median employment growth to exactly 0.0%). Details in [docs/METHODOLOGY.md](docs/METHODOLOGY.md#known-limitations).
- **Cadence**: annual, not monthly (OEWS is a point-in-time annual survey; see methodology doc for why the original monthly-cadence plan didn't match reality).

## License

MIT License — see [LICENSE](LICENSE).
