# Occupation Employment & Real Wage Tracker

![Occupation Employment & Real Wage Tracker dashboard](docs/screenshot.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/dashboard-Plotly%20Dash-blue.svg)](https://dash.plotly.com/)
[![tests](https://github.com/brianravelo28/labor-market-displacement-forecaster/actions/workflows/tests.yml/badge.svg)](https://github.com/brianravelo28/labor-market-displacement-forecaster/actions/workflows/tests.yml)

A labor-market analytics pipeline and dashboard for occupation-level employment and wages, built on real BLS data (not synthetic), comparing the US with Florida as a whole. **Status: complete** — the four-tab dashboard is built and deployed; see [Status & Limitations](#status--limitations) for the data caveats.

**Live dashboard: [labor-market-displacement-forecaster.onrender.com](https://labor-market-displacement-forecaster.onrender.com/)** — hosted on Render's free tier, which spins the app down when idle, so the first load after a quiet period can take up to a minute.

## Overview

Which jobs are shrinking, and which are quietly losing purchasing power even while nominal pay rises? This project pulls BLS Occupational Employment and Wage Statistics (OEWS) into an 11-year annual panel (~800 occupations, national + statewide Florida), adds CPI-U inflation adjustment and JOLTS job-market tightness, and ships an interactive dashboard for exploring the results. It describes what the data show; it makes no forecasts.

**Results:**
- 📊 **11-year OEWS panel**: 2015–2025, 789–831 occupations per year (17,587 occupation × region × year rows), national + statewide Florida
- 💵 **Nominal raises mostly didn't keep up with inflation in 2021–22**: deflating each year's wages with May CPI-U, more than 80% of US occupations had a real wage decline in both 2021 and 2022 (85% in 2022), against 7% in 2020 and 32% in 2023. Across the whole 2015–2025 span the median US occupation's real wage rose just 0.8%, and 46% of occupations ended lower. (Retail Salespersons' real wage is essentially flat for a decade.) In 2025, 39% of US occupations and 45% of Florida occupations had a real wage decline.
- 📉 **The 2020 shock was very uneven, and the recovery has been too**: total employment in the 2020 release fell 5.3% nationally (4.2% in Florida) against a pre-2020 trend of about +1.6% a year. Against their own earlier trends, Waiters and Waitresses fell 25 points more, Retail Salespersons 14, Registered Nurses 2. For the occupations present in every release, employment in 2025 is 103% of its 2019 level nationally and 110% in Florida. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#2020-shock-and-recovery).
- 📈 **JOLTS tightness merged in**: JOLTS publishes job openings, quits and layoffs by industry only, so each occupation gets the rates of its industries weighted by where its employment sits (Florida occupations use the national industry rates, since BLS publishes Florida JOLTS for total nonfarm only). See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#cpi-and-jolts).
- 🗺️ **Unexpected finding**: Florida pays less than the national figure for the same job, not more. Compared head-to-head on 777 occupations, Florida's mean annual wage is a median 7.7% lower, and 84% of those occupations pay less in Florida (6.5% lower when weighted by Florida employment). The Regional Comparison tab's −11.4% is a different quantity: the gap between the two *medians across occupations*. Weighting by workers instead of occupations narrows the median-wage gap to −6.3% (and flips 2025 employment growth to +2.1% in Florida vs. +1.0% nationally), because a third of Florida's occupations have under 1,000 employees and their year-over-year changes are far noisier than those of large occupations. All nominal — not cost-of-living adjusted — contrary to the +3–8% regional premium the original project brief assumed. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md#florida-vs-national-wages).
- 🏭 **Industry-sector breakdown**: 20 NAICS sectors (143–641 occupations each in 2025, ~426 on average), 92,104 occupation × sector × year rows, national level

## Quick Start

```bash
git clone https://github.com/brianravelo28/labor-market-displacement-forecaster.git
cd labor-market-displacement-forecaster
pip install -r requirements.txt
python app/app.py
# -> http://localhost:8051
```

The processed datasets are committed under `data/processed/`, so the dashboard runs immediately — no pipeline run required.

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

# Observed 2020 response per occupation, for the 2020 tab
python src/build_shock_replay.py
```

The scripts self-identify to BLS with a `User-Agent` (BLS serves an "Access Denied" page to generic browser user agents). Before running the downloads, create a `.env` file in the repo root containing `BLS_CONTACT_EMAIL=you@example.com` (read by `src/config.py`; `.env` is git-ignored). `BLS_API_KEY` is optional; with a registered key the API allows far more queries and longer windows.

The CPI and JOLTS pulls go through BLS's timeseries API, which returns at most 10 years per query without a key, so each series is fetched in two year-windows (2015–2024, 2025–now) and merged.

### Run the tests

```bash
pip install -r requirements.txt pytest   # or requirements-dev.txt
python -m pytest
```

About 70 tests, a few seconds, no network access needed. They cover the year-matched lag logic and growth features, the CPI and JOLTS merges, the 2020 response calculation, the BLS fetchers (series IDs, the API's empty-success quirk, 10-year windows, 2015-style OEWS headers) and every dashboard callback. They also check the committed data: unique keys, every year present, real wages equal to nominal in the base year, and the 2020 response table matching what the current data produce. GitHub Actions runs them on Python 3.9 and 3.11 for every push and pull request. What they do not cover: the layout and look of the pages, and the live Render deployment.

### Explore the notebooks

```bash
pip install -r requirements-dev.txt   # adds jupyter + matplotlib on top of requirements.txt
jupyter notebook notebooks/
```
- `01_eda_occupation_panel.ipynb` — employment/wage distributions, regional comparison, growth trends, real wages

## Dashboard

4 tabs, each verified against real data (locally and in production):

| Tab | What it shows |
|---|---|
| **Occupation Search** | Search any occupation, see employment history, nominal vs. inflation-adjusted wage history, YoY trends (employment, nominal wage, real wage), a contraction flag (employment down >2% YoY), and the JOLTS job-openings/quits/layoffs rates of the industries it works in, US or FL. |
| **Regional Comparison** | US vs. Florida, counting either each occupation or each worker (employment-weighted) equally via a toggle: employment growth (median with 25th–75th percentile whiskers, since a median-only bar hid how much occupations actually vary), median wage level, the share of occupations with nominal and with real (after-CPI) wage decline, the share contracting, and the JOLTS job-openings rate |
| **Industry & Sector Analysis** | Pick a NAICS sector, see its JOLTS openings, quits and layoffs rates, a real-or-nominal-wage-growth-vs-employment-change scatter (real by default; the toggle matters mainly for where the zero line falls) and top movers. Occupations under 500 employees within a sector — in either year compared — are excluded from the ranked tables (sampling noise), though still shown in the scatter |
| **2020 Shock & Recovery** | Pick an occupation and region: employment indexed to the 2019 release (= 100) against the total for the occupations present in every release, the change in the 2020 release, employment now vs. 2019, and the occupation's 2020 response (its change minus its 2016–2019 trend); plus tables of the hardest-hit and most resilient occupations. Occupations under 10,000 employees show their occupation group's response instead, and the card says so |

## Deployment (Render)

The repo is Render-ready as-is:

1. On [render.com](https://render.com), **New > Blueprint**, point it at this repo — `render.yaml` configures the build (`pip install -r requirements.txt`) and start command.
2. Or create a **New > Web Service** manually (Python 3.11+, free plan is fine) with build command `pip install -r requirements.txt` and start command:

   ```bash
   gunicorn app.app:server --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 4 --timeout 120
   ```

**Why a threaded worker:** Dash fires every tab's callbacks as simultaneous requests on page load. With gunicorn's default single sync worker those requests queue one at a time, and on Render's free tier a slow one could wait long enough to hit the 30-second default timeout — surfacing as intermittent `502`s and a blank tab. Threads share the already-loaded data inside one process (unlike extra worker processes, which would each duplicate it).

**Gotcha:** for a service created manually, the Start Command lives in Render's dashboard settings — editing `Procfile`/`render.yaml` later does *not* change an existing service. Update it under **Settings > Start Command**.

No environment variables or secrets are required — the dashboard reads only the processed CSVs committed under `data/processed/` (unlike the raw BLS archives, which are git-ignored; see [.gitignore](.gitignore) for why: Render can't reliably regenerate them at build time, since BLS's API has had multi-day outages).

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
├── src/                    # feature engineering (incl. CPI/JOLTS merge and the 2020 response)
├── notebooks/              # EDA (executed, real outputs)
├── app/                    # Plotly Dash dashboard
└── docs/                   # schema, methodology, data source citations
```

See [docs/SCHEMA.md](docs/SCHEMA.md) for exact column definitions and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full data-architecture and methods writeup.

## Data Sources

- **BLS OEWS** (annual employment/wage by occupation): downloaded directly as archive files, not via the timeseries API — BLS doesn't retain multi-year OEWS history under stable series IDs. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for how this was verified.
- **BLS JOLTS** (monthly job openings, hires, quits, layoffs and separations rates, industry-level only — not occupation-level; state-level only for Florida total nonfarm): pulled through the timeseries API (US 16 industries and Florida total nonfarm, 2015 to 2026-08 for the US and through 2025-12 for Florida) and merged into the occupation and industry feature tables.
- **BLS CPI-U** (all items, US city average, not seasonally adjusted, series `CUUR0000SA0`): used to express wages in constant 2025 dollars, deflating each OEWS year with May CPI.

Full citations and API documentation links: [docs/CITATIONS.md](docs/CITATIONS.md).

## Status & Limitations

- **Done**: OEWS annual panel (2015–2025) + engineered features, CPI-adjusted real wages and JOLTS tightness merged into the occupation and industry tables, industry-sector breakdown, the 2020 shock-and-recovery view, dashboard Tabs 1–4 deployed on Render, an executed EDA notebook, tests and CI.
- **OEWS is not built for year-over-year comparison.** BLS's own FAQ says the survey is designed for cross-sectional estimates and is "less useful for looking at changes over time", citing classification changes and its three-year pooled sample ([BLS: Can OEWS data be used to look at changes over time?](https://www.bls.gov/oes/oes_ques.htm)). Every growth rate on the dashboard inherits that: the numbers are best read as directional, and consecutive releases share most of their underlying survey panels.
- **Known data-quality caveats**: occupations with employment in the tens/hundreds show large YoY % swings from OEWS sampling variance — the sector tab filters these out of its rankings, but the notebook's top-movers tables don't. Some BLS estimates are carried forward unchanged between releases (in the 2024 release it pushed Florida's median employment growth to exactly 0.0%). Details in [docs/METHODOLOGY.md](docs/METHODOLOGY.md#known-limitations).
- **Cadence**: annual, not monthly (OEWS is a point-in-time annual survey; see methodology doc for why the original monthly-cadence plan didn't match reality).

## License

MIT License — see [LICENSE](LICENSE).
