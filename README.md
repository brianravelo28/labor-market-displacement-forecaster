# Occupation Displacement & Wage Compression Forecaster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/dashboard-Plotly%20Dash-blue.svg)](https://dash.plotly.com/)

A labor-market analytics pipeline forecasting occupation-level employment trends and flagging wage compression, built on real BLS data (not synthetic) — with a South Florida regional lens. **Status: in progress** — 4 of 5 dashboard tabs built and browser-verified; see [Status & Limitations](#status--limitations) below for exactly what's done vs. pending.

## Overview

Which jobs are shrinking, and which are quietly losing purchasing power even while nominal pay rises? This project pulls BLS Occupational Employment and Wage Statistics (OEWS) into a real 10-year annual panel (~800 occupations, national + Florida), trains a LightGBM model to forecast next-year employment, and ships an interactive dashboard for exploring the results — including a transparent, documented what-if scenario tool.

**Real, verified results:**
- 📊 **10-year OEWS panel**: 2015–2024, ~800–830 occupations/year, national + Florida
- 🤖 **Employment forecaster**: LightGBM, test RMSE ≈ 16,500, MAPE ≈ 12.1%, median error ≈ 7.4%
- 🐛 **A real bug found and fixed mid-build**: an earlier model version mispredicted the largest US occupation by ~60%, even on training data — see [docs/METHODOLOGY.md](docs/METHODOLOGY.md#model) for the root cause and fix (predicting growth rate instead of absolute employment level)
- 🗺️ **Unexpected finding**: Florida's median wage runs ~9% *below* the national median, contrary to the cost-of-living premium the project originally assumed
- 🏭 **Industry-sector breakdown**: 20 NAICS sectors × ~800 occupations × 10 years (~84K rows), national level

## Quick Start

```bash
git clone https://github.com/yourusername/labor-market-displacement-forecaster.git
cd labor-market-displacement-forecaster
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Download BLS OEWS annual archives, 2015-2024 (~800MB total, ~30-40 min the first time)
python data/fetch_oews.py
python data/fetch_oews_industry.py

# Engineer growth/trend features
python src/build_features.py
python src/build_industry_features.py

# Train the employment forecaster
python src/train_model.py

# Launch the dashboard
python app/app.py
# -> http://localhost:8051
```

Optional, currently blocked by a BLS API outage (see [Status & Limitations](#status--limitations)):
```bash
python data/fetch_jolts.py   # industry-level vacancy/hires/separations rates
python data/fetch_cpi.py     # CPI-U, for inflation-adjusted wage compression
```

### Explore the notebooks

```bash
pip install -r requirements-dev.txt   # adds jupyter + matplotlib on top of requirements.txt
jupyter notebook notebooks/
```
- `01_eda_occupation_panel.ipynb` — employment/wage distributions, regional comparison, growth trends
- `02_model_evaluation.ipynb` — model metrics, feature importance, the Retail Salespersons bug case study

## Dashboard

4 tabs, each browser-verified against real data:

| Tab | What it shows |
|---|---|
| **Occupation Search & Risk** | Search any occupation, see employment/wage history and YoY trend, US or FL |
| **Regional Comparison** | US vs. Florida: employment growth, wage level, wage-decline and contraction prevalence |
| **Industry & Sector Analysis** | Pick a NAICS sector, see wage-growth-vs-employment-change scatter and top movers |
| **Predictive Scenarios** | GDP contraction / automation / immigration sliders applied to the trained model's forecast, with documented linear-assumption transparency (no macro elasticity model exists here — treat as directional) |

A 5th tab (Fairness & Equity audit) is **not built** — the original plan's approach of inferring gender/age from job-posting text is methodologically weak, and this build has no job-postings data source at all. Needs a sounder methodology before implementation.

## Deployment (Render)

The repo is Render-ready as-is:

1. On [render.com](https://render.com), **New > Blueprint**, point it at this repo — `render.yaml` configures the build (`pip install -r requirements.txt`) and start (`gunicorn app.app:server`) commands automatically.
2. Or set up a **New > Web Service** manually with those same two commands, Python 3.11+, free plan is fine.

No environment variables or secrets are required — the dashboard reads only the processed CSVs and trained model already committed under `data/processed/` (see [.gitignore](.gitignore) for why those, unlike raw BLS archives, are tracked: Render has no reliable way to regenerate them at build time against BLS's API, which has had multi-day outages during development).

`app/app.py`'s `if __name__ == "__main__"` block still works for local dev (`python app/app.py`, reads `PORT`/`DASH_DEBUG` env vars, defaults to `localhost:8051` with debug on) — gunicorn is only used in production per the `Procfile`.

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
├── src/                    # feature engineering + model training
├── notebooks/              # EDA and model evaluation (executed, real outputs)
├── app/                    # Plotly Dash dashboard
└── docs/                   # schema, methodology, data source citations
```

See [docs/SCHEMA.md](docs/SCHEMA.md) for exact column definitions and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full data-architecture and modeling writeup.

## Data Sources

- **BLS OEWS** (annual employment/wage by occupation): downloaded directly as archive files, not via the timeseries API — BLS doesn't retain multi-year OEWS history under stable series IDs. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for how this was verified.
- **BLS JOLTS** (monthly vacancy/hires/separations, industry-level only — not occupation-level): pending, blocked by a BLS API outage during development.
- **BLS CPI-U** (for real/inflation-adjusted wages): pending, same blocker.

Full citations and API documentation links: [docs/CITATIONS.md](docs/CITATIONS.md).

## Status & Limitations

- **Done**: OEWS annual panel + engineered features, industry-sector breakdown, LightGBM employment forecaster (bug-fixed), Dashboard Tabs 1–4, 2 executed EDA/evaluation notebooks.
- **Blocked, not skipped**: JOLTS and CPI fetch scripts exist and are tested against the live BLS API, but a multi-day BLS API outage during development prevented completing those pulls. Re-run `data/fetch_jolts.py` and `data/fetch_cpi.py` once BLS is reachable, then re-run the feature/model scripts to pick up real-wage compression and a fuller displacement risk score.
- **Not started**: Fairness/equity audit (Tab 5) — needs a methodology decision, not just more data.
- **Known data-quality caveat**: occupations with employment in the tens/hundreds show large YoY % swings from OEWS sampling variance — visible in "top movers" tables throughout. Treat single-occupation extremes at small scale with caution.
- **Cadence**: annual, not monthly (OEWS is a point-in-time annual survey; see methodology doc for why the original monthly-cadence plan didn't match reality).

## License

MIT License — see [LICENSE](LICENSE).
