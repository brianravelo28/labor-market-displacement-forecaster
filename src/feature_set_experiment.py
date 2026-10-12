"""
Do CPI- and JOLTS-derived features help the employment forecaster?

Each candidate feature set gets the same treatment as the shipped model in
train_model.py: hyperparameters are chosen on the inner walk-forward folds
(features 2020-2023), and the sets are compared on those same folds. The
2024 -> 2025 holdout is deliberately not touched here, so the comparison can't
leak into the holdout number reported for the shipped model.

All candidates are scored on identical rows (pairs with every candidate input
present), so differences come from the features, not from which rows survive
dropna.

Candidates:
  A  baseline        the 8 shipped OEWS-only inputs
  B  + real wage     + real_wage_growth_yoy_pct
  C  + JOLTS levels  + industry-exposure openings/quits/layoffs rates and the
                     region's total-nonfarm openings rate
  E  + JOLTS relative  the same four JOLTS inputs as deviations from that
                     year's cross-occupation mean, so they carry only
                     cross-sectional information (eleven years can't teach a
                     tree what a year-wide level means)
  D  + real + JOLTS rel.

Result is written to data/processed/feature_set_experiment.json.
"""
import json

import pandas as pd

import train_model as tm
from config import DATA_PROCESSED

BASE = list(tm.FEATURE_COLS)
REAL = ["real_wage_growth_yoy_pct"]
JOLTS = [
    "jolts_exposure_openings_rate",
    "jolts_exposure_quits_rate",
    "jolts_exposure_layoffs_rate",
    "jolts_region_openings_rate",
]
JOLTS_REL = [c + "_rel" for c in JOLTS]

CANDIDATES = {
    "A baseline (shipped)": BASE,
    "B + real wage growth": BASE + REAL,
    "C + JOLTS levels": BASE + JOLTS,
    "E + JOLTS relative to year mean": BASE + JOLTS_REL,
    "D + real wage + JOLTS relative": BASE + REAL + JOLTS_REL,
}


def main():
    features = pd.read_csv(DATA_PROCESSED / "oews_occupation_features.csv")
    for c in JOLTS:
        features[c + "_rel"] = features[c] - features.groupby("year")[c].transform("mean")

    tm.FEATURE_COLS = BASE + REAL + JOLTS + JOLTS_REL  # build_supervised drops rows missing any of these
    data = tm.build_supervised(features)
    print(f"{len(data):,} pairs, identical for every candidate; inner folds {tm.INNER_FOLD_YEARS}\n")

    results = {}
    for name, cols in CANDIDATES.items():
        tm.FEATURE_COLS = cols
        best, _ = tm.select_params(data)
        folds = tm.walk_forward(data, tm.INNER_FOLD_YEARS, best)
        mean = tm.mean_over_folds(folds)
        results[name] = {
            "features": cols,
            "selected_params": best,
            "mean_median_error_pct": mean["model"]["median_ape"],
            "mean_rmse": mean["model"]["rmse"],
            "persistence_mean_median_error_pct": mean["persistence"]["median_ape"],
            "persistence_mean_rmse": mean["persistence"]["rmse"],
            "per_fold_median_error_pct": {str(T): f["model"]["median_ape"] for T, f in folds.items()},
        }
        print(f"{name:34s} median error {mean['model']['median_ape']:.2f}%   RMSE {mean['model']['rmse']:>8,.0f}")
    print(f"{'(no-change baseline)':34s} median error {mean['persistence']['median_ape']:.2f}%   RMSE {mean['persistence']['rmse']:>8,.0f}")

    (DATA_PROCESSED / "feature_set_experiment.json").write_text(
        json.dumps({"pairs": int(len(data)), "inner_fold_years": tm.INNER_FOLD_YEARS, "candidates": results}, indent=2)
    )


if __name__ == "__main__":
    main()
