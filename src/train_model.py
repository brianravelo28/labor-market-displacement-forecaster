"""
LightGBM employment forecaster: forecasts next-year employment_level per
occupation x region from the current year's OEWS-derived features. The
model's target is the log growth ratio log(next / now), not the level
itself; the level is reconstructed as now * exp(prediction) (see
docs/METHODOLOGY.md#model for why).

Every number this model reports is compared with two do-nothing baselines,
because a forecast is only worth something if it beats them:
  - persistence:   next year = this year (growth 0)
  - median growth: every occupation grows by the training-period median

Validation is walk-forward ("rolling origin"). An observation is a pair
(features in year t, employment in year t+1). To forecast from year T, a model
may train only on pairs whose target is already known by year T, i.e. features
from years <= T-1.
  - Inner folds T = 2020..2023 choose the hyperparameters (a small grid).
  - The final fold -- features from the latest year, target the newest OEWS
    release -- is a holdout evaluated once, after the choices are fixed.
  - The shipped model is then refit on every available pair.

Why the 5-year CAGR features are not model inputs: they are null for every
occupation until 2020, so using them silently discards every training pair
before 2020 (the first version of this model trained on a single year because
of that). They remain in oews_occupation_features.csv for description.

JOLTS (vacancy/separation rate) and CPI (real wage) features are not yet
integrated -- this model runs on OEWS-only features.
"""
import itertools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import DATA_PROCESSED

FEATURE_COLS = [
    "employment_level",
    "wage_annual_mean",
    "wage_annual_median",
    "employment_change_yoy_pct",
    "wage_growth_yoy_pct",
    "contraction_flag_12mo",
    "region",
    "occ_major_group",
]
CATEGORICAL_COLS = ["region", "occ_major_group"]
TARGET_COL = "employment_level_next"

INNER_FOLD_YEARS = [2020, 2021, 2022, 2023]

BASE_PARAMS = dict(
    learning_rate=0.05,
    feature_fraction=0.75,
    bagging_freq=1,
    bagging_fraction=0.9,
    random_state=0,
    verbosity=-1,
)
GRID = dict(
    objective=["l2", "huber", "l1"],
    n_estimators=[25, 75, 200],
    num_leaves=[8, 31],
    min_data_in_leaf=[20, 100],
)

METRICS_PATH = DATA_PROCESSED / "model_metrics.json"
MODEL_PATH = DATA_PROCESSED / "employment_forecast_model.txt"


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (region, occupation, year t) that has an observed year t+1."""
    df = df.copy()
    df["occ_major_group"] = df["occ_code"].str[:2].astype("category")
    df["region"] = df["region"].astype("category")
    df["contraction_flag_12mo"] = df["contraction_flag_12mo"].astype("float")

    df = df.sort_values(["region", "occ_code", "year"])
    grp = df.groupby(["region", "occ_code"], observed=True)
    df[TARGET_COL] = grp["employment_level"].shift(-1)
    # Only true adjacent-year pairs: a missing year must not turn into a
    # multi-year "next".
    df = df[grp["year"].shift(-1) == df["year"] + 1]

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).copy()
    # Predicting the level (even log1p(level)) fails at the top end: only a
    # handful of occupations exceed 2M employees, so a level regression
    # mispredicted the largest occupations by 40-60%, even on rows it was
    # trained on. Growth ratios share one scale across occupation sizes.
    df["target_log_growth"] = np.log(df[TARGET_COL]) - np.log(df["employment_level"])
    return df


def make_model(params: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**BASE_PARAMS, **params)


def fit_model(train: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
    model = make_model(params)
    model.fit(train[FEATURE_COLS], train["target_log_growth"], categorical_feature=CATEGORICAL_COLS)
    return model


def predictions(model, data: pd.DataFrame) -> np.ndarray:
    """Predicted next-year employment level."""
    return data["employment_level"].values * np.exp(model.predict(data[FEATURE_COLS]))


def score(data: pd.DataFrame, predicted_level: np.ndarray) -> dict:
    y = data[TARGET_COL].values
    now = data["employment_level"].values
    err = np.abs(predicted_level - y)
    ape = err / y * 100
    big = now >= 1_000_000
    return dict(
        n=int(len(data)),
        rmse=float(np.sqrt(np.mean((predicted_level - y) ** 2))),
        mape=float(ape.mean()),
        median_ape=float(np.median(ape)),
        closer_than_persistence=float(np.mean(err < np.abs(now - y))),
        rmse_1m_plus=float(np.sqrt(np.mean((predicted_level[big] - y[big]) ** 2))) if big.any() else None,
        n_1m_plus=int(big.sum()),
    )


def compare(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> dict:
    """Model vs. the two baselines on one test set."""
    model = fit_model(train, params)
    now = test["employment_level"].values
    median_growth = float(train["target_log_growth"].median())
    return {
        "model": score(test, predictions(model, test)),
        "persistence": score(test, now),
        "median_growth": score(test, now * np.exp(median_growth)),
    }


def walk_forward(data: pd.DataFrame, years: list, params: dict) -> dict:
    return {int(T): compare(data[data["year"] <= T - 1], data[data["year"] == T], params) for T in years}


def mean_over_folds(folds: dict) -> dict:
    out = {}
    for method in ["model", "persistence", "median_growth"]:
        out[method] = {
            k: float(np.mean([f[method][k] for f in folds.values() if f[method][k] is not None]))
            for k in ["rmse", "mape", "median_ape", "closer_than_persistence"]
        }
    return out


def select_params(data: pd.DataFrame) -> tuple:
    """Pick the grid point with the lowest mean median-APE over the inner folds."""
    results = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        folds = walk_forward(data, INNER_FOLD_YEARS, params)
        m = mean_over_folds(folds)["model"]
        results.append({"params": params, **m})
    results.sort(key=lambda r: (r["median_ape"], r["params"]["n_estimators"]))
    return results[0]["params"], results


def bootstrap_holdout(train: pd.DataFrame, test: pd.DataFrame, params: dict, reps: int = 2000) -> dict:
    """95% intervals for model-minus-persistence on the holdout (rows resampled)."""
    model = fit_model(train, params)
    y = test[TARGET_COL].values
    now = test["employment_level"].values
    pred = predictions(model, test)
    rng = np.random.default_rng(0)
    d_med, rmse_ratio = [], []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        d_med.append(np.median(np.abs(pred[i] - y[i]) / y[i] * 100) - np.median(np.abs(now[i] - y[i]) / y[i] * 100))
        rmse_ratio.append(np.sqrt(np.mean((pred[i] - y[i]) ** 2)) / np.sqrt(np.mean((now[i] - y[i]) ** 2)))
    lo_hi = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"median_ape_minus_persistence_pts": lo_hi(d_med), "rmse_ratio_to_persistence": lo_hi(rmse_ratio), "reps": reps}


def _fmt(label, s):
    return (f"  {label:15s} RMSE={s['rmse']:9,.0f}  MAPE={s['mape']:5.1f}%  median APE={s['median_ape']:4.1f}%  "
            f"closer than persistence={s['closer_than_persistence']:.0%}")


if __name__ == "__main__":
    features = pd.read_csv(DATA_PROCESSED / "oews_occupation_features.csv")
    data = build_supervised(features)
    last_feature_year = int(data["year"].max())
    print(f"{len(data):,} adjacent-year pairs; feature years {int(data['year'].min())}-{last_feature_year} "
          f"(targets through {last_feature_year + 1})")

    # 1. Choose hyperparameters on the inner folds only.
    best, grid_results = select_params(data)
    print(f"\nSelected on inner folds {INNER_FOLD_YEARS}: {best}")
    inner = walk_forward(data, INNER_FOLD_YEARS, best)
    for T, f in inner.items():
        print(f"\nInner fold: features {T}, target {T + 1}  (n={f['model']['n']})")
        for method, s in f.items():
            print(_fmt(method, s))
    inner_mean = mean_over_folds(inner)
    print("\nInner-fold mean:")
    for method, s in inner_mean.items():
        print(f"  {method:15s} RMSE={s['rmse']:9,.0f}  MAPE={s['mape']:5.1f}%  median APE={s['median_ape']:4.1f}%")

    # 2. Holdout: latest feature year -> newest release. Evaluated once.
    train_ho, test_ho = data[data["year"] < last_feature_year], data[data["year"] == last_feature_year]
    holdout = compare(train_ho, test_ho, best)
    boot = bootstrap_holdout(train_ho, test_ho, best)
    print(f"\nHOLDOUT: features {last_feature_year}, target {last_feature_year + 1} "
          f"(trained on {len(train_ho):,} pairs, tested on {len(test_ho):,})")
    for method, s in holdout.items():
        print(_fmt(method, s))
    print(f"  95% interval, model minus persistence, median APE (points): {boot['median_ape_minus_persistence_pts']}")
    print(f"  95% interval, model/persistence RMSE ratio: {boot['rmse_ratio_to_persistence']}")

    # 3. Ship a model refit on every pair.
    final = fit_model(data, best)
    final.booster_.save_model(str(MODEL_PATH))
    importance = pd.Series(final.booster_.feature_importance("gain"), index=FEATURE_COLS)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    print(f"\nShipped model: {final.booster_.num_trees()} trees, refit on all {len(data):,} pairs.")
    print("Share of total gain by feature:")
    print(importance.round(3).to_string())

    METRICS_PATH.write_text(json.dumps({
        "pairs": int(len(data)),
        "feature_years": [int(data["year"].min()), last_feature_year],
        "target_years": [int(data["year"].min()) + 1, last_feature_year + 1],
        "selected_params": best,
        "inner_fold_years": INNER_FOLD_YEARS,
        "inner_folds": inner,
        "inner_fold_mean": inner_mean,
        "holdout": {"feature_year": last_feature_year, "target_year": last_feature_year + 1,
                    "train_pairs": int(len(train_ho)), **holdout, "bootstrap": boot},
        "grid_top5": grid_results[:5],
        "grid_size": len(grid_results),
        "gain_share": {k: float(v) for k, v in importance.items()},
        "shipped_trees": int(final.booster_.num_trees()),
    }, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and {METRICS_PATH.name}")
