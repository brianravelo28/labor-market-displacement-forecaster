"""
LightGBM employment forecaster: predicts next-year employment_level per
occupation x region from the current year's OEWS-derived features.

"12-months-ahead" in the original brief becomes "next annual release" here,
since OEWS is annual (see README.md). Train/val/test split mirrors the
original 2015-2021 / 2022 / 2023-2024 intent, adapted to annual granularity:
  - train: feature year 2015-2020 -> target is employment_level in year+1
  - val:   feature year 2021      -> target is employment_level in 2022
  - test:  feature year 2022-2023 -> target is employment_level in year+1

JOLTS (vacancy/separation rate) and CPI (real wage) features are not yet
available (blocked on a BLS API outage) -- this model runs on OEWS-only
features and will be retrained once those land.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import DATA_PROCESSED

FEATURE_COLS = [
    "employment_level",
    "wage_annual_mean",
    "wage_annual_median",
    "employment_change_yoy_pct",
    "wage_growth_yoy_pct",
    "employment_change_5yr_cagr",
    "wage_growth_5yr_cagr",
    "contraction_flag_12mo",
    "region",
    "occ_major_group",
]
CATEGORICAL_COLS = ["region", "occ_major_group"]
TARGET_COL = "employment_level_next"

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=32,
    min_data_in_leaf=20,
    feature_fraction=0.75,
    bagging_freq=1,
    bagging_fraction=0.9,
    verbosity=-1,
)


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["occ_major_group"] = df["occ_code"].str[:2].astype("category")
    df["region"] = df["region"].astype("category")

    df = df.sort_values(["region", "occ_code", "year"])
    grp = df.groupby(["region", "occ_code"], observed=True)
    df[TARGET_COL] = grp["employment_level"].shift(-1)
    df["target_year"] = df["year"] + 1

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).copy()
    # A log1p(level) target still fails badly at the extreme high end: only 7
    # of 1276 training rows have employment > 2M (median ~14,000), so
    # LightGBM has almost no signal there and mispredicts even *training*
    # rows for the largest occupations by 40-60% (verified: Retail
    # Salespersons, the single largest US occupation, predicted ~2.1M vs
    # actual 3.7M on a row it was trained on). Predicting the log growth
    # ratio instead of the absolute level sidesteps this: growth rates for a
    # 50-employee occupation and a 3.8M-employee occupation live on the same
    # numeric scale, so the extreme tail no longer starves the model of
    # comparable training signal. Level is reconstructed as
    # employment_level * exp(predicted_log_growth).
    df["target_log_growth"] = np.log(df[TARGET_COL]) - np.log(df["employment_level"])
    return df


def split(df: pd.DataFrame):
    train = df[df["year"] <= 2020]
    val = df[df["year"] == 2021]
    test = df[df["year"].isin([2022, 2023])]
    return train, val, test


def evaluate(model, data, label):
    X, y = data[FEATURE_COLS], data[TARGET_COL]
    preds = data["employment_level"].values * np.exp(model.predict(X))
    ape = (preds - y).abs() / y * 100

    rmse = mean_squared_error(y, preds) ** 0.5
    mape = mean_absolute_percentage_error(y, preds) * 100

    large = data["employment_level"] >= 1_000_000
    rmse_large = mean_squared_error(y[large], preds[large]) ** 0.5 if large.sum() else float("nan")

    print(
        f"{label:12s} n={len(data):5d}  RMSE={rmse:12,.0f}  MAPE={mape:6.1f}%  "
        f"median APE={ape.median():5.1f}%  RMSE(1M+ occ, n={large.sum()})={rmse_large:12,.0f}"
    )
    return preds


if __name__ == "__main__":
    features = pd.read_csv(DATA_PROCESSED / "oews_occupation_features.csv")
    supervised = build_supervised(features)
    train, val, test = split(supervised)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        train[FEATURE_COLS],
        train["target_log_growth"],
        categorical_feature=CATEGORICAL_COLS,
        eval_set=[(val[FEATURE_COLS], val["target_log_growth"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    evaluate(model, train, "train")
    evaluate(model, val, "val")
    evaluate(model, test, "test")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance.to_string())

    model.booster_.save_model(str(DATA_PROCESSED / "employment_forecast_model.txt"))
    print(f"\nModel saved to {DATA_PROCESSED / 'employment_forecast_model.txt'}")
