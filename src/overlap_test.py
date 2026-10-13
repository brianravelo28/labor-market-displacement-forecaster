"""
How much of the forecaster's edge over "assume no change" could come from OEWS
releases overlapping?

Each OEWS release pools six semiannual survey panels collected over three
years, so consecutive releases share four of those six panels, releases two
years apart share two, and releases three years apart share none (BLS:
https://www.bls.gov/oes/oes_ques.htm). Next year's published employment is
therefore partly the same data as this year's. If the model's edge over the
persistence baseline were an artifact of that shared data, it should shrink as
the forecast horizon grows and the overlap disappears.

Test: forecast employment h = 1, 2, 3 years ahead (shared panels: 4, 2, 0),
from the same feature years, and compare model vs. persistence at each horizon.
  * A pair is features in year t, employment in year t + h.
  * Walk-forward as in train_model.py: to forecast from year T, train only on
    pairs whose target year is <= T (feature years <= T - h).
  * Feature years T = 2020, 2021, 2022 for every horizon, so the horizons are
    scored on the same starting points (h = 3 needs a known year T + 3 <= 2025).
  * Hyperparameters are fixed at the values selected for h = 1 (not re-tuned),
    so no horizon gets extra tuning.
  * The h = 3 training set is thinner (feature years 2016-2019 for T = 2022), a
    handicap for the model that the h = 1 rows don't have.

Pre-stated reading: if the model's advantage over persistence holds up at h = 3,
the edge is not just the overlap; if it vanishes while it exists at h = 1, the
overlap is a plausible source.

Writes data/processed/overlap_test.json.
"""
import json

import numpy as np
import pandas as pd

import train_model as tm
from config import DATA_PROCESSED

HORIZONS = [1, 2, 3]
FEATURE_YEARS = [2020, 2021, 2022]
SHARED_PANELS = {1: 4, 2: 2, 3: 0}
REPS = 2000


def fold_errors(data: pd.DataFrame, T: int, h: int, params: dict) -> dict:
    train, test = data[data["year"] <= T - h], data[data["year"] == T]
    model = tm.fit_model(train, params)
    y = test[tm.TARGET_COL].values
    now = test["employment_level"].values
    return {"y": y, "now": now, "pred": tm.predictions(model, test), "train_pairs": len(train)}


def summarize(y, now, pred) -> dict:
    ape_m = np.abs(pred - y) / y * 100
    ape_p = np.abs(now - y) / y * 100
    return {
        "n": int(len(y)),
        "model_median_error_pct": float(np.median(ape_m)),
        "persistence_median_error_pct": float(np.median(ape_p)),
        "model_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "persistence_rmse": float(np.sqrt(np.mean((now - y) ** 2))),
        "closer_than_persistence": float(np.mean(np.abs(pred - y) < np.abs(now - y))),
    }


def bootstrap(y, now, pred, reps=REPS) -> dict:
    rng = np.random.default_rng(0)
    d_med = np.empty(reps)
    ratio = np.empty(reps)
    for k in range(reps):
        i = rng.integers(0, len(y), len(y))
        d_med[k] = np.median(np.abs(pred[i] - y[i]) / y[i] * 100) - np.median(np.abs(now[i] - y[i]) / y[i] * 100)
        ratio[k] = np.sqrt(np.mean((pred[i] - y[i]) ** 2)) / np.sqrt(np.mean((now[i] - y[i]) ** 2))
    lo_hi = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"median_error_diff_pts_95ci": lo_hi(d_med), "rmse_ratio_95ci": lo_hi(ratio)}


def main():
    features = pd.read_csv(DATA_PROCESSED / "oews_occupation_features.csv")
    params = json.loads((DATA_PROCESSED / "model_metrics.json").read_text())["selected_params"]
    print(f"hyperparameters fixed at the h=1 selection: {params}\n")

    results = {}
    for h in HORIZONS:
        data = tm.build_supervised(features, horizon=h)
        folds, pooled = {}, {"y": [], "now": [], "pred": []}
        for T in FEATURE_YEARS:
            f = fold_errors(data, T, h, params)
            folds[str(T)] = {**summarize(f["y"], f["now"], f["pred"]), "train_pairs": f["train_pairs"]}
            for k in pooled:
                pooled[k].append(f[k])
        y, now, pred = (np.concatenate(pooled[k]) for k in ["y", "now", "pred"])
        results[str(h)] = {
            "shared_panels": SHARED_PANELS[h],
            "folds": folds,
            "pooled": summarize(y, now, pred),
            "pooled_bootstrap": bootstrap(y, now, pred),
        }
        s, b = results[str(h)]["pooled"], results[str(h)]["pooled_bootstrap"]
        print(f"h={h} ({SHARED_PANELS[h]} of 6 panels shared), pooled over feature years {FEATURE_YEARS}, n={s['n']:,}")
        print(f"   median error: model {s['model_median_error_pct']:.2f}%  persistence {s['persistence_median_error_pct']:.2f}%  "
              f"diff 95% CI {b['median_error_diff_pts_95ci'][0]:+.2f} to {b['median_error_diff_pts_95ci'][1]:+.2f} pts")
        print(f"   RMSE: model {s['model_rmse']:,.0f}  persistence {s['persistence_rmse']:,.0f}  ratio 95% CI "
              f"{b['rmse_ratio_95ci'][0]:.2f} to {b['rmse_ratio_95ci'][1]:.2f}   closer on {s['closer_than_persistence']:.0%}")
        for T, fr in folds.items():
            print(f"     T={T}: model {fr['model_median_error_pct']:.2f}% vs persistence {fr['persistence_median_error_pct']:.2f}%, "
                  f"RMSE {fr['model_rmse']:,.0f} vs {fr['persistence_rmse']:,.0f} (trained on {fr['train_pairs']:,} pairs)")

    (DATA_PROCESSED / "overlap_test.json").write_text(json.dumps(
        {"selected_params": params, "feature_years": FEATURE_YEARS, "horizons": results}, indent=2))


if __name__ == "__main__":
    main()
