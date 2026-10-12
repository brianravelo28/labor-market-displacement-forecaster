"""
Engineer YoY growth features for the national occupation x industry-sector
OEWS panel (oews_industry_panel.csv), for Tab 3 (Industry & Sector Analysis).

Same annual-cadence caveat as build_features.py: "YoY" here means between
consecutive OEWS annual releases, not a rolling 12-month window.
"""
import pandas as pd

from build_features import lagged_values
from build_macro_features import add_real_wage_features, add_sector_jolts
from config import DATA_PROCESSED

CONTRACTION_THRESHOLD_PCT = -2.0
KEYS = ["naics", "occ_code"]


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "oews_industry_panel.csv", dtype={"naics": str})
    return df.sort_values(KEYS + ["year"]).reset_index(drop=True)


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Matched on the actual prior year, not the previous row (see lagged_values).
    prev = lagged_values(df, KEYS, ["employment_level", "wage_annual_mean"], 1)
    df["employment_change_yoy_pct"] = (df["employment_level"] - prev["employment_level"]) / prev["employment_level"] * 100
    df["wage_growth_yoy_pct"] = (df["wage_annual_mean"] - prev["wage_annual_mean"]) / prev["wage_annual_mean"] * 100

    df["contraction_flag_12mo"] = (df["employment_change_yoy_pct"] < CONTRACTION_THRESHOLD_PCT).astype("Int64")
    df.loc[df["employment_change_yoy_pct"].isna(), "contraction_flag_12mo"] = pd.NA

    return df


if __name__ == "__main__":
    panel = load_panel()
    featured = add_real_wage_features(add_sector_jolts(add_growth_features(panel)))
    out_path = DATA_PROCESSED / "oews_industry_features.csv"
    featured.to_csv(out_path, index=False)
    print(f"Saved {len(featured):,} rows to {out_path}")
