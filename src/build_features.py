"""
Engineer growth/trend features from the OEWS annual occupation panel.

Cadence is annual (see README.md), so "12mo" features here are really
year-over-year between consecutive OEWS releases, and "5yr" features compare
against the release exactly 5 years prior when that occupation code is
present in both years (SOC code revisions mean not every occupation has a
continuous multi-year history under the same code).

When run as a script, the CPI-derived (inflation-adjusted wage) and
JOLTS-derived (job-market tightness) columns from build_macro_features.py are
merged in after the growth features are computed.
"""
import pandas as pd

from config import DATA_PROCESSED

CONTRACTION_THRESHOLD_PCT = -2.0


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "oews_occupation_panel.csv")
    return df.sort_values(["region", "occ_code", "year"]).reset_index(drop=True)


KEYS = ["region", "occ_code"]


def lagged_values(df: pd.DataFrame, keys: list, cols: list, lag: int) -> pd.DataFrame:
    """Each column's value from exactly `lag` years earlier, NaN where that year is absent.

    A positional groupby().shift(lag) silently compares the previous *row*, so
    an occupation missing a year (SOC code changes, suppressed cells) would get
    a "year-over-year" change that really spans several years. Joining on
    year + lag avoids that.
    """
    assert not df.duplicated(keys + ["year"]).any(), "panel must have one row per key and year"
    prev = df[keys + ["year"] + cols].copy()
    prev["year"] = prev["year"] + lag
    out = df[keys + ["year"]].merge(prev, on=keys + ["year"], how="left", suffixes=("", "_lag"))
    return out[cols].set_axis(df.index)


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = ["employment_level", "wage_annual_mean"]

    prev = lagged_values(df, KEYS, cols, 1)
    df["employment_change_yoy_pct"] = (df["employment_level"] - prev["employment_level"]) / prev["employment_level"] * 100
    df["wage_growth_yoy_pct"] = (df["wage_annual_mean"] - prev["wage_annual_mean"]) / prev["wage_annual_mean"] * 100

    ago5 = lagged_values(df, KEYS, cols, 5)
    df["employment_change_5yr_cagr"] = (df["employment_level"] / ago5["employment_level"]) ** (1 / 5) - 1
    df["wage_growth_5yr_cagr"] = (df["wage_annual_mean"] / ago5["wage_annual_mean"]) ** (1 / 5) - 1

    df["contraction_flag_12mo"] = (df["employment_change_yoy_pct"] < CONTRACTION_THRESHOLD_PCT).astype("Int64")
    df.loc[df["employment_change_yoy_pct"].isna(), "contraction_flag_12mo"] = pd.NA

    return df


if __name__ == "__main__":
    from build_macro_features import add_occupation_jolts, add_real_wage_features

    panel = load_panel()
    featured = add_growth_features(panel)
    featured = add_real_wage_features(featured)
    industry_panel = pd.read_csv(DATA_PROCESSED / "oews_industry_panel.csv", dtype={"naics": str})
    featured = add_occupation_jolts(featured, industry_panel)
    out_path = DATA_PROCESSED / "oews_occupation_features.csv"
    featured.to_csv(out_path, index=False)
    print(f"Saved {len(featured):,} rows to {out_path}")

    latest = featured[featured["year"] == featured["year"].max()]
    print(f"\n{latest['year'].iloc[0]} snapshot ({len(latest)} occupation-region rows):")
    print("  with YoY growth computed:", latest["employment_change_yoy_pct"].notna().sum())
    print("  with 5yr CAGR computed:  ", latest["employment_change_5yr_cagr"].notna().sum())
    print("  contraction-flagged:     ", (latest["contraction_flag_12mo"] == 1).sum())

    top_contracting = (
        latest[latest["region"] == "US"]
        .nsmallest(10, "employment_change_yoy_pct")
        [["occ_title", "employment_level", "employment_change_yoy_pct", "wage_growth_yoy_pct"]]
    )
    print("\nTop 10 contracting US occupations (most recent year):")
    print(top_contracting.to_string(index=False))
