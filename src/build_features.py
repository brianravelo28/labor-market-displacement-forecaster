"""
Engineer growth/trend features from the OEWS annual occupation panel.

Cadence is annual (see README.md), so "12mo" features here are really
year-over-year between consecutive OEWS releases, and "5yr" features compare
against the release exactly 5 years prior when that occupation code is
present in both years (SOC code revisions mean not every occupation has a
continuous 10-year history under the same code).

JOLTS-derived (vacancy/separation rate) and CPI-derived (inflation-adjusted
wage) features are added in a later merge step once those fetches complete --
they're independent of this script and not required to compute the
employment/wage growth features here.
"""
import pandas as pd

from config import DATA_PROCESSED

CONTRACTION_THRESHOLD_PCT = -2.0


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "oews_occupation_panel.csv")
    return df.sort_values(["region", "occ_code", "year"]).reset_index(drop=True)


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["region", "occ_code"])

    emp_prev = grp["employment_level"].shift(1)
    wage_prev = grp["wage_annual_mean"].shift(1)
    df["employment_change_yoy_pct"] = (df["employment_level"] - emp_prev) / emp_prev * 100
    df["wage_growth_yoy_pct"] = (df["wage_annual_mean"] - wage_prev) / wage_prev * 100

    emp_5yr_ago = grp["employment_level"].shift(5)
    wage_5yr_ago = grp["wage_annual_mean"].shift(5)
    df["employment_change_5yr_cagr"] = (df["employment_level"] / emp_5yr_ago) ** (1 / 5) - 1
    df["wage_growth_5yr_cagr"] = (df["wage_annual_mean"] / wage_5yr_ago) ** (1 / 5) - 1

    df["contraction_flag_12mo"] = (df["employment_change_yoy_pct"] < CONTRACTION_THRESHOLD_PCT).astype("Int64")
    df.loc[df["employment_change_yoy_pct"].isna(), "contraction_flag_12mo"] = pd.NA

    return df


if __name__ == "__main__":
    panel = load_panel()
    featured = add_growth_features(panel)
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
