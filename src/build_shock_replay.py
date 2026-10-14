"""
How each occupation responded to the 2020 shock, for the 2020 Shock & Recovery tab.

Nothing here is a fitted elasticity. It is an observed episode: the change in an
occupation's employment in the 2020 OEWS release, minus how fast the same
occupation had been growing in the releases before it, so only the part that is
plausibly the shock remains. It is a description of what happened, not a forecast.

Reliability rule: a single occupation's year-over-year change is mostly
sampling noise below about 10,000 employees (the standard deviation of annual
growth is roughly 59 points under 1,000 employees vs. 12 above 10,000), so
occupations that small borrow the response of their SOC major group instead
(the group's total employment change, same adjustment). The `source` column
says which was used.

Caveat built into the data: each OEWS release pools three years of survey
panels, so the 2020 release shows the shock smeared and the rebound shows up
mostly in the 2022 release. The response covers the 2020 release only.

Writes data/processed/shock_2020_response.csv.
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED

SHOCK_YEAR = 2020
PRE_YEARS = range(2016, SHOCK_YEAR)       # releases used for the "normal" growth rate
MIN_PRE_YEARS = 2                         # need at least this many pre-shock changes
MIN_OWN_EMPLOYMENT = 10_000               # in both 2019 and 2020, to trust an occupation's own response


def _pairs(features: pd.DataFrame) -> pd.DataFrame:
    """Rows with a valid year-over-year change, plus the prior year's employment."""
    f = features[["region", "occ_code", "year", "employment_level", "employment_change_yoy_pct"]].dropna().copy()
    f["prior_employment"] = f["employment_level"] / (1 + f["employment_change_yoy_pct"] / 100)
    f["group"] = f["occ_code"].str[:2]
    return f


def own_response(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (region, occupation): 2020 growth minus mean pre-2020 growth, if the occupation is large enough."""
    shock = pairs[pairs["year"] == SHOCK_YEAR].set_index(["region", "occ_code"])
    pre = pairs[pairs["year"].isin(PRE_YEARS)].groupby(["region", "occ_code"])["employment_change_yoy_pct"].agg(["mean", "count"])
    out = shock[["employment_change_yoy_pct", "employment_level", "prior_employment"]].join(pre, how="left")
    big = np.minimum(out["employment_level"], out["prior_employment"]) >= MIN_OWN_EMPLOYMENT
    ok = big & (out["count"] >= MIN_PRE_YEARS)
    out["excess_pct"] = np.where(ok, out["employment_change_yoy_pct"] - out["mean"], np.nan)
    return out[["excess_pct"]].dropna()


def group_response(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (region, SOC major group): total employment change in 2020 minus its mean change in earlier releases."""
    totals = pairs.groupby(["region", "group", "year"]).agg(now=("employment_level", "sum"), prior=("prior_employment", "sum"))
    totals["growth"] = (totals["now"] / totals["prior"] - 1) * 100
    g = totals["growth"].unstack("year")
    if SHOCK_YEAR not in g.columns:        # no occupation in any group has a 2020 change
        return pd.DataFrame(columns=["region", "group", "group_excess_pct"])
    pre_cols = [y for y in PRE_YEARS if y in g.columns]
    pre = g[pre_cols]
    ok = pre.notna().sum(axis=1) >= MIN_PRE_YEARS
    excess = (g[SHOCK_YEAR] - pre.mean(axis=1)).where(ok)
    return excess.rename("group_excess_pct").dropna().reset_index()


def build(features: pd.DataFrame) -> pd.DataFrame:
    pairs = _pairs(features)
    own = own_response(pairs).reset_index()
    grp = group_response(pairs)

    universe = features[["region", "occ_code"]].drop_duplicates().copy()
    universe["group"] = universe["occ_code"].str[:2]
    out = universe.merge(own, on=["region", "occ_code"], how="left").merge(grp, on=["region", "group"], how="left")
    use_own = out["excess_pct"].notna()
    out["response_pct"] = out["excess_pct"].where(use_own, out["group_excess_pct"])
    out["source"] = np.where(use_own, "occupation", np.where(out["group_excess_pct"].notna(), "group", "none"))
    return out[["region", "occ_code", "response_pct", "source"]].sort_values(["region", "occ_code"]).reset_index(drop=True)


if __name__ == "__main__":
    features = pd.read_csv(DATA_PROCESSED / "oews_occupation_features.csv")
    result = build(features)
    result.to_csv(DATA_PROCESSED / "shock_2020_response.csv", index=False)
    print(f"Saved {len(result):,} rows")
    print(result.groupby(["region", "source"]).size().unstack(fill_value=0))
    print("\nresponse (percentage points of growth, 2020 excess over earlier trend):")
    print(result.groupby("region")["response_pct"].describe().round(1).to_string())
    pairs = _pairs(features)
    for region, p in pairs.groupby("region"):
        us = p[p["year"] == SHOCK_YEAR]
        print(f"{region}: total employment change in the {SHOCK_YEAR} release "
              f"{(us['employment_level'].sum() / us['prior_employment'].sum() - 1) * 100:+.1f}%")
