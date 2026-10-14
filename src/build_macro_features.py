"""
Macro features from CPI-U (inflation-adjusted wages) and JOLTS (labor-market
tightness), merged onto the OEWS occupation and industry tables.

CPI: OEWS wages describe a May reference period, so each year is deflated with
that year's May CPI-U and expressed in dollars of the latest OEWS year. Adding
a newer OEWS year changes the base, so rerun the feature builders afterwards.

JOLTS: published by industry only, never by occupation. To get an occupation
number, each occupation's rate is the average of its industries' rates weighted
by where that occupation's employment sits (from the national occupation x
sector OEWS panel). That is an exposure measure, not an occupation-specific
vacancy rate. Florida occupations use the same national sector rates, because
BLS publishes Florida JOLTS for total nonfarm only; the Florida total-nonfarm
series enters separately as a regional-tightness feature.

Each JOLTS annual figure is the mean of the available months in the OEWS
reference calendar year (data through December of that year).
"""
from typing import Optional

import pandas as pd

from config import DATA_PROCESSED

# OEWS NAICS sector -> the closest JOLTS industry in config.JOLTS_INDUSTRY_SECTORS.
# Approximate: wholesale trade has no series of its own here, so it takes the
# parent "trade, transportation, and utilities" rate; agriculture (11) is
# outside JOLTS' nonfarm scope and gets none.
SECTOR_TO_JOLTS = {
    "21": "110099", "22": "480099", "23": "230000", "31-33": "300000",
    "42": "400000", "44-45": "440000", "48-49": "480099", "51": "510000",
    "52": "510099", "53": "510099", "54": "540099", "55": "540099",
    "56": "540099", "61": "600000", "62": "620000", "71": "700000",
    "72": "720000", "81": "810000", "99": "900000",
}
JOLTS_ELEMENTS = {  # element name in jolts_industry_panel.csv -> short column suffix
    "Job openings rate": "openings",
    "Quits rate": "quits",
    "Layoffs and discharges rate": "layoffs",
}


def load_cpi_may() -> pd.Series:
    cpi = pd.read_csv(DATA_PROCESSED / "cpi_monthly.csv")
    return cpi[cpi["month"] == 5].set_index("year")["cpi"]


def add_real_wage_features(df: pd.DataFrame, base_year: Optional[int] = None) -> pd.DataFrame:
    """Adds wage_annual_mean_real (base-year May dollars), real_wage_growth_yoy_pct
    and purchasing_power_loss_flag (nominal wage up, real wage down)."""
    df = df.copy()
    cpi = load_cpi_may()
    base_year = base_year or int(df["year"].max())
    df["wage_annual_mean_real"] = df["wage_annual_mean"] * cpi[base_year] / df["year"].map(cpi)

    prev_cpi = (df["year"] - 1).map(cpi)
    df["real_wage_growth_yoy_pct"] = ((1 + df["wage_growth_yoy_pct"] / 100) * prev_cpi / df["year"].map(cpi) - 1) * 100
    loss = (df["wage_growth_yoy_pct"] > 0) & (df["real_wage_growth_yoy_pct"] < 0)
    df["purchasing_power_loss_flag"] = loss.astype("Int64")
    df.loc[df["real_wage_growth_yoy_pct"].isna(), "purchasing_power_loss_flag"] = pd.NA
    return df


def annual_jolts(region: str = "US") -> pd.DataFrame:
    """Mean monthly rate per (year, industry_code, element) for one region."""
    j = pd.read_csv(DATA_PROCESSED / "jolts_industry_panel.csv", dtype={"industry_code": str})
    j = j[(j["region"] == region) & j["data_element"].isin(JOLTS_ELEMENTS)]
    out = j.groupby(["year", "industry_code", "data_element"])["rate"].mean().reset_index()
    out["suffix"] = out["data_element"].map(JOLTS_ELEMENTS)
    return out


def add_sector_jolts(industry_df: pd.DataFrame) -> pd.DataFrame:
    """Each occupation x sector row gets its sector's JOLTS annual rates (jolts_*_rate)."""
    df = industry_df.copy()
    code = df["naics"].map(SECTOR_TO_JOLTS)
    rates = annual_jolts("US")
    for suffix, g in rates.groupby("suffix"):
        lookup = g.set_index(["year", "industry_code"])["rate"]
        idx = pd.MultiIndex.from_arrays([df["year"], code])
        df[f"jolts_{suffix}_rate"] = lookup.reindex(idx).to_numpy()
    return df


def add_occupation_jolts(occ_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    """Employment-weighted JOLTS exposure per (year, occupation), plus the region's
    own total-nonfarm openings rate (FL series for FL rows, US for US rows)."""
    df = occ_df.copy()
    sect = add_sector_jolts(industry_df)
    rate_cols = [f"jolts_{s}_rate" for s in JOLTS_ELEMENTS.values()]
    sect = sect.dropna(subset=["employment_level"] + rate_cols)
    w = sect["employment_level"]
    weighted = sect[["year", "occ_code"]].copy()
    for c in rate_cols:
        weighted[c] = sect[c] * w
    weighted["w"] = w
    agg = weighted.groupby(["year", "occ_code"]).sum()
    exposure = agg[rate_cols].div(agg["w"], axis=0).rename(columns={c: c.replace("jolts_", "jolts_exposure_") for c in rate_cols})
    df = df.merge(exposure.reset_index(), on=["year", "occ_code"], how="left")

    totals = []
    for region in ["US", "FL"]:
        t = annual_jolts(region)
        t = t[(t["industry_code"] == "000000") & (t["suffix"] == "openings")]
        totals.append(pd.DataFrame({"year": t["year"], "region": region, "jolts_region_openings_rate": t["rate"]}))
    return df.merge(pd.concat(totals), on=["year", "region"], how="left")
