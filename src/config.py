"""Shared paths and constants for the labor market pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OEWS_RAW = DATA_RAW / "oews"
JOLTS_RAW = DATA_RAW / "jolts"

BLS_API_KEY = os.getenv("BLS_API_KEY", "")
BLS_CONTACT_EMAIL = os.getenv("BLS_CONTACT_EMAIL", "")
BLS_USER_AGENT = f"labor-market-project ({BLS_CONTACT_EMAIL or 'no-contact-set'})"

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# OEWS annual "All Data" archives, published each May, one file per year.
# Source: https://www.bls.gov/oes/tables.htm -> special-requests/oesm{YY}all.zip
OEWS_YEARS = list(range(2015, 2025))  # 2015-2024
OEWS_ARCHIVE_URL = "https://www.bls.gov/oes/special-requests/oesm{yy}all.zip"

# Florida state FIPS code, used for JOLTS state-level series.
FL_STATE_FIPS = "12"

# OEWS 2-digit/range NAICS sector codes (national-level industry breakdown).
# Verified against the 2024 all_data file (i_group == "sector"); used as a
# direct NAICS-value filter rather than filtering on i_group because older
# release years (2015-2018) don't have an i_group column at all.
OEWS_SECTOR_NAICS_CODES = [
    "11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51", "52",
    "53", "54", "55", "56", "61", "62", "71", "72", "81", "99",
]

# NAICS industry codes used for industry-level JOLTS series (position 4-9 of
# a JTS series id). Verified against https://download.bls.gov/pub/time.series/jt/jt.industry
# "000000" = total nonfarm (all industries). Sub-sectors chosen for South Florida
# relevance (retail, healthcare, hospitality) in addition to the major groups.
JOLTS_INDUSTRY_SECTORS = {
    "000000": "Total nonfarm",
    "110099": "Mining and logging",
    "230000": "Construction",
    "300000": "Manufacturing",
    "400000": "Trade, transportation, and utilities",
    "440000": "Retail trade",
    "480099": "Transportation, warehousing, and utilities",
    "510000": "Information",
    "510099": "Financial activities",
    "540099": "Professional and business services",
    "600000": "Private education and health services",
    "620000": "Health care and social assistance",
    "700000": "Leisure and hospitality",
    "720000": "Accommodation and food services",
    "810000": "Other services",
    "900000": "Government",
}

# JOLTS data element / rate codes (position 19-21 of a JTS series id).
JOLTS_DATA_ELEMENTS = {
    "JOR": "Job openings rate",
    "HIR": "Hires rate",
    "TSR": "Total separations rate",
    "QUR": "Quits rate",
    "LDR": "Layoffs and discharges rate",
}
